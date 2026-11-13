"""本地 LLM 加载与推理封装。

支持：
- HuggingFace transformers pipeline
- bitsandbytes 4bit 量化（NF4 / FP4）
- 设备自动分配（device_map="auto"）
- 流式输出（generate stream）
- 简单 chat template（Qwen / ChatML）
- 单条 / 批量推理
- 超时控制与重试机制

针对 GTX 1060 4GB：建议使用 4bit 量化 + 小模型（Qwen2.5-1.5B-Instruct）。
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from .llm_concurrency import get_llm_call_gate
from .utils import Timer, get_logger, resolve_path, select_torch_device

logger = get_logger("llm")

# 默认 LLM 推理超时（秒）
DEFAULT_TIMEOUT = 120.0
# 最大重试次数
DEFAULT_MAX_RETRIES = 2

# 超时控制用的共享线程池。线程无法被强制中断，因此超时后放弃的是
# "等待结果"，worker 线程会随解释器退出（daemon）。池只按需创建。
_TIMEOUT_POOL: Optional[ThreadPoolExecutor] = None


def _detect_device(device: str) -> str:
    """委托到 :func:`src.utils.select_torch_device`，与 embeddings/reranker
    行为一致（CPU 版 torch + CUDA 驱动时回落 cpu）。"""
    return select_torch_device(device, logger=logger)


def _detect_dtype(torch_dtype: str):
    """解析 torch_dtype 字符串为 torch 类型。"""
    import torch  # type: ignore

    if torch_dtype in ("", "auto", None):
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(torch_dtype.lower(), "auto")


def _get_timeout_pool() -> ThreadPoolExecutor:
    """按需创建超时控制线程池。

    worker 数与 LLM 并发上限对齐（``LLM_MAX_CONCURRENT``，默认 4）：如果池
    比 LLM 并发小，后续请求会在队列里等待，而 ``future.result(timeout)``
    把排队时间也计入超时，导致"本可完成的请求被误判超时"。池足够大时，
    timeout 基本只度量生成本身。
    """
    global _TIMEOUT_POOL
    if _TIMEOUT_POOL is None:
        max_workers = max(
            2,
            int(os.getenv("LLM_TIMEOUT_POOL_WORKERS", os.getenv("LLM_MAX_CONCURRENT", "4"))),
        )
        _TIMEOUT_POOL = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="llm-timeout"
        )
    return _TIMEOUT_POOL


@dataclass
class GenerationConfig:
    """生成参数容器。"""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    repetition_penalty: float = 1.05
    do_sample: bool = True


class LocalLLM:
    """本地大语言模型封装（HuggingFace transformers）。

    Args:
        model_name: HF 模型名或本地路径。
        device: auto / cpu / cuda。
        device_map: auto / balanced / cuda:0 / cpu。
        torch_dtype: auto / float16 / bfloat16 / float32。
        quant: 量化配置 dict（含 enabled、quant_type、double_quant、compute_dtype）。
        cache_dir: HF 缓存目录。
        generation: 生成参数 dict 或 GenerationConfig 实例。
        chat_template: auto / qwen / chatml / raw。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        quant: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        generation: Optional[Dict[str, Any] | GenerationConfig] = None,
        chat_template: str = "auto",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.model_name = model_name
        self.device = _detect_device(device)
        self.device_map = device_map
        self.torch_dtype = _detect_dtype(torch_dtype)
        self.quant = quant or {"enabled": False}
        self.chat_template = chat_template
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.max_retries = max_retries
        self._active_generate = 0
        self._active_lock = threading.Lock()
        self._stop_states = {}
        self._next_generation_counter = 0

        # 解析缓存目录
        self.cache_dir = resolve_path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(self.cache_dir))

        # 解析生成参数
        if isinstance(generation, GenerationConfig):
            self.gen_cfg = generation
        elif isinstance(generation, dict):
            self.gen_cfg = GenerationConfig(
                **{k: v for k, v in generation.items() if k in GenerationConfig.__annotations__}
            )
        else:
            self.gen_cfg = GenerationConfig()

        self._load_model()
        self._load_tokenizer()

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore

        # device_map 支持 JSON dict 字符串（如 '{"": 0}'）——强制全部层进
        # 第一块 GPU，绕开 accelerate 在小显存卡上的保守卸载策略
        device_map_kw: Any = self.device_map
        if isinstance(self.device_map, str) and self.device_map.strip().startswith("{"):
            try:
                device_map_kw = json.loads(self.device_map)
            except ValueError:
                pass
        base_kwargs: Dict[str, Any] = {
            "device_map": device_map_kw,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }

        quant_exc: Optional[Exception] = None
        if self.quant and self.quant.get("enabled"):
            try:
                from transformers import BitsAndBytesConfig  # type: ignore

                compute_dtype = self.quant.get("compute_dtype", "float16")
                compute_dtype = _detect_dtype(compute_dtype)
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.quant.get("quant_type", "nf4"),
                    bnb_4bit_use_double_quant=bool(self.quant.get("double_quant", True)),
                    bnb_4bit_compute_dtype=compute_dtype if compute_dtype != "auto" else torch.float16,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                quant_kwargs = {**base_kwargs, "quantization_config": bnb_cfg, "device_map": "auto"}
                with Timer(f"加载 LLM {self.model_name} (4bit)"):
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **quant_kwargs)
                logger.info("已启用 4bit 量化：%s", self.quant)
            except Exception as exc:
                # 量化加载失败（bitsandbytes 未装/不支持当前 CUDA/版本不兼容等）
                # 回退到非量化 fp16/fp32 加载，保证功能可用
                quant_exc = exc
                logger.warning(
                    "4bit 量化加载失败（%s: %s），回退到非量化加载",
                    type(exc).__name__, exc,
                )

        if not (self.quant and self.quant.get("enabled")) or quant_exc is not None:
            dtype = self.torch_dtype
            fallback_kwargs = {
                **base_kwargs,
                "torch_dtype": dtype if dtype != "auto" else (torch.float16 if self.device == "cuda" else torch.float32),
            }
            with Timer(f"加载 LLM {self.model_name}"):
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **fallback_kwargs)
        self.model.eval()
        # 设置 pad token id（生成时需要 eos/pad 区分）
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def _load_tokenizer(self) -> None:
        from transformers import AutoTokenizer  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            use_fast=True,
            local_files_only=self.local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """根据 chat_template 构造 prompt。"""
        ct = self.chat_template
        if ct == "auto":
            ct = "qwen" if "qwen" in self.model_name.lower() else "chatml"

        if ct == "raw":
            # 直接拼接，不加特殊模板
            parts = []
            for m in messages:
                role = m.get("role", "user")
                parts.append(f"{role}: {m.get('content','')}")
            return "\n".join(parts) + "\nassistant:"

        # 默认使用 tokenizer 的 chat template
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # 回退到 ChatML
            parts = []
            for m in messages:
                role = m.get("role", "user")
                parts.append(f"<|im_start|>{role}\n{m.get('content','')}<|im_end|>\n")
            parts.append("<|im_start|>assistant\n")
            return "".join(parts)

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        generation: Optional[GenerationConfig] = None,
        stream: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> str | Generator[str, None, None]:
        """对话式生成。

        Args:
            messages: OpenAI 风格消息列表 [{"role":..., "content":...}]。
            generation: 覆盖默认生成参数。
            stream: 是否流式返回（str 生成器）。
            timeout: 同步调用超时秒数（仅非流式生效）。

        Returns:
            非流式：完整回答；流式：逐片段生成器。
        """
        prompt = self._build_prompt(messages)
        gen_cfg = generation or self.gen_cfg
        if stream:
            return self._stream_generate(prompt, gen_cfg)
        return self._generate_with_timeout(prompt, gen_cfg, timeout)

    def _generate_with_retry(self, prompt: str, gen_cfg: GenerationConfig) -> str:
        """带重试的生成方法，处理临时性失败。"""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._generate(prompt, gen_cfg)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    wait_time = (attempt + 1) * 1.0  # 简单指数退避
                    logger.warning("LLM 生成失败（第 %d 次），%.1fs 后重试: %s", attempt + 1, wait_time, exc)
                    time.sleep(wait_time)
                else:
                    logger.error("LLM 生成最终失败（已重试 %d 次）: %s", self.max_retries, exc)
        raise last_error

    def _generate_with_timeout(self, prompt: str, gen_cfg: GenerationConfig, timeout: float) -> str:
        """带超时的生成方法。

        P1-5 修复：超时度量不准确
        - 问题：future.result(timeout) 把排队时间计入超时
        - 方案：在任务开始时记录时刻，只度量真正的生成耗时
        
        说明：之前这里用 ``signal.alarm/SIGALRM`` 实现，但 Windows 上根本没有
        ``alarm``，且 signal 处理器只能在主线程生效（FastAPI 的同步端点跑在
        线程池里），因此跨平台一律改为线程池 + ``future.result(timeout)``。
        超时后放弃等待并抛出 :class:`TimeoutError`；推理线程本身无法被强制
        中断，作为 daemon 线程随进程退出。
        """
        gate = get_llm_call_gate()
        pool = _get_timeout_pool()
        
        # P1-5 修复：使用队列传递开始时间，排除排队延迟
        import queue
        start_queue: queue.Queue = queue.Queue(maxsize=1)

        def _run(generation_id: int) -> str:
            # 任务真正开始时记录时刻
            start_queue.put(time.time())
            return self._generate_with_retry(prompt, gen_cfg)

        submit_time = time.time()
        generation_id = self._register_generation()
        future = pool.submit(_run, generation_id)
        stop_flag = _TimeoutStopFlag(self, generation_id)
        
        try:
            # 等待任务开始（最多等 timeout）
            try:
                actual_start = start_queue.get(timeout=timeout)
                # 计算剩余超时时间（排除排队时间）
                elapsed_queue = actual_start - submit_time
                remaining_timeout = max(0.1, timeout - elapsed_queue)
            except queue.Empty:
                # 排队超时
                future.cancel()
                logger.warning("LLM 任务排队超时（%.1f 秒）", timeout)
                raise TimeoutError(f"LLM 任务排队超时（{timeout}s）") from None
            
            # 等待实际生成完成（只计生成时间），同时持有全局 LLM 并发闸。
            with gate.blocking(timeout=remaining_timeout):
                self._increment_active_generate()
                try:
                    return future.result(timeout=remaining_timeout)
                finally:
                    stop_flag.request_stop()
                    self._unregister_generation(generation_id)
                    self._decrement_active_generate()
        except FuturesTimeoutError:
            future.cancel()
            logger.warning("LLM 生成超时（%.1f 秒）", timeout)
            raise TimeoutError(f"LLM 生成超时（{timeout}s）") from None

    def _generate(self, prompt: str, gen_cfg: GenerationConfig) -> str:
        import torch  # type: ignore

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_id = self._register_generation()
        with Timer("llm generate"):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_cfg.max_new_tokens,
                    temperature=gen_cfg.temperature,
                    top_p=gen_cfg.top_p,
                    repetition_penalty=gen_cfg.repetition_penalty,
                    do_sample=gen_cfg.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    stopping_criteria=_ActiveStopCriteria(self, generation_id),
                )
        new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        self._unregister_generation(generation_id)
        return text.strip()

    def _stream_generate(self, prompt: str, gen_cfg: GenerationConfig) -> Generator[str, None, None]:
        """基于 TextIteratorStreamer 的流式输出。"""
        from threading import Thread

        import torch  # type: ignore
        from transformers import TextIteratorStreamer  # type: ignore

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_id = self._register_generation()
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            repetition_penalty=gen_cfg.repetition_penalty,
            do_sample=gen_cfg.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
            stopping_criteria=_ActiveStopCriteria(self, generation_id),
        )

        def _worker():
            # generate 抛异常（OOM 等）时 TextIteratorStreamer 永远收不到
            # 结束信号，消费端 `for piece in streamer` 会无限阻塞——必须
            # 在 finally 里补发 end()，保证迭代器一定终止。
            try:
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.error("流式生成失败: %s", exc)
            finally:
                try:
                    streamer.end()
                except Exception:  # noqa: BLE001
                    pass

        th = Thread(target=_worker, daemon=True)
        th.start()
        try:
            for piece in streamer:
                yield piece
        finally:
            self.request_stop_generation(generation_id)
            self._unregister_generation(generation_id)
            th.join(timeout=5)

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_memory_gb(model_name: str, quant: bool) -> float:
        """粗略估计模型显存占用（GB），仅用于 UI 提示。"""
        n = model_name.lower()
        size_map = {
            "0.5b": 1.0,
            "1.5b": 3.0,
            "2b": 4.0,
            "3b": 6.0,
            "7b": 14.0,
        }
        for tag, gb in size_map.items():
            if tag in n:
                return round(gb * 0.4 if quant else gb, 2)
        return 4.0 if quant else 8.0

    def _increment_active_generate(self) -> None:
        with self._active_lock:
            self._active_generate += 1

    def _decrement_active_generate(self) -> None:
        with self._active_lock:
            self._active_generate = max(0, self._active_generate - 1)

    def _next_generation_id(self) -> int:
        with self._active_lock:
            self._next_generation_counter += 1
            return self._next_generation_counter

    def _register_generation(self) -> int:
        generation_id = self._next_generation_id()
        return generation_id

    def _unregister_generation(self, generation_id: int) -> None:
        with self._active_lock:
            self._stop_states.pop(generation_id, None)

    def request_stop_generation(self, generation_id: int) -> None:
        state = self._stop_states.get(generation_id)
        if state is not None:
            state.request_stop()

    def active_generate_count(self) -> int:
        with self._active_lock:
            return self._active_generate


class _ActiveStopState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stopped = False

    def request_stop(self) -> None:
        with self._lock:
            self._stopped = True

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped


class _ActiveStopCriteria:
    """Transformers StoppingCriteria，超时后请求当前 generate 停止。"""

    def __init__(self, owner: LocalLLM, generation_id: int) -> None:
        self.owner = owner
        self._state = _ActiveStopState()
        self._generation_id = generation_id
        with owner._active_lock:
            owner._stop_states[generation_id] = self._state

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._state.stopped


class _TimeoutStopFlag:
    """把超时事件转发给最近一次 generate 的停止条件。"""

    def __init__(self, owner: LocalLLM, generation_id: int) -> None:
        self.owner = owner
        self.generation_id = generation_id

    def request_stop(self) -> None:
        self.owner.request_stop_generation(self.generation_id)
