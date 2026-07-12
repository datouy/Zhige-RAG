"""本地 LLM 加载与推理封装。

支持：
- HuggingFace transformers pipeline
- bitsandbytes 4bit 量化（NF4 / FP4）
- 设备自动分配（device_map="auto"）
- 流式输出（generate stream）
- 简单 chat template（Qwen / ChatML）
- 单条 / 批量推理

针对 GTX 1060 4GB：建议使用 4bit 量化 + 小模型（Qwen2.5-1.5B-Instruct）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from .utils import Timer, get_logger, resolve_path

logger = get_logger("llm")


def _detect_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


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
    ) -> None:
        self.model_name = model_name
        self.device = _detect_device(device)
        self.device_map = device_map
        self.torch_dtype = _detect_dtype(torch_dtype)
        self.quant = quant or {"enabled": False}
        self.chat_template = chat_template
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code

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

        model_kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }

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
                )
                model_kwargs["quantization_config"] = bnb_cfg
                logger.info("已启用 4bit 量化：%s", self.quant)
            except Exception as exc:
                logger.warning("量化配置失败，回退到非量化加载：%s", exc)
        else:
            dtype = self.torch_dtype
            model_kwargs["torch_dtype"] = dtype if dtype != "auto" else (torch.float16 if self.device == "cuda" else torch.float32)

        with Timer(f"加载 LLM {self.model_name}"):
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
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
    ) -> str | Generator[str, None, None]:
        """对话式生成。

        Args:
            messages: OpenAI 风格消息列表 [{"role":..., "content":...}]。
            generation: 覆盖默认生成参数。
            stream: 是否流式返回（str 生成器）。

        Returns:
            非流式：完整回答；流式：逐片段生成器。
        """
        prompt = self._build_prompt(messages)
        gen_cfg = generation or self.gen_cfg
        if stream:
            return self._stream_generate(prompt, gen_cfg)
        return self._generate(prompt, gen_cfg)

    def _generate(self, prompt: str, gen_cfg: GenerationConfig) -> str:
        import torch  # type: ignore

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with Timer(f"llm generate"):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_cfg.max_new_tokens,
                    temperature=gen_cfg.temperature,
                    top_p=gen_cfg.top_p,
                    repetition_penalty=gen_cfg.repetition_penalty,
                    do_sample=gen_cfg.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
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
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            repetition_penalty=gen_cfg.repetition_penalty,
            do_sample=gen_cfg.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )

        def _worker():
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        th = Thread(target=_worker, daemon=True)
        th.start()
        for piece in streamer:
            yield piece
        th.join()

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