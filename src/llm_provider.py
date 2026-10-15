"""LLM 后端抽象：``local``（HuggingFace transformers）/ ``ollama`` / ``openai``（OpenAI 兼容 HTTP）。

为什么需要这一层
----------------
在引入本模块之前，主链路只能跑本地 transformers 模型（``LocalLLM``），意味着：

- 用户必须先自行下载 GB 级权重，否则**首次启动必然失败**；
- 没有 GPU 的机器只能跑 1.5B 级小模型，回答质量撑不起真实业务；
- 无法复用企业已有的模型服务（DeepSeek / 通义千问 / 智谱 / 硅基流动 / vLLM / LM Studio …）。

而中小企业和个人用户的现实是：**要么机器上已经跑着 Ollama，要么手上有一个 API Key**。
把「模型从哪来」从代码里解耦出来，是"开箱即用"这件事的前提。

设计要点
--------
1. ``ollama`` 与 ``openai`` **共用同一实现**。Ollama 自带
   ``/v1/chat/completions`` 兼容端点，两者只差默认 base_url 与是否需要 api_key。
2. **零新增依赖**：HTTP 与 SSE 解析复用 :mod:`src.http_client`（标准库实现），
   不引入 ``openai`` SDK —— 少一个依赖，就少一个"装不上"的理由。
3. 接口与 :class:`src.llm.LocalLLM` 严格对齐（``chat(messages, generation, stream, timeout)``），
   因此可以在 :meth:`src.rag_pipeline.RAGPipeline._build_llm_from_cfg` 处原地替换，
   上层（RAG 流程 / Agent / Streamlit UI）无需任何改动。

配置示例（``config/config.yaml``）
----------------------------------
::

    llm:
      backend: ollama            # local | ollama | openai
      ollama:
        model: qwen2.5:7b
        base_url: http://localhost:11434/v1
      openai:
        model: deepseek-chat
        base_url: https://api.deepseek.com/v1
        api_key: ${DEEPSEEK_API_KEY:-}
"""
from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

from src.http_client import (
    RemoteServiceError,
    build_headers,
    is_local_endpoint,
    post_json,
    post_sse,
)
from src.llm import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, GenerationConfig, LocalLLM
from src.utils import get_logger

logger = get_logger("llm_provider")


# ======================================================================
#  后端标识
# ======================================================================
BACKEND_LOCAL = "local"
BACKEND_OLLAMA = "ollama"
BACKEND_OPENAI = "openai"
SUPPORTED_BACKENDS = (BACKEND_LOCAL, BACKEND_OLLAMA, BACKEND_OPENAI)

_DEFAULT_OLLAMA_BASE = "http://localhost:11434/v1"
_DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
_DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# 保留历史名字。调用方一直写 `except LLMProviderError`，它必须与
# http_client 实际抛出的 RemoteServiceError 是同一类型，否则异常会被漏捕。
LLMProviderError = RemoteServiceError


# ======================================================================
#  OpenAI 兼容后端
# ======================================================================
class OpenAICompatLLM:
    """OpenAI 兼容的 HTTP 后端（零第三方依赖）。

    适用于一切暴露 ``POST {base_url}/chat/completions`` 的服务：
    OpenAI、DeepSeek、通义千问（DashScope 兼容模式）、智谱、硅基流动、
    vLLM、LM Studio、Ollama 等。

    Args:
        model_name: 模型标识（如 ``qwen2.5:7b``、``deepseek-chat``）。
        base_url: 服务根地址，需含 ``/v1`` 这类前缀（本类会追加 ``/chat/completions``）。
        api_key: 鉴权密钥；本地服务（Ollama / vLLM）可为空。
        generation: 默认生成参数 :class:`~src.llm.GenerationConfig`。
        timeout: 单次请求超时（秒）。
        provider_label: 用于日志与错误提示的后端名（``ollama`` / ``openai``）。
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: Optional[str] = None,
        generation: Optional[GenerationConfig] = None,
        timeout: float = DEFAULT_TIMEOUT,
        provider_label: str = BACKEND_OPENAI,
    ) -> None:
        if not model_name:
            raise LLMProviderError(f"{provider_label} 后端缺少 model 配置")
        if not base_url:
            raise LLMProviderError(f"{provider_label} 后端缺少 base_url 配置")

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.gen_cfg = generation or GenerationConfig()
        self.timeout = float(timeout)
        self.provider_label = provider_label

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (
            f"OpenAICompatLLM(provider={self.provider_label}, "
            f"model={self.model_name}, base_url={self.base_url})"
        )

    def is_loaded(self) -> bool:
        """与 ``LocalLLM`` 语义对齐：HTTP 后端无加载态，恒为已就绪。"""
        return True

    def ensure_loaded(self) -> None:
        """与 ``LocalLLM`` 语义对齐的空操作。"""
        return None

    # ------------------------------------------------------------------
    #  与 LocalLLM.chat 同签名
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        generation: Optional[GenerationConfig] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
    ) -> str | Generator[str, None, None]:
        """对话式生成。

        Args:
            messages: OpenAI 风格消息列表 ``[{"role":..., "content":...}]``。
            generation: 覆盖默认生成参数。
            stream: ``True`` 时返回逐块增量文本的生成器，否则返回完整字符串。
            timeout: 覆盖默认超时（秒）。
        """
        gen = generation or self.gen_cfg
        if stream:
            return self._stream_chat(messages, gen, timeout)
        return self._chat_once(messages, gen, timeout)

    # ------------------------------------------------------------------
    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _chat_once(
        self, messages: List[Dict[str, str]], gen: GenerationConfig, timeout: Optional[float]
    ) -> str:
        data = post_json(
            self._endpoint,
            self._build_payload(messages, gen, stream=False),
            headers=build_headers(self.api_key),
            timeout=timeout or self.timeout,
            label=self.provider_label,
            model=self.model_name,
        )
        choices = data.get("choices") or []
        if not choices:
            raise LLMProviderError(
                f"{self.provider_label} 返回体缺少 choices 字段：{str(data)[:200]}"
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            # 部分推理模型把正文放在 reasoning_content，content 为空
            content = message.get("reasoning_content") or ""
        return str(content).strip()

    def _stream_chat(
        self,
        messages: List[Dict[str, str]],
        gen: GenerationConfig,
        timeout: Optional[float],
    ) -> Generator[str, None, None]:
        for chunk in post_sse(
            self._endpoint,
            self._build_payload(messages, gen, stream=True),
            headers=build_headers(self.api_key),
            timeout=timeout or self.timeout,
            label=self.provider_label,
            model=self.model_name,
        ):
            delta = self._extract_delta(chunk)
            if delta:
                yield delta

    # ------------------------------------------------------------------
    def _build_payload(
        self, messages: List[Dict[str, str]], gen: GenerationConfig, stream: bool
    ) -> Dict[str, Any]:
        """把内部生成参数翻译成 OpenAI 协议字段。

        注意：``repetition_penalty`` 是 HuggingFace 的叫法，OpenAI 协议里没有
        语义等价的字段（``frequency_penalty`` / ``presence_penalty`` 与它并不
        等价），因此**不做强行映射**，避免给出与用户预期不符的行为。
        """
        return {
            "model": self.model_name,
            "messages": [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ],
            "temperature": float(getattr(gen, "temperature", 0.7)),
            "top_p": float(getattr(gen, "top_p", 0.8)),
            "max_tokens": int(getattr(gen, "max_new_tokens", 512)),
            "stream": bool(stream),
        }

    @staticmethod
    def _extract_delta(obj: Any) -> str:
        """从流式分片中取出增量文本。"""
        try:
            choices = obj.get("choices") or []
            if not choices:
                return ""
            choice = choices[0]
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text is None:
                # 兜底：少数服务在流式响应里仍用 message.content
                text = (choice.get("message") or {}).get("content")
            return str(text) if text else ""
        except (AttributeError, TypeError):
            return ""


# ======================================================================
#  工厂
# ======================================================================
def _env_api_key(backend: str) -> str:
    """从环境变量取密钥（配置文件中不落明文密钥）。"""
    import os

    if backend == BACKEND_OLLAMA:
        return os.getenv("OLLAMA_API_KEY", "")
    return os.getenv("OPENAI_API_KEY", "")


def build_generation_config(gen_cfg: Optional[Dict[str, Any]]) -> GenerationConfig:
    """把 ``cfg.llm.generation`` 子 dict 转成 :class:`GenerationConfig`。"""
    gen_cfg = gen_cfg or {}
    return GenerationConfig(
        max_new_tokens=gen_cfg.get("max_new_tokens", 512),
        temperature=gen_cfg.get("temperature", 0.7),
        top_p=gen_cfg.get("top_p", 0.8),
        repetition_penalty=gen_cfg.get("repetition_penalty", 1.05),
        do_sample=gen_cfg.get("do_sample", True),
    )


def _build_local_llm(llm_cfg: Dict[str, Any], gen: GenerationConfig) -> LocalLLM:
    return LocalLLM(
        model_name=llm_cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
        device=llm_cfg.get("device", "auto"),
        device_map=llm_cfg.get("device_map", "auto"),
        torch_dtype=llm_cfg.get("torch_dtype", "auto"),
        quant=llm_cfg.get("quantization"),
        cache_dir=llm_cfg.get("cache_dir"),
        generation=gen,
        chat_template=llm_cfg.get("chat_template", "auto"),
        local_files_only=llm_cfg.get("local_files_only", False),
        trust_remote_code=llm_cfg.get("trust_remote_code", False),
        max_retries=llm_cfg.get("max_retries", DEFAULT_MAX_RETRIES),
    )


def _build_remote_llm(
    llm_cfg: Dict[str, Any], backend: str, gen: GenerationConfig
) -> OpenAICompatLLM:
    section = llm_cfg.get(backend) or {}
    if backend == BACKEND_OLLAMA:
        default_base, default_model = _DEFAULT_OLLAMA_BASE, _DEFAULT_OLLAMA_MODEL
    else:
        default_base, default_model = _DEFAULT_OPENAI_BASE, _DEFAULT_OPENAI_MODEL

    base_url = (
        section.get("base_url")
        or llm_cfg.get("base_url")
        # 历史字段名（scripts/synthesize.py 用过）
        or llm_cfg.get("openai_api_base")
        or default_base
    )
    model_name = section.get("model") or llm_cfg.get("model_name") or default_model
    api_key = section.get("api_key") or llm_cfg.get("api_key") or _env_api_key(backend)
    timeout = float(section.get("timeout") or llm_cfg.get("timeout") or DEFAULT_TIMEOUT)

    return OpenAICompatLLM(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        generation=gen,
        timeout=timeout,
        provider_label=backend,
    )


def create_llm(llm_cfg: Optional[Dict[str, Any]] = None) -> Any:
    """按 ``llm.backend`` 创建后端实例（接口与 ``LocalLLM`` 一致）。

    ``backend`` 取值：

    - ``local``（默认）：本地 HuggingFace 模型，行为与改造前完全一致；
    - ``ollama``：本机/内网 Ollama 服务，**无需 GPU、无需下载权重**；
    - ``openai``：任意 OpenAI 兼容远程服务（DeepSeek / 通义 / 智谱 / vLLM / …）。

    Raises:
        ValueError: ``backend`` 取值非法。
        LLMProviderError: 远程后端缺少必要配置。
    """
    cfg = dict(llm_cfg or {})
    backend = str(cfg.get("backend") or BACKEND_LOCAL).strip().lower()
    gen = build_generation_config(cfg.get("generation"))

    if backend in (BACKEND_OLLAMA, BACKEND_OPENAI):
        llm = _build_remote_llm(cfg, backend, gen)
        logger.info(
            "LLM 后端：%s（model=%s, base_url=%s）",
            backend,
            llm.model_name,
            llm.base_url,
        )
        return llm

    if backend != BACKEND_LOCAL:
        raise ValueError(
            f"未知的 llm.backend={backend!r}，可选值：{' / '.join(SUPPORTED_BACKENDS)}"
        )

    return _build_local_llm(cfg, gen)


__all__ = [
    "BACKEND_LOCAL",
    "BACKEND_OLLAMA",
    "BACKEND_OPENAI",
    "SUPPORTED_BACKENDS",
    "LLMProviderError",
    "OpenAICompatLLM",
    "build_generation_config",
    "create_llm",
    "is_local_endpoint",
]
