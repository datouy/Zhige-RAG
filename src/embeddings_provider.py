"""Embedding 后端抽象：``local``（sentence-transformers）/ ``ollama`` / ``openai``（OpenAI 兼容）。

为什么必须和 LLM 一起做
-----------------------
只让 LLM 支持多后端是不够的：用户把 ``llm.backend`` 改成 ``ollama`` 之后，
Embedding 仍然会去加载本地 ``sentence-transformers`` 模型 —— 而它依赖 torch
（2~3 GB）。结果就是**用户依然卡在"下载模型 + 装 torch"这一步**，
"零门槛"根本没落地。

本模块让 Embedding 也能走 HTTP：

- ``ollama``：本地环境下，``ollama pull nomic-embed-text`` 即可，
  **完全不需要 torch、不需要 GPU**；
- ``openai``：任何提供 ``/v1/embeddings`` 的服务（OpenAI / 通义 DashScope /
  硅基流动 / 智谱 / vLLM …）。

设计要点
--------
1. **接口与 :class:`src.embeddings.EmbeddingModel` 严格对齐**：
   ``encode(texts, batch_size=...)`` 返回带 ``.tolist()`` 的 numpy 数组，
   另有 ``encode_one`` / ``dim`` / ``dimension``。
   调用方（``ChromaStore``）自己算完向量再交给 Chroma，因此只要能满足这个契约
   就能无缝替换。
2. **自动归一化**：不同服务返回的向量是否已归一化并不一致，这里统一按
   ``normalize`` 处理，保证余弦相似度量纲正确。
3. **维度必须与已有 collection 一致**：换 Embedding 模型会改变向量维度，
   直接复用旧 collection 会在写入时报错。见
   :func:`embedding_collection_suffix` —— 非 local 后端会自动给 collection
   名加后缀，避免新旧数据混在一个库里。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.http_client import RemoteServiceError, build_headers, is_local_endpoint, post_json
from src.utils import Timer, get_logger

logger = get_logger("embeddings_provider")

BACKEND_LOCAL = "local"
BACKEND_OLLAMA = "ollama"
BACKEND_OPENAI = "openai"
SUPPORTED_BACKENDS = (BACKEND_LOCAL, BACKEND_OLLAMA, BACKEND_OPENAI)

_DEFAULT_OLLAMA_BASE = "http://localhost:11434/v1"
_DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
_DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
_DEFAULT_OPENAI_MODEL = "text-embedding-3-small"

# 单次请求最多提交多少条文本（OpenAI 协议上限 2048，这里留足余量）
_MAX_INPUTS_PER_REQUEST = 512

EmbeddingProviderError = RemoteServiceError


class OpenAICompatEmbedding:
    """OpenAI 兼容的 Embedding 后端（零第三方依赖，仅需 numpy）。

    适用于一切暴露 ``POST {base_url}/embeddings`` 的服务：OpenAI、阿里云
    DashScope（兼容模式）、硅基流动、智谱、vLLM、LM Studio、Ollama 等。

    Args:
        model_name: 模型标识（如 ``nomic-embed-text``、``text-embedding-3-small``）。
        base_url: 服务根地址，需含 ``/v1`` 这类前缀。
        api_key: 鉴权密钥；本地服务可为空。
        batch_size: 默认每批提交的文本条数。
        normalize: 是否对返回向量做 L2 归一化。
        timeout: 单次请求超时（秒）。
        dim: 向量维度。**建议显式配置**：它决定 collection 的向量维度，
            留空时会在首次 ``encode`` 后自动探测。
        provider_label: 日志与错误提示中显示的后端名。
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: Optional[str] = None,
        batch_size: int = 16,
        normalize: bool = True,
        timeout: float = 60.0,
        dim: Optional[int] = None,
        provider_label: str = BACKEND_OPENAI,
    ) -> None:
        if not model_name:
            raise EmbeddingProviderError(f"{provider_label} Embedding 后端缺少 model 配置")
        if not base_url:
            raise EmbeddingProviderError(f"{provider_label} Embedding 后端缺少 base_url 配置")

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.batch_size = max(1, int(batch_size))
        self.normalize = bool(normalize)
        self.timeout = float(timeout)
        self.provider_label = provider_label
        # 0 表示"尚未探测到"。与 EmbeddingModel 一样是普通属性，便于外部读取。
        self.dim: int = int(dim) if dim else 0
        self._expected_dim = int(dim) if dim else 0

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (
            f"OpenAICompatEmbedding(provider={self.provider_label}, "
            f"model={self.model_name}, base_url={self.base_url}, dim={self.dim or '未知'})"
        )

    # ------------------------------------------------------------------
    #  与 EmbeddingModel.encode 同签名
    # ------------------------------------------------------------------
    def encode(
        self,
        texts: Sequence[str] | str,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        convert_to_numpy: bool = True,
    ):
        """将文本列表编码为向量，返回 ``numpy.ndarray``（shape=[N, dim]）。

        参数与 :meth:`src.embeddings.EmbeddingModel.encode` 完全一致，
        以便两者可互换使用。
        """
        if isinstance(texts, str):
            texts = [texts]
        # 空串会被部分服务判为非法输入，统一替换成空格（与本地实现一致）
        texts = [t if t else " " for t in list(texts)]
        if not texts:
            return np.zeros((0, self.dim or 0), dtype="float32")

        bs = max(1, int(batch_size or self.batch_size))
        vectors: List[List[float]] = []
        with Timer(f"embedding(api) encode x{len(texts)}"):
            for start in range(0, len(texts), bs):
                chunk = texts[start:start + bs]
                vectors.extend(self._embed_batch(chunk))

        array = np.asarray(vectors, dtype="float32")
        if self.normalize and array.size:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            array = array / norms

        if array.size:
            self._remember_dim(int(array.shape[1]))
        return array if convert_to_numpy else array.tolist()

    def encode_one(self, text: str):
        """编码单条文本，返回一维向量。"""
        return self.encode([text], batch_size=1)[0]

    @property
    def dimension(self) -> int:
        return self.dim

    # ------------------------------------------------------------------
    def _remember_dim(self, dim: int) -> None:
        if self._expected_dim and dim != self._expected_dim:
            raise EmbeddingProviderError(
                f"{self.provider_label} Embedding 返回维度 {dim} 与配置的 "
                f"{self._expected_dim} 不一致。请修正配置中的 dim，"
                f"或更换已用该模型建库的 collection。"
            )
        if not self.dim:
            self.dim = dim
            logger.info("Embedding 维度探测结果：%d（model=%s）", dim, self.model_name)

    def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:
        payload = {
            "model": self.model_name,
            "input": list(batch),
            "encoding_format": "float",
        }
        data = post_json(
            f"{self.base_url}/embeddings",
            payload,
            headers=build_headers(self.api_key),
            timeout=self.timeout,
            label=f"{self.provider_label}(embeddings)",
            model=self.model_name,
        )
        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise EmbeddingProviderError(
                f"{self.provider_label} Embedding 返回体缺少 data 字段：{str(data)[:200]}"
            )
        if len(items) != len(batch):
            raise EmbeddingProviderError(
                f"{self.provider_label} Embedding 返回条数不符："
                f"请求 {len(batch)} 条，返回 {len(items)} 条"
            )
        # 协议要求按 index 对应，但个别服务不保证顺序，这里显式排序
        try:
            items = sorted(items, key=lambda d: int(d.get("index", 0)))
        except (TypeError, ValueError):
            pass

        result: List[List[float]] = []
        for item in items:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list):
                raise EmbeddingProviderError(
                    f"{self.provider_label} Embedding 返回了非 float 数组的 embedding"
                    f"（可能启用了 base64 编码）：{str(item)[:120]}"
                )
            result.append(vector)
        return result


# ======================================================================
#  维度 / collection 隔离
# ======================================================================
def embedding_fingerprint(emb_cfg: Dict[str, Any]) -> str:
    """返回当前 Embedding 配置的短指纹（backend + model）。

    用于避免"换了 Embedding 模型却复用旧 collection"——两者维度不同，
    写入时会直接报错，且混库后检索结果不可信。
    """
    backend = str(emb_cfg.get("backend") or BACKEND_LOCAL).strip().lower()
    if backend == BACKEND_LOCAL:
        model = emb_cfg.get("model_name", "")
    else:
        model = (emb_cfg.get(backend) or {}).get("model") or emb_cfg.get("model_name", "")
    raw = f"{backend}:{model}".strip().lower()
    # 简单可读的指纹：只保留字母数字，取前 8 位；同名模型得到同一指纹
    safe = "".join(ch if ch.isalnum() else "_" for ch in raw)[:40]
    return safe or "default"


def embedding_collection_name(base_name: str, emb_cfg: Dict[str, Any]) -> str:
    """按 Embedding 后端为 collection 起名。

    - ``local`` 后端：**沿用原名**，保证升级不破坏已有知识库；
    - 其它后端：追加后端与模型指纹，避免与本地向量（维度不同）混在一个库里。
    """
    backend = str(emb_cfg.get("backend") or BACKEND_LOCAL).strip().lower()
    if backend == BACKEND_LOCAL:
        return base_name
    return f"{base_name}__{embedding_fingerprint(emb_cfg)}"


# ======================================================================
#  工厂
# ======================================================================
def _env_api_key(backend: str) -> str:
    if backend == BACKEND_OLLAMA:
        return os.getenv("OLLAMA_API_KEY", "")
    return os.getenv("OPENAI_API_KEY", "")


def _build_remote_embedding(emb_cfg: Dict[str, Any], backend: str):
    section = emb_cfg.get(backend) or {}
    if backend == BACKEND_OLLAMA:
        default_base, default_model = _DEFAULT_OLLAMA_BASE, _DEFAULT_OLLAMA_MODEL
    else:
        default_base, default_model = _DEFAULT_OPENAI_BASE, _DEFAULT_OPENAI_MODEL

    base_url = section.get("base_url") or emb_cfg.get("base_url") or default_base
    model_name = section.get("model") or emb_cfg.get("model_name") or default_model
    api_key = section.get("api_key") or emb_cfg.get("api_key") or _env_api_key(backend)

    return OpenAICompatEmbedding(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        batch_size=int(section.get("batch_size") or emb_cfg.get("batch_size") or 16),
        normalize=bool(emb_cfg.get("normalize_embeddings", True)),
        timeout=float(section.get("timeout") or emb_cfg.get("timeout") or 60.0),
        dim=section.get("dim") or emb_cfg.get("dim"),
        provider_label=backend,
    )


def _build_local_embedding(emb_cfg: Dict[str, Any]):
    from src.embeddings import EmbeddingModel

    return EmbeddingModel(
        model_name=emb_cfg.get("model_name", "BAAI/bge-small-zh-v1.5"),
        device=emb_cfg.get("device", "auto"),
        batch_size=int(emb_cfg.get("batch_size", 32)),
        max_seq_length=int(emb_cfg.get("max_seq_length", 512)),
        normalize=bool(emb_cfg.get("normalize_embeddings", True)),
        cache_dir=emb_cfg.get("cache_dir"),
        local_files_only=bool(emb_cfg.get("local_files_only", False)),
    )


def create_embedding(emb_cfg: Optional[Dict[str, Any]] = None):
    """按 ``embedding.backend`` 创建 Embedding 实例。

    ``backend`` 取值：

    - ``local``（默认）：sentence-transformers，需要 torch，行为与改造前一致；
    - ``ollama``：本机 Ollama 的 embedding 模型，**不需要 torch**；
    - ``openai``：任意 OpenAI 兼容的 ``/v1/embeddings`` 服务。

    Raises:
        ValueError: ``backend`` 取值非法。
        EmbeddingProviderError: 远程后端缺少必要配置。
    """
    cfg = dict(emb_cfg or {})
    backend = str(cfg.get("backend") or BACKEND_LOCAL).strip().lower()

    if backend in (BACKEND_OLLAMA, BACKEND_OPENAI):
        embedding = _build_remote_embedding(cfg, backend)
        logger.info(
            "Embedding 后端：%s（model=%s, base_url=%s）",
            backend,
            embedding.model_name,
            embedding.base_url,
        )
        return embedding

    if backend != BACKEND_LOCAL:
        raise ValueError(
            f"未知的 embedding.backend={backend!r}，可选值：{' / '.join(SUPPORTED_BACKENDS)}"
        )

    return _build_local_embedding(cfg)


__all__ = [
    "BACKEND_LOCAL",
    "BACKEND_OLLAMA",
    "BACKEND_OPENAI",
    "SUPPORTED_BACKENDS",
    "EmbeddingProviderError",
    "OpenAICompatEmbedding",
    "create_embedding",
    "embedding_collection_name",
    "embedding_fingerprint",
    "is_local_endpoint",
]
