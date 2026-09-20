"""Embedding 模型封装。

基于 ``sentence-transformers`` 提供：
- 单文本/批量文本向量化
- 自动设备选择（cuda / cpu）
- 向量归一化（适合点积检索）
- 编码耗时统计

典型用法：

    >>> from src.embeddings import EmbeddingModel
    >>> model = EmbeddingModel("BAAI/bge-small-zh-v1.5")
    >>> vecs = model.encode(["你好世界", "今天天气真好"])
    >>> print(vecs.shape)
    (2, 512)
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

from .utils import Timer, get_logger, resolve_path, select_torch_device

logger = get_logger("embeddings")


def _select_device(device: str) -> str:
    """委托到 :func:`src.utils.select_torch_device`（三处共用，避免漂移）。"""
    return select_torch_device(device, logger=logger)


class EmbeddingModel:
    """Sentence-Transformers Embedding 封装。

    Args:
        model_name: HF 模型名或本地路径。
        device: auto / cpu / cuda。
        batch_size: 批大小。
        max_seq_length: 最大序列长度。
        normalize: 是否归一化向量。
        cache_dir: HF 模型缓存目录。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "auto",
        batch_size: int = 32,
        max_seq_length: int = 512,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.normalize = normalize

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "未安装 sentence-transformers，请运行 `pip install sentence-transformers`"
            ) from exc

        resolved_cache = resolve_path(cache_dir) if cache_dir else None
        if resolved_cache:
            resolved_cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(resolved_cache))

        self.device = _select_device(device)
        logger.info("加载 Embedding 模型: %s (device=%s)", model_name, self.device)

        with Timer(f"加载 Embedding {model_name}"):
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=str(resolved_cache) if resolved_cache else None,
                local_files_only=local_files_only,
            )
        try:
            self.model.max_seq_length = max_seq_length
        except Exception:
            pass
        try:
            self.dim = int(self.model.get_embedding_dimension())
        except AttributeError:
            self.dim = int(self.model.get_sentence_embedding_dimension())

    # ------------------------------------------------------------------
    def encode(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        convert_to_numpy: bool = True,
    ):
        """将文本列表编码为向量。

        Args:
            texts: 输入文本列表。
            batch_size: 覆盖默认批大小。
            show_progress: 是否显示 tqdm 进度条。
            convert_to_numpy: 是否返回 numpy 数组（True 为 numpy）。

        Returns:
            numpy.ndarray 或 list（shape=[N, dim]）。
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if t else " " for t in texts]
        bs = batch_size or self.batch_size
        with Timer(f"embedding encode x{len(texts)}"):
            vecs = self.model.encode(
                list(texts),
                batch_size=bs,
                show_progress_bar=show_progress,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=self.normalize,
            )
        return vecs

    def encode_one(self, text: str):
        """编码单条文本，返回一维向量。"""
        return self.encode([text], batch_size=1)[0]

    # ------------------------------------------------------------------
    @property
    def dimension(self) -> int:
        return self.dim