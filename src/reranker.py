"""重排序（Reranker）模型封装。

基于 ``FlagEmbedding`` 或 ``sentence-transformers`` 的 CrossEncoder，
对初检结果进行精细重排，进一步提升 Top-K 准确率。

默认模型：BAAI/bge-reranker-base（轻量、中文表现好）。
"""

from __future__ import annotations

from typing import List, Sequence

from .utils import Timer, get_logger, resolve_path, select_torch_device
from .vector_store import Hit

logger = get_logger("reranker")


def _select_device(device: str) -> str:
    """委托到 :func:`src.utils.select_torch_device`，与 embeddings 行为一致
    （CPU 版 torch + CUDA 驱动时回落 cpu，避免 FlagReranker/CrossEncoder
    在 ``to('cuda')`` 时崩溃）。"""
    return select_torch_device(device, logger=logger)


class BgeReranker:
    """BGE 风格 Reranker。

    Args:
        model_name: HF 模型名或本地路径。
        device: auto / cpu / cuda。
        cache_dir: HF 缓存目录。
        max_length: 模型最大序列长度。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "auto",
        cache_dir: str = "models/hf_cache",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = _select_device(device)
        self.max_length = max_length

        cache_path = str(resolve_path(cache_dir)) if cache_dir else None
        if cache_path:
            resolve_path(cache_dir).mkdir(parents=True, exist_ok=True)

        # 优先使用 FlagEmbedding
        try:
            from FlagEmbedding import FlagReranker  # type: ignore

            logger.info("使用 FlagEmbedding 加载 Reranker: %s", model_name)
            with Timer(f"加载 Reranker {model_name}"):
                # devices 必须显式收窄为单设备：默认的多设备探测（cuda+cpu）
                # 会走 encode_multi_process，CPU 环境下进程池为空，
                # compute_score 一调用就 ZeroDivisionError（预存在的坑）。
                self.model = FlagReranker(
                    model_name,
                    use_fp16=(self.device == "cuda"),
                    cache_dir=cache_path,
                    devices=[self.device],
                )
            self._backend = "flag"
        except Exception as exc:
            logger.info("FlagEmbedding 不可用（%s），回退到 sentence-transformers CrossEncoder", exc)
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self.model = CrossEncoder(
                    model_name,
                    max_length=max_length,
                    device=self.device,
                )
                self._backend = "cross_encoder"
            except Exception as exc2:
                raise ImportError(
                    "未安装 FlagEmbedding 或 sentence-transformers，无法加载 Reranker"
                ) from exc2

    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        hits: Sequence[Hit],
        top_n: int = 3,
    ) -> List[Hit]:
        """对初检结果重排序。

        Args:
            query: 查询文本。
            hits: 检索命中的 Hit 列表。
            top_n: 返回前 N 条。

        Returns:
            重新排序后的 Hit 列表。
        """
        if not hits:
            return []
        top_n = max(1, min(top_n, len(hits)))

        pairs = [(query, h.text) for h in hits]
        with Timer(f"rerank x{len(pairs)}"):
            if self._backend == "flag":
                try:
                    scores = self.model.compute_score(pairs, normalize=True)
                except Exception as exc:  # noqa: BLE001
                    # 重排是"锦上添花"环节：失败时保持初检顺序降级返回，
                    # 绝不能让重排故障拖垮整条问答链路
                    logger.warning("重排失败，按初检顺序降级返回: %s", exc)
                    return list(hits)[:top_n]
                if isinstance(scores, float):
                    scores = [scores]
            else:
                scores = self.model.predict(pairs, show_progress_bar=False)
                scores = [float(s) for s in scores]

        # 重排
        sorted_pairs = sorted(zip(hits, scores), key=lambda x: -float(x[1]))
        top = sorted_pairs[:top_n]
        reranked: List[Hit] = []
        for i, (h, s) in enumerate(top):
            new_hit = Hit(
                id=h.id,
                text=h.text,
                score=float(s),
                metadata={**h.metadata, "_rerank_score": float(s), "_rerank_rank": i},
            )
            reranked.append(new_hit)
        logger.info("重排完成：%d → %d", len(hits), len(reranked))
        return reranked