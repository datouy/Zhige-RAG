"""重排序（Reranker）模型封装。

提供两种可替换的重排后端（由 ``reranker.backend`` 决定）：

- ``bge``：基于 ``FlagEmbedding`` / ``sentence-transformers`` 的 CrossEncoder，
  默认模型 ``BAAI/bge-reranker-base``（轻量、中文表现好）。
- ``d1`` ：基于 d1 **决策模型**的 ``score`` 原语（见 :mod:`src.decision_provider`），
  对每个候选做一次"单次前向"的相关度分级打分。优势是毫秒级、0 输出 token，
  且在无 GPU 的 CPU 机器上通常比交叉编码器更快；代价是需要 d1 权重（或退化为
  用现有 LLM 模拟）。两种后端输出接口完全一致，可无缝互换。

两者都返回"按相关度降序、分数归一到 0~1"的 Hit 列表，因此上层
``RAGPipeline._rerank`` 的 ``min_relevance`` 二次过滤对两者同样生效。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

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


# ======================================================================
#  d1 决策重排（score 原语）
# ======================================================================
# 相关度阶梯：7 级（落在 d1 的 2~10 级约束内）。粒度够细，且天然给出
# "完全不相关 / 完全相关" 两个端点，便于上层用 min_relevance 做硬护栏。
DEFAULT_RELEVANCE_LADDER = [
    "完全不相关",
    "基本无关",
    "弱相关",
    "部分相关",
    "较为相关",
    "高度相关",
    "完全相关",
]

DEFAULT_RERANK_INSTRUCTION = (
    "判断「片段」对回答「问题」的相关程度。只依据片段内容是否真的能支撑回答该问题，"
    "不要因为出现相同的关键词就判为高相关。"
)


class D1Reranker:
    """基于 d1 ``score`` 原语的决策式重排器。

    对每个候选片段单独做一次分级打分（期望分值归一化到 0~1），再按分数降序。
    与 :class:`BgeReranker` 保持相同的 ``rerank()`` 签名，可直接互换。

    Args:
        provider: :class:`src.decision_provider.BaseDecisionProvider` 实例。
        criteria: 相关度阶梯（由低到高）；默认 7 级。
        max_candidates: 最多重排多少个候选（成本护栏；超出部分按原序保留在尾部）。
        instruction: 判定指令，可覆盖默认文案。
    """

    def __init__(
        self,
        provider: Any,
        criteria: Optional[Sequence[str]] = None,
        max_candidates: int = 24,
        instruction: Optional[str] = None,
    ) -> None:
        if provider is None:
            raise ValueError("D1Reranker 需要非空 decision provider")
        self.provider = provider
        self.criteria: List[str] = list(criteria or DEFAULT_RELEVANCE_LADDER)
        self.max_candidates = max(1, int(max_candidates))
        self.instruction = instruction or DEFAULT_RERANK_INSTRUCTION

    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        hits: Sequence[Hit],
        top_n: int = 3,
    ) -> List[Hit]:
        """对初检结果用决策打分重排。

        决策层失败时**按初检顺序降级返回**，绝不拖垮问答链路。
        """
        if not hits:
            return []
        top_n = max(1, min(top_n, len(hits)))

        head = list(hits)[: self.max_candidates]
        tail = list(hits)[self.max_candidates :]

        scored: List[tuple] = []
        with Timer(f"d1-rerank x{len(head)}"):
            for h in head:
                text = f"问题：{query}\n片段：{h.text}"
                res = self.provider.score(text, self.instruction, self.criteria)
                if res is None:
                    # 单条失败：沿用初检分，保持可排序
                    scored.append((h, float(h.score)))
                else:
                    scored.append((h, float(res.normalized)))

        scored.sort(key=lambda t: -t[1])
        reranked: List[Hit] = []
        for i, (h, s) in enumerate(scored):
            reranked.append(
                Hit(
                    id=h.id,
                    text=h.text,
                    score=float(s),
                    metadata={
                        **h.metadata,
                        "_rerank_score": float(s),
                        "_rerank_rank": i,
                        "_rerank_backend": "d1",
                    },
                )
            )
        # 超出重排预算的候选按原序追加在尾部（分数沿用初检分）
        for h in tail:
            reranked.append(h)
        logger.info("d1 重排完成：%d → %d", len(hits), len(reranked))
        return reranked[:top_n] if top_n < len(reranked) else reranked


# ======================================================================
#  工厂
# ======================================================================
def create_reranker(
    cfg: Optional[Dict[str, Any]] = None,
    decision_provider: Optional[Any] = None,
    llm: Optional[Any] = None,
    llm_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """按配置创建重排器（``bge`` 或 ``d1``），未启用时返回 ``None``。

    配置示例::

        reranker:
          enabled: true
          backend: "d1"            # bge | d1 | auto
          model_name: "./models/bge-reranker-base"   # backend=bge 时使用
          top_n: 3
          min_relevance: 0.3

    Args:
        cfg: 完整配置 dict（读取其 ``reranker`` / ``decision`` 子节）。
        decision_provider: 复用已有决策 provider（避免重复加载）。
        llm: 构造 ``llm`` 决策后端时可选注入。
        llm_cfg: ``config['llm']`` 子 dict，供决策后端现建 LLM 时使用。

    Returns:
        :class:`BgeReranker` / :class:`D1Reranker` / ``None``。
    """
    cfg = cfg or {}
    rcfg = cfg.get("reranker", {}) or {}
    if not rcfg.get("enabled"):
        return None

    backend = str(rcfg.get("backend") or "bge").strip().lower()

    # ---- d1 决策重排 ----
    if backend in ("d1", "auto"):
        provider = decision_provider
        if provider is None:
            try:
                from .decision_provider import create_decision_provider

                provider = create_decision_provider(
                    cfg.get("decision"), llm=llm, llm_cfg=llm_cfg
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("d1 重排 provider 构造失败：%s", exc)
                provider = None
        if provider is not None:
            return D1Reranker(
                provider,
                criteria=rcfg.get("relevance_ladder"),
                max_candidates=int(rcfg.get("max_candidates", 24)),
                instruction=rcfg.get("instruction"),
            )
        if backend == "d1":
            logger.warning(
                "reranker.backend=d1 但决策 provider 不可用，已回落 bge 重排。"
                "（请检查 decision 配置或安装 requirements-d1.txt）"
            )
        # auto 或 d1 回落：继续走 bge

    # ---- bge 交叉编码重排 ----
    if not rcfg.get("model_name"):
        logger.warning("reranker.enabled=true 但未配置 model_name，重排停用。")
        return None
    try:
        return BgeReranker(
            model_name=rcfg["model_name"],
            device=rcfg.get("device", "auto"),
            cache_dir=rcfg.get("cache_dir"),
            max_length=int(rcfg.get("max_length", 512)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("bge 重排器加载失败，重排停用：%s", exc)
        return None


__all__ = [
    "BgeReranker",
    "D1Reranker",
    "DEFAULT_RELEVANCE_LADDER",
    "create_reranker",
]