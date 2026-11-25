"""Comparison queries must not lose a strong secondary source after rerank."""

from __future__ import annotations

from src.rag_pipeline import RAGPipeline
from src.vector_store import Hit


class _FakeReranker:
    def __init__(self, ranked: list[Hit]) -> None:
        self.ranked = ranked

    def rerank(self, query: str, hits: list[Hit], top_n: int = 3) -> list[Hit]:
        del query, hits
        return list(self.ranked)[:top_n]


def _pipeline(ranked: list[Hit], min_relevance: float = 0.05) -> RAGPipeline:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.config = {"reranker": {"min_relevance": min_relevance}}
    pipeline.reranker = _FakeReranker(ranked)
    return pipeline


def _hit(chunk_id: str, source: str, raw_score: float, rerank_score: float) -> tuple[Hit, Hit]:
    raw = Hit(
        id=chunk_id,
        text=f"{source} 的原文证据",
        score=raw_score,
        metadata={"source": source},
    )
    reranked = Hit(
        id=chunk_id,
        text=raw.text,
        score=rerank_score,
        metadata={"source": source},
    )
    return raw, reranked


def test_comparison_query_keeps_strong_secondary_source_after_score_filter():
    scan_raw, scan_ranked = _hit("scan", "扫描件.pdf", 0.49, 0.09)
    manual_raw, manual_ranked = _hit("manual", "员工手册.md", 0.46, 0.005)
    noise_raw, noise_ranked = _hit("noise", "其他.md", 0.10, 0.001)
    pipeline = _pipeline([scan_ranked, manual_ranked, noise_ranked])

    out = pipeline._rerank(
        "扫描件与员工手册记录的成立年份和总部城市是否一致？",
        [scan_raw, manual_raw, noise_raw],
        top_n=3,
    )

    assert [hit.id for hit in out] == ["scan", "manual"]
    assert out[1].metadata["_source_diversity_guard"] is True


def test_single_source_question_keeps_relevance_filter_strict():
    scan_raw, scan_ranked = _hit("scan", "扫描件.pdf", 0.49, 0.09)
    manual_raw, manual_ranked = _hit("manual", "员工手册.md", 0.46, 0.005)
    pipeline = _pipeline([scan_ranked, manual_ranked])

    out = pipeline._rerank("公司总部在哪里？", [scan_raw, manual_raw], top_n=2)

    assert [hit.id for hit in out] == ["scan"]


def test_diversity_guard_never_resurrects_an_all_filtered_result():
    scan_raw, scan_ranked = _hit("scan", "扫描件.pdf", 0.49, 0.01)
    manual_raw, manual_ranked = _hit("manual", "员工手册.md", 0.46, 0.005)
    pipeline = _pipeline([scan_ranked, manual_ranked], min_relevance=0.05)

    out = pipeline._rerank("扫描件与员工手册是否一致？", [scan_raw, manual_raw], top_n=2)

    assert out == []
