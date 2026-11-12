"""min_score 与关键词分数量纲冲突 —— 回归测试。

问题：``min_score`` 是"语义相关性"的门槛（默认 0.15），但它此前直接作用在
**所有命中的最大分**上。而混合检索里两类命中的分数量纲根本不同：

- 向量命中：余弦相似度，0~1，确实代表语义相关性；
- 关键词命中：BM25 原始分压缩成 ``s / (1 + s)``，长文档命中词多就能上 0.90，
  但它只说明"字面命中了查询词"。

于是"字面沾边但毫不相关"的长文档（0.90）会顶开一条弱语义命中（0.10），
让 min_score 形同虚设 —— 系统在根本没匹配上的问题上硬答。

修复：阈值按分数**类型**分口径判定。存在向量命中时用向量命中里的最高语义分
比 ``min_score``；一个向量命中都没有（纯关键词召回）才退到
``keyword_min_score`` 这个更高的独立门槛。
"""

from __future__ import annotations

from src.rag_pipeline import RAGPipeline
from src.vector_store import Hit


def _pipeline(hits, retrieval_cfg=None):
    class _Store:
        def query(self, query_text, top_k, where=None, score_threshold=0.0):
            return list(hits)

    p = object.__new__(RAGPipeline)
    p.config = {
        "retrieval": {
            "min_score": 0.15,
            "keyword_min_score": 0.60,
            "fetch_k": 24,
            **(retrieval_cfg or {}),
        }
    }
    p.vector_store = _Store()
    return p


def _kw(hid, score, rank=0):
    return Hit(
        id=hid,
        text="字面命中但语义无关",
        score=score,
        metadata={"source": "无关.docx", "_score_kind": "keyword", "_fusion_rank": rank},
    )


def _vec(hid, score, rank=0):
    return Hit(
        id=hid,
        text="语义相关片段",
        score=score,
        metadata={"source": "论文.docx", "_score_kind": "vector", "_fusion_rank": rank},
    )


class TestMinScoreUsesSemanticScale:
    def test_high_keyword_score_cannot_rescue_weak_semantic_hit(self):
        """关键词分 0.90 也不能让语义分 0.05 的命中越过 min_score。"""
        hits = [_vec("vec-1", 0.05), _kw("kw-1", 0.90, rank=1)]
        out = _pipeline(hits)._retrieve("问题", top_k=5, where=None)
        assert out == [], "弱语义命中 + 强关键词分不应通过 min_score"

    def test_weak_semantic_hit_alone_is_rejected(self):
        hits = [_vec("vec-1", 0.05)]
        assert _pipeline(hits)._retrieve("问题", top_k=5, where=None) == []

    def test_strong_semantic_hit_passes(self):
        hits = [_vec("vec-1", 0.55)]
        out = _pipeline(hits)._retrieve("问题", top_k=5, where=None)
        assert [h.id for h in out] == ["vec-1"]

    def test_strong_semantic_hit_passes_despite_weak_keyword_hit(self):
        hits = [_vec("vec-1", 0.55), _kw("kw-1", 0.05, rank=1)]
        out = _pipeline(hits)._retrieve("问题", top_k=5, where=None)
        assert [h.id for h in out] == ["vec-1", "kw-1"]


class TestKeywordOnlyRecall:
    """纯关键词召回（一个向量命中都没有）走独立的 keyword_min_score 门槛。"""

    def test_low_keyword_score_is_rejected(self):
        hits = [_kw("kw-1", 0.30)]
        assert _pipeline(hits)._retrieve("问题", top_k=5, where=None) == []

    def test_high_keyword_score_passes(self):
        hits = [_kw("kw-1", 0.95)]
        out = _pipeline(hits)._retrieve("问题", top_k=5, where=None)
        assert [h.id for h in out] == ["kw-1"]

    def test_keyword_threshold_can_be_disabled(self):
        """keyword_min_score=0 表示不额外拦截（用户明确要求时）。"""
        hits = [_kw("kw-1", 0.20)]
        out = _pipeline(hits, {"keyword_min_score": 0.0})._retrieve(
            "问题", top_k=5, where=None
        )
        assert [h.id for h in out] == ["kw-1"]

    def test_semantic_min_score_does_not_apply_to_keyword_only(self):
        """min_score=0.15 对关键词路不适用：0.30 的关键词分低于 0.60 门槛被拒，
        但不该因为 0.30 < 0.15 这种巧合逻辑而通过。"""
        hits = [_kw("kw-1", 0.30)]
        assert _pipeline(hits)._retrieve("问题", top_k=5, where=None) == []


class TestMinScoreDisabled:
    def test_zero_min_score_disables_the_guard(self):
        hits = [_vec("vec-1", 0.01)]
        out = _pipeline(hits, {"min_score": 0.0})._retrieve("问题", top_k=5, where=None)
        assert [h.id for h in out] == ["vec-1"]

    def test_no_hits_is_still_empty(self):
        assert _pipeline([])._retrieve("问题", top_k=5, where=None) == []


class TestScoreKindClassification:
    """命中必须能被正确区分成"语义分"和"关键词分"。"""

    def test_missing_score_kind_counts_as_semantic(self):
        """没有 _score_kind 标记的（纯向量路径）按语义分处理。"""
        h = Hit(id="x", text="t", score=0.5, metadata={})
        assert h.score_kind == "vector"

    def test_keyword_tag_is_respected(self):
        h = Hit(id="x", text="t", score=0.5, metadata={"_score_kind": "keyword"})
        assert h.score_kind == "keyword"
