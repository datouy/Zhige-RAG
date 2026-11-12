"""RRF 融合序被原始 score 重排覆盖 —— 回归测试。

真实故障（混合检索）：
- 向量路命中的是**语义相关**片段，余弦相似度约 0.55；
- 关键词路命中的是**无关但字面命中**的长文档，BM25 原始分经
  ``s / (1 + s)`` 压缩后约 0.90 —— 比向量路的 0.55 还高。

``bm25_weight = 0.5`` 本来就是为了防这件事：让关键词只做召回补充、
不压过语义相似度。RRF 融合也确实是按**名次**而不是分数做的。

但 ``RAGPipeline._retrieve`` 在融合之后又来了一句::

    hits = sorted(hits, key=lambda h: h.score, reverse=True)[:requested_k]

这一句把 RRF 算好的名次**整个推翻**，按混合量纲的原始分重排 ——
``bm25_weight`` 的保护作用在最后一步被自己废掉了，无关文档重新顶进引用列表。

修复：融合层把名次写进 ``metadata["_fusion_rank"]``，截断时按融合序走，
不再按 ``score`` 重排。
"""

from __future__ import annotations

import pytest

from src.rag_pipeline import RAGPipeline
from src.vector_store import BM25Index, ChromaStore, Hit


# ============================ 融合层 ============================
class TestFusionRankIsRecorded:
    """融合层必须把 RRF 名次写进 metadata，供下游按名次排序。"""

    def _store(self) -> ChromaStore:
        store = ChromaStore.__new__(ChromaStore)
        store.hybrid_fetch_k = 20
        store.hybrid_bm25_weight = 0.5
        store._bm25 = BM25Index(
            ids=["kw-1"],
            texts=["关键词命中的无关长文档" * 20],
            metadatas=[{"source": "无关.docx", "doc_status": "active"}],
        )
        return store

    def test_fusion_rank_is_monotonic_and_dense(self):
        vec = [
            Hit(id="vec-1", text="语义相关", score=0.55, metadata={"source": "论文.docx"}),
            Hit(id="vec-2", text="语义相关2", score=0.40, metadata={"source": "论文.docx"}),
        ]
        hits = self._store()._fuse_hybrid(
            "无关关键词", vec, 5, where={"doc_status": "active"}
        )
        ranks = [h.metadata.get("_fusion_rank") for h in hits]
        assert ranks == sorted(ranks), "融合名次必须单调递增"
        assert ranks == list(range(len(hits))), "融合名次必须是 0..n-1 的稠密序列"

    def test_fusion_score_is_recorded(self):
        vec = [Hit(id="vec-1", text="x", score=0.5, metadata={"source": "a.md"})]
        hits = self._store()._fuse_hybrid(
            "无关关键词", vec, 5, where={"doc_status": "active"}
        )
        assert all("_fusion_score" in h.metadata for h in hits)

    def test_vector_hit_metadata_is_copied_not_mutated(self):
        """融合层不应就地改写调用方传入的 Hit 的 metadata（否则会污染缓存）。"""
        vec = [
            Hit(id="vec-1", text="x", score=0.5, metadata={"source": "a.md"}),
        ]
        original_md = dict(vec[0].metadata)
        self._store()._fuse_hybrid(
            "无关关键词", vec, 5, where={"doc_status": "active"}
        )
        assert vec[0].metadata == original_md, "原始 Hit 的 metadata 不应被就地修改"

    def test_reranker_preserves_fusion_metadata(self):
        """多路融合（_rrf_fuse_variants）同样要写名次。"""
        pipe = object.__new__(RAGPipeline)
        lists = [
            [Hit(id="a", text="A", score=0.9, metadata={})],
            [Hit(id="b", text="B", score=0.1, metadata={})],
        ]
        fused = pipe._rrf_fuse_variants(lists, 5)
        ranks = [h.metadata.get("_fusion_rank") for h in fused]
        assert ranks == list(range(len(fused)))


# ============================ 截断层 ============================
class TestRetrieveDoesNotResortByRawScore:
    """``_retrieve`` 截断时必须保留融合序，不能按原始 score 重排。"""

    def _pipeline(self, hits):
        class _Store:
            def query(self, query_text, top_k, where=None, score_threshold=0.0):
                return list(hits)

        p = object.__new__(RAGPipeline)
        p.config = {"retrieval": {"min_score": 0.0, "fetch_k": 24}}
        p.vector_store = _Store()
        return p

    def test_keyword_hit_with_higher_raw_score_does_not_jump_ahead(self):
        """关键词路原始分更高，也不得把语义相关的片段挤出去。

        这是本 bug 的核心断言：融合序里 ``vec-1`` 在前，``kw-1`` 在后；
        但 ``kw-1`` 的 score(0.90) > ``vec-1`` 的 score(0.55)。
        按 score 重排会把顺序颠倒。

        注意必须让命中数**超过** top_k —— 截断排序只在这条路径上发生。
        """
        hits = [
            Hit(
                id="vec-1",
                text="语义真正相关的片段",
                score=0.55,
                metadata={"source": "论文.docx", "_fusion_rank": 0, "_score_kind": "vector"},
            ),
            Hit(
                id="kw-1",
                text="字面命中但无关的片段",
                score=0.90,
                metadata={
                    "source": "无关.docx",
                    "_fusion_rank": 1,
                    "_score_kind": "keyword",
                },
            ),
            Hit(
                id="vec-2",
                text="语义相关的第三块",
                score=0.30,
                metadata={"source": "论文.docx", "_fusion_rank": 2, "_score_kind": "vector"},
            ),
        ]
        out = self._pipeline(hits)._retrieve("问题", top_k=2, where=None)
        assert [h.id for h in out] == ["vec-1", "kw-1"], (
            "融合序被按 score 重排覆盖了：关键词命中的无关片段被顶到了前面"
        )

    def test_truncation_keeps_the_first_k_of_fused_order(self):
        hits = [
            Hit(id=f"h{i}", text=f"t{i}", score=1.0 - i * 0.1, metadata={"_fusion_rank": i})
            for i in range(5)
        ]
        out = self._pipeline(hits)._retrieve("问题", top_k=3, where=None)
        assert [h.id for h in out] == ["h0", "h1", "h2"]

    def test_high_score_keyword_hit_cannot_evict_low_score_vector_hit(self):
        """即使关键词命中的原始分远高于向量命中，也不能挤掉它。

        截断时按 score 重排会让 kw 的 0.90 把 vec 的 0.55 顶到 top_k 之外，
        真正相关的内容就此丢失。
        """
        hits = [
            Hit(
                id="vec-1",
                text="语义相关",
                score=0.20,
                metadata={"source": "论文.docx", "_fusion_rank": 0, "_score_kind": "vector"},
            ),
            Hit(
                id="kw-1",
                text="无关但字面命中",
                score=0.95,
                metadata={"source": "无关.docx", "_fusion_rank": 1, "_score_kind": "keyword"},
            ),
        ]
        out = self._pipeline(hits)._retrieve("问题", top_k=1, where=None)
        assert [h.id for h in out] == ["vec-1"], (
            "按 score 重排把语义相关的 vec-1 挤掉了，关键词命中反而留下了"
        )


# ============================ 语义回归 ============================
class TestBm25WeightProtectionHoldsEndToEnd:
    """端到端：降低 bm25_weight 不应让无关关键词文档压过语义相关文档。"""

    def _fused(self, bm25_weight: float):
        store = ChromaStore.__new__(ChromaStore)
        store.hybrid_fetch_k = 20
        store.hybrid_bm25_weight = bm25_weight
        store._bm25 = BM25Index(
            ids=["kw-1"],
            texts=["关键词命中的无关长文档" * 20],
            metadatas=[{"source": "无关.docx", "doc_status": "active"}],
        )
        vec = [
            Hit(id="vec-1", text="语义真正相关", score=0.55, metadata={"source": "论文.docx"}),
        ]
        hits = store._fuse_hybrid(
            "无关关键词", vec, 5, where={"doc_status": "active"}
        )
        return [h.id for h in hits]

    def test_keyword_hit_is_never_first_when_weight_is_low(self):
        """bm25_weight=0.5（默认）时，关键词命中不该排在语义命中之前。"""
        ids = self._fused(0.5)
        assert ids.index("vec-1") < ids.index("kw-1"), (
            f"默认权重下关键词命中仍排在了前面：{ids}"
        )

    def test_full_weight_can_let_keyword_win(self):
        """对照组：权重拉满时关键词路可以领先 —— 说明权重确实生效。"""
        ids = self._fused(1.0)
        assert "kw-1" in ids


# ============================ Hit 辅助属性 ============================
class TestHitHelpers:
    def test_fusion_rank_property(self):
        h = Hit(id="x", text="t", score=0.5, metadata={"_fusion_rank": 3})
        assert h.fusion_rank == 3

    def test_fusion_rank_is_none_when_absent(self):
        h = Hit(id="x", text="t", score=0.5, metadata={})
        assert h.fusion_rank is None

    def test_fusion_rank_handles_garbage(self):
        h = Hit(id="x", text="t", score=0.5, metadata={"_fusion_rank": "abc"})
        assert h.fusion_rank is None

    def test_score_kind_defaults_to_vector(self):
        h = Hit(id="x", text="t", score=0.5, metadata={})
        assert h.score_kind == "vector"

    @pytest.mark.parametrize("kind", ["vector", "keyword", "fusion"])
    def test_score_kind_reads_metadata(self, kind):
        h = Hit(id="x", text="t", score=0.5, metadata={"_score_kind": kind})
        assert h.score_kind == kind
