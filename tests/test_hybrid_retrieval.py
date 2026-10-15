"""混合检索的过滤与权重回归测试。

背景（真实故障）：混合检索里 BM25 那一路**没有应用 where 过滤**，于是被
`doc_status` / `acl` 排除的文档会从关键词路溜回结果；而且 BM25 分数被直接
折算成"相似度"，短查询下无关长文档能拿到 0.90，压过真正相关片段的 0.55，
把无关内容顶进引用列表（用户问的是论文，引用里却出现 Spring 面试题）。
"""

from __future__ import annotations

import pytest

from src.vector_store import BM25Index, ChromaStore, Hit, _metadata_matches


# =========================== _metadata_matches ===========================
class TestMetadataMatches:
    """where 求值器：支持项目用到的算子，认不出来就保守排除。"""

    META = {"doc_status": "active", "acl": "*", "department": ""}

    def test_none_where_matches(self):
        assert _metadata_matches(self.META, None) is True
        assert _metadata_matches(self.META, {}) is True

    def test_plain_equality(self):
        assert _metadata_matches(self.META, {"doc_status": "active"}) is True
        assert _metadata_matches(self.META, {"doc_status": "draft"}) is False

    def test_in_and_nin(self):
        assert _metadata_matches(self.META, {"acl": {"$in": ["*", "hr"]}}) is True
        assert _metadata_matches(self.META, {"acl": {"$in": ["hr"]}}) is False
        assert _metadata_matches(self.META, {"acl": {"$nin": ["hr"]}}) is True

    def test_and_or(self):
        where_and = {"$and": [{"doc_status": "active"}, {"acl": {"$in": ["*"]}}]}
        assert _metadata_matches(self.META, where_and) is True
        where_and_bad = {"$and": [{"doc_status": "active"}, {"acl": {"$in": ["hr"]}}]}
        assert _metadata_matches(self.META, where_and_bad) is False
        where_or = {"$or": [{"doc_status": "draft"}, {"acl": "*"}]}
        assert _metadata_matches(self.META, where_or) is True

    def test_unknown_operator_fails_closed(self):
        """不认识的算子必须排除，不能放过（否则过滤会被静默绕过）。"""
        assert _metadata_matches(self.META, {"doc_status": {"$foo": 1}}) is False
        assert _metadata_matches(self.META, {"$where": 1}) is False


# =========================== _fuse_hybrid ===========================
def _store_with_bm25(*, bm25_weight: float = 0.5, fetch_k: int = 20) -> ChromaStore:
    """构造一个只带 BM25 索引的 ChromaStore 壳，避免真的去连 Chroma。"""
    ids = ["in-1", "in-2", "out-1", "out-2"]
    texts = [
        "本系统使用 Qwen2.5-1.5B-Instruct 模型完成本地化部署。",
        "系统实现了知识库管理与引用溯源功能。",
        "Spring 循环依赖靠三级缓存解决，涉及 singletonObjects。",
        "Node.js 的 process.nextTick 与 Promise 执行时机不同。",
    ]
    metas = [
        {"source": "论文.docx", "doc_status": "active", "acl": "*"},
        {"source": "论文.docx", "doc_status": "active", "acl": "*"},
        {"source": "话术.docx", "doc_status": "active", "acl": "hr"},
        {"source": "面试题.docx", "doc_status": "draft", "acl": "*"},
    ]
    store = ChromaStore.__new__(ChromaStore)
    store.hybrid_fetch_k = fetch_k
    store.hybrid_bm25_weight = bm25_weight
    store._bm25 = BM25Index(ids=ids, texts=texts, metadatas=metas)
    return store


class TestFuseHybridRespectsWhere:
    def test_keyword_side_does_not_leak_filtered_docs(self):
        """限定 source=论文.docx 时，关键词路不得返回其它文档。"""
        store = _store_with_bm25()
        where = {"$and": [{"doc_status": "active"}, {"source": "论文.docx"}]}
        hits = store._fuse_hybrid("用的是什么模型 完成的是什么功能", [], 4, where=where)
        assert hits, "过滤后至少应保留本文档的关键词命中"
        assert all((h.metadata or {}).get("source") == "论文.docx" for h in hits)

    def test_acl_filtered_doc_is_excluded(self):
        """acl 不含当前用户的文档不能出现（越权回归）。"""
        store = _store_with_bm25()
        where = {"$and": [{"doc_status": "active"}, {"acl": {"$in": ["*"]}}]}
        hits = store._fuse_hybrid("循环依赖 三级缓存", [], 4, where=where)
        sources = {(h.metadata or {}).get("source") for h in hits}
        assert "话术.docx" not in sources, "acl=hr 的文档不该出现"

    def test_draft_status_doc_is_excluded(self):
        store = _store_with_bm25()
        where = {"doc_status": "active"}
        hits = store._fuse_hybrid("process.nextTick Promise", [], 4, where=where)
        sources = {(h.metadata or {}).get("source") for h in hits}
        assert "面试题.docx" not in sources, "draft 状态的文档不该参与召回"

    def test_keyword_hits_are_tagged(self):
        """关键词命中的分数要标注类型，避免被当成余弦相似度展示。"""
        store = _store_with_bm25()
        hits = store._fuse_hybrid("Qwen2.5 模型", [], 4, where={"doc_status": "active"})
        keyword_hits = [h for h in hits if (h.metadata or {}).get("_score_kind") == "keyword"]
        assert keyword_hits, "纯关键词召回应带上 _score_kind=keyword 标记"


class TestBm25Weight:
    def test_zero_weight_keeps_vector_order(self):
        """权重为 0 时，关键词路不该左右排序（只保留向量命中原序）。"""
        store = _store_with_bm25(bm25_weight=0.0)
        vec = [
            Hit(id="in-1", text="向量命中", score=0.6, metadata={"source": "论文.docx"}),
            Hit(id="in-2", text="向量命中2", score=0.5, metadata={"source": "论文.docx"}),
        ]
        hits = store._fuse_hybrid("Qwen2.5", vec, 2, where={"doc_status": "active"})
        assert [h.id for h in hits] == ["in-1", "in-2"]

    def test_positive_weight_can_add_keyword_only_candidates(self):
        store = _store_with_bm25(bm25_weight=0.5)
        vec = [Hit(id="in-2", text="向量命中", score=0.5, metadata={"source": "论文.docx"})]
        hits = store._fuse_hybrid("Qwen2.5", vec, 3, where={"doc_status": "active"})
        ids = {h.id for h in hits}
        assert "in-1" in ids, "关键词召回应能把向量漏掉的候选补进来"

    @pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
    def test_weight_is_stored(self, weight):
        assert _store_with_bm25(bm25_weight=weight).hybrid_bm25_weight == weight


# =========================== 引用分数 ===========================
class TestCitationScores:
    def test_citations_carry_score_and_kind(self):
        """引用必须带真实分数与类型 —— 之前只取 _rerank_score，未开重排时恒为
        None，界面上四条来源全是「相似度 —」。"""
        from src.prompt_template import PromptTemplate
        from src.rag_pipeline import RAGPipeline

        pipe = RAGPipeline.__new__(RAGPipeline)
        pipe.prompts = PromptTemplate.from_yaml("config/prompts.yaml")

        hits = [
            Hit(id="a", text="论文正文片段" * 5, score=0.546, metadata={"source": "论文.docx", "page": 1}),
            Hit(
                id="b",
                text="关键词召回片段",
                score=0.9,
                metadata={"source": "话术.docx", "page": 3, "_score_kind": "keyword"},
            ),
        ]
        chunks, _ = pipe._build_context(hits, max_chars=4000)
        cites = pipe._format_citations(chunks)

        assert cites[0]["score"] == pytest.approx(0.546)
        assert cites[0]["score_kind"] == "dense"
        assert cites[1]["score_kind"] == "keyword"
        assert cites[1]["score"] == pytest.approx(0.9)
        assert all("cite" in c for c in cites)
