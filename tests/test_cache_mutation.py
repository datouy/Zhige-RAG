"""答案缓存键纳入内容变更版本号 + 检索缓存 LRU 回归测试。"""
from __future__ import annotations

import threading

from src.rag_pipeline import RAGPipeline


class _Store:
    def __init__(self, n: int = 10, mutation: int = 0):
        self._n = n
        self._mutation_version = mutation
        self.collection_name = "test_kb"

    def count(self) -> int:
        return self._n

    def query(self, *a, **k):
        return []


def _pipeline(store: _Store) -> RAGPipeline:
    p = object.__new__(RAGPipeline)
    p.config = {"retrieval": {"answer_cache_enabled": True, "answer_cache_ttl_seconds": 86400}}
    p.vector_store = store
    p._answer_cache = {}
    p._answer_cache_lock = threading.Lock()
    p._retrieval_cache = None
    p._retrieval_cache_lock = None
    return p


def test_version_includes_mutation():
    s = _Store(n=10, mutation=0)
    p = _pipeline(s)
    k1 = p._answer_cache_key("问题", 8, None, None)
    s._mutation_version = 1  # 替换等量文档，count 不变
    k2 = p._answer_cache_key("问题", 8, None, None)
    assert k1 != k2, "mutation version 变化后缓存键必须变化，否则替换文档旧答案会残留"


def test_version_tuple():
    s = _Store(n=10, mutation=0)
    p = _pipeline(s)
    assert p._collection_version() == (10, 0)
    s._mutation_version = 5
    assert p._collection_version() == (10, 5)


def test_retrieval_cache_lru_evicts():
    import src.rag_pipeline as rp

    class FakeVS:
        collection_name = "kb"

        def query(self, query_text, top_k, where=None, score_threshold=0.0):
            return []

    p = object.__new__(RAGPipeline)
    p.config = {"retrieval": {}}
    p.vector_store = FakeVS()
    p._retrieval_cache = None
    p._retrieval_cache_lock = None

    orig = rp.RETRIEVAL_CACHE_MAX_ENTRIES
    rp.RETRIEVAL_CACHE_MAX_ENTRIES = 2
    try:
        for i in range(3):
            p._cached_vector_query("q%d" % i, 5, None)
        assert len(p._retrieval_cache) <= 2, "超过上限应逐出最久未使用条目"
    finally:
        rp.RETRIEVAL_CACHE_MAX_ENTRIES = orig

