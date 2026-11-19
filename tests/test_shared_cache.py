"""P1-1：可选 Redis 共享缓存 + 缓存值序列化 —— 回归测试。

原始诉求：多 worker 部署下，检索/答案缓存只活在各自进程内存里，无法共享命中。
新增 ``src.cache.SharedCache``（Redis）与 JSON 序列化；未配置 / 未装 redis 时
自动退化为纯进程内缓存，行为不变。
"""
from __future__ import annotations

import threading

from src.cache import (
    SharedCache,
    build_shared_cache,
    deserialize_hits,
    deserialize_result,
    serialize_hits,
    serialize_result,
)
from src.rag_pipeline import RAGPipeline, RAGResult
from src.vector_store import Hit


# ============================ 序列化 ============================
class TestSerialization:
    def test_hits_roundtrip(self):
        hits = [
            Hit(id="a1", text="你好", score=0.8, metadata={"source": "a.md", "chunk_index": 1}),
            Hit(id="b0", text="世界", score=0.6, metadata={"source": "b.md"}),
        ]
        out = deserialize_hits(serialize_hits(hits))
        assert [h.id for h in out] == ["a1", "b0"]
        assert out[0].metadata["chunk_index"] == 1
        assert abs(out[0].score - 0.8) < 1e-9

    def test_result_roundtrip(self):
        res = RAGResult(
            answer="答案",
            sources=[{"index": 1, "title": "a.md", "score": 0.8}],
            raw_hits=[Hit(id="a1", text="正文", score=0.8, metadata={"source": "a.md"})],
            timings={"llm_total_ms": 123.0},
            verification={"grounded": True},
        )
        out = deserialize_result(serialize_result(res))
        assert out.answer == "答案"
        assert out.sources == res.sources
        assert out.verification == {"grounded": True}
        assert out.raw_hits[0].id == "a1"


# ============================ SharedCache（注入假 redis） ============================
class _FakeRedis:
    def __init__(self):
        self.data = {}
        self.deleted = []

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value

    def scan_iter(self, match=None, count=None):
        import fnmatch

        return [k for k in self.data if fnmatch.fnmatch(k, match)]

    def delete(self, *keys):
        for k in keys:
            self.data.pop(k, None)
            self.deleted.append(k)
        return len(keys)


class TestSharedCache:
    def _cache(self):
        fake = _FakeRedis()
        return SharedCache("redis://localhost:6379/0", client=fake), fake

    def test_set_get_uses_namespaced_key(self):
        sc, fake = self._cache()
        sc.set("scope1", "k1", b"v1", 10)
        assert fake.data["zhige:rag:scope1:k1"] == b"v1"
        assert sc.get("scope1", "k1") == b"v1"

    def test_get_missing_returns_none(self):
        sc, _ = self._cache()
        assert sc.get("scope1", "nope") is None

    def test_clear_scope_only_removes_that_scope(self):
        sc, fake = self._cache()
        sc.set("s1", "a", b"1", 10)
        sc.set("s1", "b", b"2", 10)
        sc.set("s2", "c", b"3", 10)
        sc.clear_scope("s1")
        assert sc.get("s1", "a") is None
        assert sc.get("s1", "b") is None
        assert sc.get("s2", "c") == b"3", "其它 scope 不应被清掉"


# ============================ build_shared_cache ============================
class TestBuildSharedCache:
    def test_no_url_returns_none(self, monkeypatch):
        monkeypatch.delenv("CACHE_STORAGE_URL", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
        assert build_shared_cache() is None


# ============================ 集成：pipeline 走共享缓存 ============================
class _MemShared:
    """内存版共享缓存替身，验证 pipeline 与 SharedCache 的契约。"""
    def __init__(self):
        self.store = {}

    def get(self, scope, key):
        return self.store.get((scope, key))

    def set(self, scope, key, value, ttl):
        self.store[(scope, key)] = value

    def clear_scope(self, scope):
        for (s, k) in list(self.store):
            if s == scope:
                del self.store[(s, k)]


class _Store:
    def __init__(self):
        self.collection_name = "test_kb"
        self._mutation_version = 0
        self.query_calls = 0

    def count(self):
        return 10

    def query(self, query_text, top_k, where=None, score_threshold=0.0):
        self.query_calls += 1
        return [Hit(id="h1", text="正文", score=0.9, metadata={"source": "a.md"})]


def _pipeline(store):
    p = object.__new__(RAGPipeline)
    p.config = {"retrieval": {"answer_cache_enabled": True, "answer_cache_ttl_seconds": 86400,
                              "cache_ttl_seconds": 900}}
    p.vector_store = store
    p._answer_cache = {}
    p._answer_cache_lock = threading.Lock()
    p._retrieval_cache = {}
    p._retrieval_cache_lock = threading.Lock()
    p._shared_cache = _MemShared()
    return p


class TestPipelineSharedCache:
    def test_retrieval_cache_shared_across_processes(self):
        store = _Store()
        p = _pipeline(store)

        first = p._cached_vector_query("问题", 8, None)
        assert store.query_calls == 1
        assert first and first[0].id == "h1"

        # 模拟"另一个进程"：清空进程内缓存，共享缓存仍在
        p._retrieval_cache.clear()
        store.query_calls = 0
        second = p._cached_vector_query("问题", 8, None)
        assert second and second[0].id == "h1"
        assert store.query_calls == 0, "共享缓存命中后不应再次真实检索"

    def test_answer_cache_shared_across_processes(self):
        store = _Store()
        p = _pipeline(store)
        key = p._answer_cache_key("问题", 8, None, None)

        res = RAGResult(answer="答案", sources=[{"index": 1, "title": "a.md"}],
                        raw_hits=[Hit(id="h1", text="正文", score=0.9, metadata={"source": "a.md"})])
        p._answer_cache_put(key, res)

        # 模拟另一进程：清空进程内缓存
        p._answer_cache.clear()
        got = p._answer_cache_get(key)
        assert got is not None and got.answer == "答案"

    def test_invalidate_clears_shared_scope(self):
        store = _Store()
        p = _pipeline(store)
        key = p._answer_cache_key("问题", 8, None, None)
        p._answer_cache_put(key, RAGResult(answer="答案"))
        assert p._shared_cache.get(p._cache_scope(), p._shared_key("a", key)) is not None

        p.invalidate_answer_cache()
        assert p._shared_cache.get(p._cache_scope(), p._shared_key("a", key)) is None
