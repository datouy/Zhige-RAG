"""答案缓存：库版本入键 + 副本返回 —— 回归测试。

两个真实缺陷：

1. **缓存键不含库版本**
   答案缓存 TTL 是 24 小时。任何"改了知识库但没走到 invalidate_answer_cache"
   的路径（直接改库、换 embedding 模型重灌、后台任务入库、删除文档），
   都会让旧答案在 24 小时内继续被复用 —— 用户明明补了文档，系统还在答
   "没找到相关信息"。把分块总数（内容版本号）纳入缓存键后，内容一变键就变，
   旧答案自动失效，不需要谁来记得调 invalidate。

2. **缓存返回可变引用**
   ``_answer_cache_get`` 原本直接返回缓存里那个 :class:`RAGResult` 对象。
   调用方会就地改写它（``res.timings["cache_hit"] = True``），于是第一次命中
   就把缓存本身改了：后面所有命中都带着上一次调用的脏数据，计时信息越写越乱。
   必须返回深拷贝。
"""

from __future__ import annotations

import threading

from src.rag_pipeline import RAGPipeline, RAGResult
from src.vector_store import Hit


class _Store:
    """最小可用的向量库替身：只提供 count() 与 collection_name。"""

    def __init__(self, n: int = 10):
        self._n = n
        self.collection_name = "test_kb"

    def count(self) -> int:
        return self._n

    def query(self, query_text, top_k, where=None, score_threshold=0.0):
        return []


def _pipeline(store: _Store) -> RAGPipeline:
    p = object.__new__(RAGPipeline)
    p.config = {
        "retrieval": {
            "answer_cache_enabled": True,
            "answer_cache_ttl_seconds": 86400,
            "min_score": 0.0,
            "fetch_k": 24,
        }
    }
    p.vector_store = store
    p._answer_cache = {}
    p._answer_cache_lock = threading.Lock()
    p._retrieval_cache = {}
    p._retrieval_cache_lock = threading.Lock()
    return p


def _result(answer: str = "答案") -> RAGResult:
    return RAGResult(
        answer=answer,
        sources=[{"index": 1, "title": "a.md", "score": 0.8}],
        raw_hits=[Hit(id="h1", text="正文", score=0.8, metadata={"source": "a.md"})],
        timings={"llm_total_ms": 123.0},
    )


# ============================ 库版本入键 ============================
class TestCacheKeyIncludesCollectionVersion:
    def test_key_changes_when_collection_count_changes(self):
        store = _Store(n=10)
        p = _pipeline(store)
        k1 = p._answer_cache_key("问题", 8, None, None)

        store._n = 11  # 入库了一个分块
        k2 = p._answer_cache_key("问题", 8, None, None)
        assert k1 != k2, "分块数变化后缓存键必须变化，否则旧答案会继续被复用"

    def test_key_is_stable_when_nothing_changed(self):
        p = _pipeline(_Store(n=10))
        assert p._answer_cache_key("问题", 8, None, None) == p._answer_cache_key(
            "问题", 8, None, None
        )

    def test_version_is_none_when_count_unavailable(self):
        """取不到分块数时退化成 None，不应抛异常。"""

        class _Broken(_Store):
            def count(self):
                raise RuntimeError("chroma 不可用")

        p = _pipeline(_Broken())
        assert p._collection_version() is None
        p._answer_cache_key("问题", 8, None, None)  # 不应抛

    def test_different_collections_do_not_share_cache(self):
        s1, s2 = _Store(n=10), _Store(n=10)
        s2.collection_name = "other_kb"
        p1, p2 = _pipeline(s1), _pipeline(s2)
        assert p1._answer_cache_key("问题", 8, None, None) != p2._answer_cache_key(
            "问题", 8, None, None
        ), "不同知识库不能共用同一个缓存键"


# ============================ 副本返回 ============================
class TestCacheReturnsCopy:
    def test_get_returns_distinct_object(self):
        p = _pipeline(_Store())
        key = p._answer_cache_key("问题", 8, None, None)
        original = _result()
        p._answer_cache_put(key, original)

        got = p._answer_cache_get(key)
        assert got is not None
        assert got is not original, "缓存命中必须返回副本，不能是缓存里那一份"

    def test_mutating_returned_result_does_not_pollute_cache(self):
        """命中后改 timings，不得写回缓存。"""
        p = _pipeline(_Store())
        key = p._answer_cache_key("问题", 8, None, None)
        p._answer_cache_put(key, _result())

        first = p._answer_cache_get(key)
        first.timings["cache_hit"] = True
        first.timings["llm_total_ms"] = 999.0

        second = p._answer_cache_get(key)
        assert "cache_hit" not in second.timings, "缓存内容被上一次命中改脏了"
        assert second.timings["llm_total_ms"] == 123.0, "缓存里的计时被改写了"

    def test_mutating_sources_does_not_pollute_cache(self):
        p = _pipeline(_Store())
        key = p._answer_cache_key("问题", 8, None, None)
        p._answer_cache_put(key, _result())

        first = p._answer_cache_get(key)
        first.sources.append({"index": 999})

        second = p._answer_cache_get(key)
        assert len(second.sources) == 1, "引用列表被外部追加后污染了缓存"

    def test_mutating_raw_hit_metadata_does_not_pollute_cache(self):
        p = _pipeline(_Store())
        key = p._answer_cache_key("问题", 8, None, None)
        p._answer_cache_put(key, _result())

        first = p._answer_cache_get(key)
        first.raw_hits[0].metadata["source"] = "被改了"

        second = p._answer_cache_get(key)
        assert second.raw_hits[0].metadata["source"] == "a.md"

    def test_put_stores_a_copy(self):
        """存进去的也应是副本：调用方后续改动不影响缓存内容。"""
        p = _pipeline(_Store())
        key = p._answer_cache_key("问题", 8, None, None)
        res = _result()
        p._answer_cache_put(key, res)

        res.answer = "被改了"
        got = p._answer_cache_get(key)
        assert got.answer == "答案"

    def test_expiry_still_works(self):
        p = _pipeline(_Store())
        p.config["retrieval"]["answer_cache_ttl_seconds"] = 0.0
        key = p._answer_cache_key("问题", 8, None, None)
        p._answer_cache_put(key, _result())
        assert p._answer_cache_get(key) is None, "过期后不应命中"

    def test_disabled_cache_never_stores(self):
        p = _pipeline(_Store())
        p.config["retrieval"]["answer_cache_enabled"] = False
        key = p._answer_cache_key("问题", 8, None, None)
        p._answer_cache_put(key, _result())
        assert p._answer_cache_get(key) is None


# ============================ 流式路径也走缓存 ============================
class TestStreamPathUsesCache:
    """Web UI 走的是 WS 流式；此前只有非流式 answer() 走缓存，
    等于"同问免二次生成"这个优化在主界面上完全没生效。"""

    def _pipeline(self):
        p = _pipeline(_Store())
        # 流式路径会先做意图分类等准备，这里直接绕过到缓存命中分支：
        # 用一个 history 非空会让缓存跳过，所以这里保持 history 为空。
        p._classify = lambda q: type(
            "D", (), {"intent": "qa", "reply": "", "entities": []}
        )()
        p.context_assembler = type(
            "CA", (), {"enabled": False, "build": lambda *a, **k: []}
        )()
        p._rewrite_retrieval_query = lambda q, h: q
        p._answer_cache_put(
            p._answer_cache_key("问题", 8, None, None),
            RAGResult(
                answer="缓存里的答案",
                sources=[{"index": 1, "title": "a.md"}],
                verification={"grounded": True},
            ),
        )
        return p

    def test_stream_hits_cache_and_skips_generation(self):
        p = self._pipeline()
        p._retrieve = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("缓存命中时不应再走检索")
        )
        events = list(p.stream_answer("问题", top_k=8))
        kinds = [e["event"] for e in events]
        assert kinds[0] == "hits"
        assert kinds[-1] == "done"
        tokens = [e["data"] for e in events if e["event"] == "token"]
        assert "".join(tokens) == "缓存里的答案"

    def test_stream_done_carries_cached_verification(self):
        p = self._pipeline()
        p._retrieve = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("缓存命中时不应再走检索")
        )
        events = list(p.stream_answer("问题", top_k=8))
        done = [e for e in events if e["event"] == "done"][0]
        assert done["verification"] == {"grounded": True}
        assert done["data"][0]["title"] == "a.md"
