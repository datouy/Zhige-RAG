"""P1-2：``get_by_sources`` 批量读取 + 邻居回填 N+1 收敛 —— 回归测试。

原始缺陷：``_expand_with_neighbors`` / ``_expand_to_fulltext`` 对每个命中
``source`` 各调一次 ``get_by_source``（N+1）。一批命中来自多个文档时会被放大
成多次 Chroma 往返。新增 ``ChromaStore.get_by_sources`` 用一次 ``$in`` 批量取回，
两个扩展路径都改走它，并为不具备该方法的替身保留 ``get_by_source`` 回退。
"""
from __future__ import annotations

from src.rag_pipeline import RAGPipeline
from src.vector_store import ChromaStore, Hit


def _hit(id: str, source: str, chunk_index: int, score: float = 0.9) -> Hit:
    return Hit(id=id, text=id, score=score, metadata={"source": source, "chunk_index": chunk_index})


class _FakeCollection:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, where=None, include=None):
        self.calls.append(where)
        return self.response


def _store(response) -> ChromaStore:
    s = object.__new__(ChromaStore)
    s.collection = _FakeCollection(response)
    return s


def test_get_by_sources_single_in_query_and_sorted():
    store = _store({
        "ids": ["b1", "a0", "a1"],
        "documents": ["B1", "A0", "A1"],
        "metadatas": [
            {"source": "b.md", "chunk_index": 1},
            {"source": "a.md", "chunk_index": 0},
            {"source": "a.md", "chunk_index": 1},
        ],
    })
    grouped = store.get_by_sources(["a.md", "b.md"], where={"doc_status": "active"})

    assert set(grouped) == {"a.md", "b.md"}
    # a.md 应按 chunk_index 升序（0 在 1 前），即使存储顺序是乱的
    assert [h.id for h in grouped["a.md"]] == ["a0", "a1"]
    assert [h.id for h in grouped["b.md"]] == ["b1"]

    # 只应调用一次 collection.get，且 where 用 $in + $and 合并
    assert len(store.collection.calls) == 1
    where = store.collection.calls[0]
    assert where["$and"][0] == {"source": {"$in": ["a.md", "b.md"]}}
    assert where["$and"][1] == {"doc_status": "active"}


def test_get_by_sources_empty_input_returns_empty_dict():
    store = _store({"ids": [], "documents": [], "metadatas": []})
    assert store.get_by_sources([]) == {}
    assert store.collection.calls == []


class _NeighborStore:
    """只提供 get_by_sources、不提供 get_by_source，用于证明批量路径被优先使用。"""
    def __init__(self, grouped):
        self.grouped = grouped
        self.calls = []

    def get_by_sources(self, sources, where=None):
        self.calls.append(sources)
        return self.grouped


def _pipeline(store) -> RAGPipeline:
    p = object.__new__(RAGPipeline)
    p.config = {"retrieval": {"neighbor_expand": True, "neighbor_window": 1,
                              "status_filter": False, "acl_filter": False}}
    p.vector_store = store
    return p


def test_expand_with_neighbors_uses_batch_instead_of_n_plus_1():
    grouped = {
        "a.md": [_hit("a0", "a.md", 0), _hit("a1", "a.md", 1), _hit("a2", "a.md", 2)],
        "b.md": [_hit("b0", "b.md", 0)],
    }
    store = _NeighborStore(grouped)
    p = _pipeline(store)

    hits = [_hit("a1", "a.md", 1), _hit("b0", "b.md", 0)]
    out = p._expand_with_neighbors(hits, where=None, allowed_groups=None)

    # 原命中仍在，且补入 a.md 的 chunk 0/2 两个邻居
    ids = {h.id for h in out}
    assert {"a1", "b0", "a0", "a2"} <= ids
    # 批量接口只被调用一次（不再是每个 source 一次）
    assert store.calls == [["a.md", "b.md"]]
