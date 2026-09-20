"""``src.vector_store`` 的单元测试（P3.2）。

用 ``unittest.mock.MagicMock`` 替换 ``chromadb``；不实际启动持久化客户端。
覆盖：
- ``query()`` 返回 :class:`Hit` 列表，score 是 1-distance。
- ``delete_by_metadata({})`` 抛 ValueError（避免误删）。
- ``list_sources()`` 按 source 聚合 chunk 数并按 chunk 数降序。
- ``add_chunks`` 调用 ``upsert`` 时正确切片。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.vector_store import ChromaStore, Hit


# ----------------------------------------------------------------------
#  Fake chunks / embedding
# ----------------------------------------------------------------------


@dataclass
class _FakeChunk:
    text: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class _FakeEmbedding:
    dim = 4

    def encode(self, texts, batch_size=8, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
        import numpy as np

        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            arr[i] = [(hash(t) >> j) & 0xFF for j in range(self.dim)]
        return arr


# ----------------------------------------------------------------------
#  Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def store_with_mock_client(tmp_path, monkeypatch):
    """构造一个 ``ChromaStore``，其中 ``chromadb.PersistentClient`` 返回 MagicMock。"""
    fake_collection = MagicMock(name="fake_collection")
    fake_collection.count.return_value = 0  # 默认空

    fake_settings = MagicMock(name="fake_settings")
    fake_client = MagicMock(name="fake_client")
    fake_client.get_or_create_collection.return_value = fake_collection

    fake_chromadb = MagicMock()
    fake_chromadb.PersistentClient.return_value = fake_client

    fake_chromadb_config = MagicMock()
    fake_chromadb_config.Settings.return_value = fake_settings

    monkeypatch.setitem(__import__("sys").modules, "chromadb", fake_chromadb)
    monkeypatch.setitem(
        __import__("sys").modules,
        "chromadb.config",
        fake_chromadb_config,
    )

    with patch("src.vector_store.Timer"):
        store = ChromaStore(
            persist_directory=str(tmp_path),
            collection_name="test_coll",
            embedding_model=_FakeEmbedding(),
        )

    # 让 collection 引用透明
    store.collection = fake_collection
    store.embedding_model = _FakeEmbedding()
    return store, fake_collection


# ----------------------------------------------------------------------
#  Tests
# ----------------------------------------------------------------------


def test_query_returns_hit_list(store_with_mock_client):
    store, coll = store_with_mock_client
    coll.count.return_value = 3  # non-empty so query() proceeds

    coll.query.return_value = {
        "ids": [["c1", "c2", "c3"]],
        "documents": [["text1", "text2", "text3"]],
        "metadatas": [[{"source": "a.txt"}, {"source": "a.txt"}, {"source": "b.txt"}]],
        "distances": [[0.1, 0.4, 0.9]],
    }

    hits = store.query("hello", top_k=3)
    assert isinstance(hits, list)
    assert len(hits) == 3
    for h in hits:
        assert isinstance(h, Hit)
    # score = max(0, 1 - distance)
    assert hits[0].score == pytest.approx(0.9)
    assert hits[1].score == pytest.approx(0.6)
    assert hits[2].score == pytest.approx(0.1)
    # 传给底层
    args, kwargs = coll.query.call_args
    assert kwargs["n_results"] == 3


def test_query_filters_by_score_threshold(store_with_mock_client):
    store, coll = store_with_mock_client
    coll.count.return_value = 2
    coll.query.return_value = {
        "ids": [["c1", "c2"]],
        "documents": [["t1", "t2"]],
        "metadatas": [[{}, {}]],
        "distances": [[0.1, 0.9]],  # scores = 0.9, 0.1
    }
    hits = store.query("x", score_threshold=0.5)
    # only c1 should survive
    assert len(hits) == 1
    assert hits[0].id == "c1"


def test_query_returns_empty_when_collection_empty(store_with_mock_client):
    store, coll = store_with_mock_client
    coll.count.return_value = 0
    assert store.query("anything") == []
    coll.query.assert_not_called()


def test_delete_by_metadata_empty_raises(store_with_mock_client):
    store, coll = store_with_mock_client
    with pytest.raises(ValueError):
        store.delete_by_metadata({})


def test_delete_by_metadata_returns_count(store_with_mock_client):
    store, coll = store_with_mock_client
    coll.count.side_effect = [10, 7]  # before/after
    coll.delete.return_value = None

    deleted = store.delete_by_metadata({"source": "a.txt"})
    assert deleted == 3
    kwargs = coll.delete.call_args.kwargs
    assert kwargs["where"] == {"source": "a.txt"}


def test_list_sources_aggregates(store_with_mock_client):
    store, coll = store_with_mock_client
    coll.count.return_value = 5
    coll.get.return_value = {
        "metadatas": [
            {"source": "a.txt"},
            {"source": "a.txt"},
            {"source": "a.txt"},
            {"source": "b.txt"},
            {"source": "c.txt", "filepath": "fallback.txt"},
        ]
    }
    sources = store.list_sources()
    # 按 chunks 降序：a.txt=3, c.txt=1 (filepath fallback), b.txt=1
    assert sources[0]["source"] == "a.txt"
    assert sources[0]["chunks"] == 3
    # c.txt 的 filepath 优先级低于 source；b/c 各 1
    assert sum(s["chunks"] for s in sources) == 5
    assert all(s["chunks"] >= 1 for s in sources)


def test_list_sources_empty(store_with_mock_client):
    store, coll = store_with_mock_client
    coll.count.return_value = 0
    assert store.list_sources() == []


def test_add_chunks_uses_upsert(store_with_mock_client):
    store, coll = store_with_mock_client
    chunks = [
        _FakeChunk(text="hello world", index=0, metadata={"source": "a.txt", "page": 1}),
        _FakeChunk(text="foo bar", index=1, metadata={"source": "a.txt", "page": 1}),
    ]
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ]
    n = store.add_chunks(chunks, embeddings=embeddings)
    assert n == 2
    assert coll.upsert.called
    args, kwargs = coll.upsert.call_args
    assert len(kwargs["ids"]) == 2
    assert kwargs["documents"] == ["hello world", "foo bar"]
    assert len(kwargs["embeddings"]) == 2
    assert all(md["source"] == "a.txt" for md in kwargs["metadatas"])


def test_make_id_is_deterministic():
    """不同 source/page/index 应给出不同的稳定 id。"""
    from src.vector_store import _make_id

    a = _make_id("a.txt", 1, 0)
    b = _make_id("a.txt", 1, 1)
    c = _make_id("b.txt", 1, 0)
    assert a != b != c
    assert _make_id("a.txt", 1, 0) == a  # stable


def test_tokenize_jieba_word_level():
    """装了 jieba 时 CJK 按词切分；停用词（的/了/和…）被过滤。"""
    from src.vector_store import _tokenize

    toks = _tokenize("引用了多少篇参考文献")
    assert "参考文献" in toks, f"jieba 词级切分失效: {toks}"
    assert "的" not in toks and "了" not in toks, f"停用词未过滤: {toks}"
    # 拉丁词保持小写整词
    assert "rag" in _tokenize("RAG系统")


def test_tokenize_fallback_bigram_when_no_jieba(monkeypatch):
    """jieba 不可用时回退 unigram+bigram，功能不中断。"""
    import src.vector_store as vs_mod

    monkeypatch.setattr(vs_mod, "_JIEBA_STATE", "off")
    toks = vs_mod._tokenize("文献")
    assert "文献" in toks and "文" in toks, f"bigram 回退失效: {toks}"
