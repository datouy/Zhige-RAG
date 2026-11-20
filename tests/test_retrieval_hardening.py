"""检索硬化回归：实体词法、BM25 查询分流、索引版本和词典热加载。"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src import vector_store as vs
from src.rag_pipeline import RAGPipeline
from src.vector_store import BM25Index, Hit, _multi_granularity_tokens, _tokenize


def test_identifiers_survive_lexing_and_keep_searchable_parts():
    cases = {
        "GB/T 28181": {"gb/t28181", "gb", "t", "28181"},
        "BGE-M3": {"bge-m3", "bge", "m3"},
        "XYZ-900": {"xyz-900", "xyz", "900"},
        "v1.2.3": {"v1.2.3", "v1", "2", "3"},
        "C++": {"c++", "c"},
        "C#": {"c#", "c"},
    }
    for text, expected in cases.items():
        tokens = set(_tokenize(text))
        assert expected <= tokens, (text, tokens)


def test_cjk_uses_dictionary_words_and_bigrams_together():
    result = vs.reload_custom_dictionary("config/custom_dict.txt")
    assert result["word_count"] >= 200
    tokens = _multi_granularity_tokens("冠状动脉粥样硬化性心脏病")
    assert "冠状动脉粥样硬化性心脏病" in tokens
    assert "冠状" in tokens
    assert "动脉" in tokens


def test_pipeline_sends_instruction_only_to_vector_and_raw_query_to_bm25():
    calls = []

    class _Store:
        collection_name = "test"
        lexical_signature = "sig-1"

        def query(
            self,
            query_text,
            top_k,
            where=None,
            score_threshold=0.0,
            keyword_query=None,
        ):
            calls.append((query_text, keyword_query))
            return [Hit(id="1", text="命中", score=0.9, metadata={})]

    pipe = object.__new__(RAGPipeline)
    pipe.config = {
        "retrieval": {
            "min_score": 0.0,
            "neighbor_expand": False,
            "status_filter": False,
            "acl_filter": False,
        }
    }
    pipe.vector_store = _Store()
    pipe.reranker = None
    pipe._query_variants = lambda query: [query]
    pipe._apply_query_instruction = lambda query: f"BGE指令：{query}"

    assert pipe._retrieve("GB/T 28181", top_k=3, where=None)
    assert calls == [("BGE指令：GB/T 28181", "GB/T 28181")]


def test_bge_instruction_would_pollute_bm25_when_reused_as_lexical_query():
    index = BM25Index(
        ids=["exact", "distractor"],
        texts=[
            "AB 标准编号为 GB/T 28181。",
            "这个句子用于生成表示以检索相关文章。" * 3,
        ],
        metadatas=[{"source": "exact"}, {"source": "distractor"}],
    )
    raw = index.search("GB/T 28181", top_k=2)
    polluted = index.search(
        "为这个句子生成表示以用于检索相关文章：GB/T 28181", top_k=2
    )
    assert raw[0][0] == "exact"
    assert polluted[0][0] == "distractor"


def test_chroma_batch_uses_separate_lexical_queries():
    class _Embedding:
        def encode(self, texts, batch_size=8):
            import numpy as np

            return np.zeros((len(texts), 2), dtype=np.float32)

    class _Collection:
        def query(self, **_kwargs):
            return {
                "ids": [["1"], ["2"]],
                "documents": [["A"], ["B"]],
                "metadatas": [[{}], [{}]],
                "distances": [[0.1], [0.2]],
            }

    store = object.__new__(vs.ChromaStore)
    store.embedding_model = _Embedding()
    store.collection = _Collection()
    store.distance_fn = "cosine"
    store.hybrid_enabled = True
    store.hybrid_fetch_k = 20
    store.hybrid_bm25_weight = 0.5
    lexical_queries = []
    store._fuse_hybrid = lambda query, hits, top_k, where=None: (
        lexical_queries.append(query) or hits
    )

    results = store.query_batch(
        ["BGE指令：苹果", "BGE指令：香蕉"],
        top_k=2,
        keyword_queries=["苹果", "香蕉"],
    )
    assert len(results) == 2
    assert lexical_queries == ["苹果", "香蕉"]


def test_retrieval_cache_distinguishes_lexical_query():
    calls = []

    class _Store:
        collection_name = "test"
        lexical_signature = "sig-1"

        def query(
            self,
            query_text,
            top_k,
            where=None,
            score_threshold=0.0,
            keyword_query=None,
        ):
            calls.append(keyword_query)
            return [Hit(id=str(keyword_query), text="命中", score=0.9, metadata={})]

    pipe = object.__new__(RAGPipeline)
    pipe.vector_store = _Store()
    pipe.config = {"retrieval": {"cache_ttl_seconds": 60}}
    pipe._retrieval_cache = __import__("collections").OrderedDict()
    pipe._retrieval_cache_lock = threading.Lock()

    first = pipe._cached_vector_query("相同向量文本", 5, None, keyword_query="苹果")
    second = pipe._cached_vector_query("相同向量文本", 5, None, keyword_query="香蕉")
    third = pipe._cached_vector_query("相同向量文本", 5, None, keyword_query="苹果")
    assert first[0].id == "苹果"
    assert second[0].id == "香蕉"
    assert third[0].id == "苹果"
    assert calls == ["苹果", "香蕉"]


def test_answer_cache_key_changes_with_lexical_signature():
    class _Store:
        collection_name = "same-kb"
        lexical_signature = "signature-a"

        def count(self):
            return 3

    pipe = object.__new__(RAGPipeline)
    pipe.vector_store = _Store()
    key_a = pipe._answer_cache_key("同一个问题", 8, None, None)
    pipe.vector_store.lexical_signature = "signature-b"
    key_b = pipe._answer_cache_key("同一个问题", 8, None, None)
    assert key_a != key_b


def test_bm25_pickle_with_old_tokenizer_signature_is_rebuilt(tmp_path):
    pickle_path = tmp_path / "bm25_test.pkl"
    stale = BM25Index(
        ids=["old"],
        texts=["GB/T 28181"],
        metadatas=[{"source": "old.md"}],
        tokenizer_signature="stale-v1",
    )
    stale.save(pickle_path)

    class _Collection:
        def count(self):
            return 1

        def get(self, **_kwargs):
            return {
                "ids": ["new"],
                "documents": ["GB/T 28181"],
                "metadatas": [{"source": "new.md"}],
            }

    store = object.__new__(vs.ChromaStore)
    store.collection = _Collection()
    store._bm25 = None
    store._bm25_lock = threading.Lock()
    store._bm25_path = pickle_path

    rebuilt = store._ensure_bm25()
    assert rebuilt.ids == ["new"]
    assert rebuilt.tokenizer_signature == store.lexical_signature
    assert rebuilt.tokenizer_signature != "stale-v1"


def test_dictionary_reload_invalidates_registered_bm25(tmp_path):
    dictionary = tmp_path / "custom_dict.txt"
    dictionary.write_text("专有名称\n检索术语\n", encoding="utf-8")
    pickle_path = tmp_path / "bm25_registered.pkl"

    store = object.__new__(vs.ChromaStore)
    store._bm25 = BM25Index(
        ids=["1"],
        texts=["旧词典内容"],
        metadatas=[{}],
    )
    store._bm25_path = pickle_path
    store._bm25.save(pickle_path)
    vs._ACTIVE_BM25_STORES.add(store)

    try:
        result = vs.reload_custom_dictionary(str(dictionary))
        assert result["word_count"] == 2
        assert store._bm25 is None
        assert not pickle_path.exists()
    finally:
        vs.reload_custom_dictionary("config/custom_dict.txt")


def test_dictionary_reload_api_is_admin_only_and_reports_generation(monkeypatch):
    from api.routes.settings import api_reload_dictionary

    monkeypatch.setattr(
        vs,
        "reload_custom_dictionary",
        lambda path: {
            "word_count": 10,
            "generation": 3,
            "signature": "abc",
            "path": path,
        },
    )
    result = asyncio.run(api_reload_dictionary(SimpleNamespace(is_admin=True)))
    assert result["status"] == "ok"
    assert result["generation"] == 3
    assert result["bm25_rebuild"] == "next_query"

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_reload_dictionary(SimpleNamespace(is_admin=False)))
    assert exc_info.value.status_code == 403
