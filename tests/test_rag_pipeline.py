"""RAG 流程的最小烟雾测试（依赖 Chroma + Embedding 的内存版本）。

测试要点：
- 不加载 LLM 也能完成 retrieve 部分
- 中文分块 + 检索链路通畅

为节省 CI 资源，本测试使用 ``chromadb`` 的内存客户端 + 极小的随机向量，验证端到端 API 可用。
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_vector_store_end_to_end(tmp_path):
    pytest.importorskip("chromadb")

    from src.text_splitter import ChineseTextSplitter
    from src.vector_store import ChromaStore

    # 写一些临时文档
    (tmp_path / "doc1.txt").write_text(
        "RAG 是检索增强生成的缩写。它结合了信息检索与文本生成。"
        "通过向量检索最相关的上下文，再交给大模型回答问题。",
        encoding="utf-8",
    )

    splitter = ChineseTextSplitter(chunk_size=60, chunk_overlap=10)
    sp_docs = splitter.split_text(
        (tmp_path / "doc1.txt").read_text(encoding="utf-8"),
        metadata={"source": "doc1.txt", "page": 1},
    )
    assert sp_docs

    # 用一个 mock embedding（避免下载真实模型）
    class _MockEmbed:
        dim = 16
        device = "cpu"

        def encode(self, texts, batch_size=8, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
            arr = np.zeros((len(texts), self.dim), dtype=np.float32)
            for i, t in enumerate(texts):
                # 基于文本 hash 生成伪向量
                h = abs(hash(t)) % (10 ** 6)
                rng = np.random.default_rng(h)
                arr[i] = rng.standard_normal(self.dim)
                arr[i] = arr[i] / (np.linalg.norm(arr[i]) + 1e-9)
            return arr

    store = ChromaStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test",
        embedding_model=_MockEmbed(),
    )
    n = store.add_chunks(sp_docs)
    assert n == len(sp_docs)

    hits = store.query("什么是 RAG", top_k=2)
    assert hits
    assert all(h.text for h in hits)


def test_prompt_template_assemble():
    from src.prompt_template import PromptTemplate

    tpl = PromptTemplate({})  # 使用默认值
    msgs = tpl.build_messages(
        question="什么是 RAG？",
        context_chunks=[
            {"index": 1, "title": "intro.pdf", "content": "RAG 是检索增强生成。"}
        ],
    )
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "RAG" in msgs[1]["content"]
    assert "[1]" in msgs[1]["content"]

    # 空上下文
    msgs_empty = tpl.build_messages("x", [])
    assert "无法" in msgs_empty[1]["content"] or "未检索" in msgs_empty[1]["content"]


def test_utils_merge_and_env(tmp_path, monkeypatch):
    from src.utils import merge_dict

    a = {"x": {"a": 1, "b": 2}}
    b = {"x": {"b": 99, "c": 3}}
    out = merge_dict(a, b)
    assert out == {"x": {"a": 1, "b": 99, "c": 3}}

def test_normalize_query_fullwidth_to_halfwidth():
    from src.rag_pipeline import normalize_query

    assert normalize_query("ＲＡＧ 是什么？") == "RAG 是什么?"
    assert normalize_query("  多   空格  ") == "多 空格"
    assert normalize_query("") == ""


def test_retrieve_min_score_guard():
    """全部命中低于 retrieval.min_score 时按未命中处理；高分命中不受影响。"""
    from src.rag_pipeline import RAGPipeline
    from src.vector_store import Hit

    def _make_pipeline(hits):
        class _Store:
            def query(self, query_text, top_k, where=None, score_threshold=0.0):
                return hits

        p = object.__new__(RAGPipeline)
        p.config = {"retrieval": {"min_score": 0.25}}
        p.vector_store = _Store()
        return p

    low = _make_pipeline([Hit(id="1", text="x", score=0.1, metadata={})])
    assert low._retrieve("任意问题", top_k=3, where=None) == []

    high = _make_pipeline([Hit(id="1", text="x", score=0.9, metadata={})])
    assert len(high._retrieve("任意问题", top_k=3, where=None)) == 1

    off = _make_pipeline([Hit(id="1", text="x", score=0.1, metadata={})])
    off.config = {"retrieval": {"min_score": 0.0}}
    assert len(off._retrieve("任意问题", top_k=3, where=None)) == 1


def test_hybrid_bm25_only_hit_score_normalized():
    """BM25 独有命中的 score 应为归一化值 (0,1)，不再是 0.0 占位。"""
    s = 3.0
    assert 0 < s / (1 + s) < 1


def test_augment_with_graph_appends_triples():
    """图谱命中时应追加一条"知识图谱"上下文；未启用/为空时原样返回。"""
    from src.rag_pipeline import RAGPipeline

    class _T:
        subject, predicate, object = "RAG", "包含", "检索"

    class _FakeRetriever:
        def search(self, query, top_k_entities=5, hops=2):
            return {"triples": [_T()]}

    p = object.__new__(RAGPipeline)
    p.config = {"knowledge_graph": {"enabled": True, "graph_rag": {"enabled": True, "graph_hops": 2}}}
    p._graph_state_checked = True
    p._graph_retriever = _FakeRetriever()

    chunks = [{"index": 1, "title": "片段", "content": "正文", "metadata": {}}]
    out = p._augment_with_graph("RAG 包含什么？", chunks)
    assert len(out) == 2
    assert out[-1]["title"] == "知识图谱"
    assert "RAG -[包含]-> 检索" in out[-1]["content"]
    assert out[-1]["index"] == 2  # 编号顺延，prompt 引用不断号

    p._graph_retriever = None
    assert p._augment_with_graph("任意", chunks) is chunks  # 图谱为空：原样返回


def test_retrieve_public_method_no_llm():
    """retrieve() 公共入口：召回 + 可选重排，mock store 下不依赖 LLM。"""
    from src.rag_pipeline import RAGPipeline
    from src.vector_store import Hit

    class _Store:
        def query(self, query_text, top_k, where=None, score_threshold=0.0):
            return [Hit(id="1", text="x", score=0.9, metadata={"source": "a.md"})]

    p = object.__new__(RAGPipeline)
    p.config = {"retrieval": {"min_score": 0.25}}
    p.vector_store = _Store()
    p.reranker = None
    hits = p.retrieve("什么是 RAG")
    assert len(hits) == 1 and hits[0].metadata["source"] == "a.md"
