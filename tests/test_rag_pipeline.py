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