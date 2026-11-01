from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.extractive_answer import build_extractive_answer


def test_extractive_answer_has_only_source_sentences_and_citations():
    chunks = [
        {
            "index": 1,
            "content": (
                "公司年假制度：员工入职满一年可休五天年假。"
                "申请年假需提前三个工作日提交审批。"
            ),
        },
        {"index": 2, "content": "差旅报销需在出差结束后三十日内提交。"},
    ]
    answer = build_extractive_answer("年假怎么申请？", chunks, max_chunks=2)
    assert "年假" in answer
    assert "[1]" in answer
    assert "差旅报销" not in answer
    assert all(line.startswith("- ") and "[" in line for line in answer.splitlines())


def test_pipeline_extractive_mode_never_loads_llm():
    from src.prompt_template import PromptTemplate
    from src.rag_pipeline import RAGPipeline
    from src.vector_store import Hit

    class _Store:
        def query(self, query_text, top_k, where=None, score_threshold=0.0):
            return [
                Hit(
                    id="1",
                    text="公司年假制度：员工入职满一年可休五天年假。申请需提前三个工作日。",
                    score=0.9,
                    metadata={"source": "员工手册.md", "page": 1},
                )
            ]

    pipeline = object.__new__(RAGPipeline)
    pipeline.config = {
        "llm": {"backend": "extractive"},
        "vector_store": {},
        "retrieval": {"min_score": 0.0, "status_filter": False, "acl_filter": False},
        "reranker": {"enabled": False},
        "rag": {
            "extractive": {"max_chunks": 1, "max_sentences_per_chunk": 2},
            "verification": {"enabled": False},
        },
        "decision": {},
        "knowledge_graph": {"enabled": False},
    }
    pipeline.vector_store = _Store()
    pipeline.reranker = None
    pipeline.llm = None
    pipeline._decision_state_checked = True
    pipeline._graph_state_checked = True
    pipeline._graph_retriever = None
    pipeline.context_assembler = SimpleNamespace(enabled=False)
    pipeline.prompts = PromptTemplate({})

    result = pipeline.answer("年假怎么申请？")
    assert "年假" in result.answer
    assert "[1]" in result.answer
    assert result.timings["answer_mode"] == "extractive"
    assert pipeline.llm is None


def test_onnx_embedding_cpu_backend():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from pathlib import Path

    model = Path("models/bge-small-zh-v1.5-onnx/onnx/model_quantized.onnx")
    if not model.exists():
        pytest.skip("ONNX embedding model is not installed")

    from src.onnx_embedding import OnnxEmbedding

    embedding = OnnxEmbedding("models/bge-small-zh-v1.5-onnx", num_threads=2)
    vectors = embedding.encode(
        ["年假怎么申请？", "员工可提前申请年假。", "今天天气晴朗。"], batch_size=3
    )
    assert vectors.shape == (3, 512)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])
