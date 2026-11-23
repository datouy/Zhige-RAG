"""固定语料的检索质量硬门禁。

在临时 Chroma 目录中构建小型企业语料，运行真实 ONNX Embedding + BM25/RRF
检索，不读取生产知识库，也不依赖 LLM。CI 中任一阈值未通过即返回非零状态。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from argparse import Namespace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate import check_thresholds, evaluate_retrieval, load_eval_dataset
from src.data_quality import run_data_pipeline
from src.document_meta import enrich_directory
from src.rag_pipeline import RAGPipeline
from src.text_splitter import ChineseTextSplitter, RecursiveTextSplitter
from src.utils import apply_env_overrides, get_logger, load_config

logger = get_logger("ci_eval")


def _splitter(cfg: dict[str, Any]):
    section = cfg.get("text_splitter") or {}
    strategy = str(section.get("strategy", "chinese"))
    if strategy == "chinese":
        return ChineseTextSplitter(
            chunk_size=int(section.get("chunk_size", 400)),
            chunk_overlap=int(section.get("chunk_overlap", 50)),
            separators=section.get("chinese_separators") or section.get("separators"),
            keep_separator=bool(section.get("keep_separator", True)),
            min_chunk_size=int(section.get("min_chunk_size", 32)),
        )
    return RecursiveTextSplitter(
        chunk_size=int(section.get("chunk_size", 512)),
        chunk_overlap=int(section.get("chunk_overlap", 64)),
    )


def run_gate(
    *,
    corpus: Path,
    dataset: Path,
    report: Path,
    min_samples: int,
    min_hit_rate: float,
    min_ndcg: float,
    max_p95_latency_ms: float,
) -> dict[str, Any]:
    if not corpus.exists():
        raise FileNotFoundError(f"评估语料不存在：{corpus}")
    items = load_eval_dataset(dataset)
    if len(items) < min_samples:
        raise RuntimeError(f"评估样本 {len(items)} < {min_samples}")

    temp = Path(tempfile.mkdtemp(prefix="zhige-ci-eval-"))
    try:
        # from_config 会最后应用环境变量；这里显式保证 CI 不连生产 pgvector。
        os.environ["VECTOR_STORE_BACKEND"] = "chroma"
        os.environ["VECTOR_STORE_PERSIST_DIR"] = str(temp / "chroma")
        cfg = apply_env_overrides(load_config(str(ROOT / "config" / "config.yaml")))
        cfg.setdefault("llm", {})["backend"] = "extractive"
        cfg.setdefault("decision", {})["enabled"] = False
        cfg.setdefault("knowledge_graph", {})["enabled"] = False
        cfg.setdefault("reranker", {})["enabled"] = False
        retrieval = cfg.setdefault("retrieval", {})
        retrieval["multiquery_enabled"] = False
        retrieval["llm_query_rewrite"] = False
        retrieval["neighbor_expand"] = False
        retrieval["top_k"] = 5
        cfg.setdefault("rag", {})["top_k"] = 5
        cfg.setdefault("vector_store", {})["backend"] = "chroma"
        cfg["vector_store"]["persist_directory"] = str(temp / "chroma")
        cfg["vector_store"]["collection_name"] = f"ci_eval_{uuid.uuid4().hex[:8]}"

        pipeline = RAGPipeline.from_config(
            str(ROOT / "config" / "config.yaml"),
            overrides=cfg,
            lazy_llm=True,
        )
        docs = enrich_directory(corpus, default_acl="*", recursive=True)
        chunks, ingest_report = run_data_pipeline(docs, cfg, _splitter(cfg))
        if not chunks:
            raise RuntimeError("CI 评估语料没有产生可检索分块")
        for chunk in chunks:
            chunk.metadata["kb_id"] = "ci-eval"
        sources = sorted(
            {
                str((chunk.metadata or {}).get("source") or "unknown")
                for chunk in chunks
            }
        )
        pipeline.vector_store.replace_chunks_for_sources(chunks, sources)
        summary = evaluate_retrieval(pipeline, items)
    finally:
        # Chroma 在 Windows 上会保持内存映射文件句柄；允许残留清理失败。
        shutil.rmtree(temp, ignore_errors=True)

    args = Namespace(
        min_samples=min_samples,
        min_hit_rate=min_hit_rate,
        min_ndcg=min_ndcg,
        min_verified_rate=None,
        max_p95_latency_ms=max_p95_latency_ms,
    )
    failures = check_thresholds(summary, args)
    summary["ingest"] = ingest_report
    summary["thresholds"] = {
        "min_samples": min_samples,
        "min_hit_rate": min_hit_rate,
        "min_ndcg": min_ndcg,
        "max_p95_latency_ms": max_p95_latency_ms,
    }
    summary["failures"] = failures
    summary["status"] = "passed" if not failures else "failed"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError("质量门禁未通过：" + "; ".join(failures))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="固定语料 RAG 质量硬门禁")
    parser.add_argument(
        "--corpus", default="tests/fixtures/eval_corpus"
    )
    parser.add_argument(
        "--dataset", default="tests/fixtures/eval_questions.jsonl"
    )
    parser.add_argument("--report", default="data/eval/ci_gate_report.json")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--min-hit-rate", type=float, default=0.8)
    parser.add_argument("--min-ndcg", type=float, default=0.7)
    parser.add_argument("--max-p95-latency-ms", type=float, default=3000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = Path(args.corpus)
    dataset = Path(args.dataset)
    report = Path(args.report)
    try:
        summary = run_gate(
            corpus=corpus,
            dataset=dataset,
            report=report,
            min_samples=args.min_samples,
            min_hit_rate=args.min_hit_rate,
            min_ndcg=args.min_ndcg,
            max_p95_latency_ms=args.max_p95_latency_ms,
        )
    except Exception as exc:
        logger.error("CI 评估失败：%s", exc)
        return 1
    print(json.dumps(
        {
            "status": summary["status"],
            "samples": summary["samples"],
            "retrieval_hit_rate": summary["retrieval_hit_rate"],
            "ndcg_at_5": summary["ndcg_at_5"],
            "p95_retrieval_latency_ms": summary["p95_retrieval_latency_ms"],
            "report": str(report),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
