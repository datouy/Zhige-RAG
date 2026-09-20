"""Build knowledge graph from raw documents.

Usage:
    python -m scripts.build_kg --input data/raw --output data/kg.db --limit 5

This script reads documents from a directory, splits them into chunks,
extracts entities/relations using KGExtractor, and stores them into KGStore.
The LLM is optional; if not provided, only lexical fallback triples are added.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.document_loader import load_directory  # noqa: E402
from src.kg import KGExtractor, create_kg_store  # noqa: E402
from src.llm import LocalLLM  # noqa: E402
from src.text_splitter import ChineseTextSplitter, RecursiveTextSplitter  # noqa: E402
from src.utils import ensure_dir, get_logger, load_config, resolve_path  # noqa: E402

logger = get_logger("build_kg")


def get_splitter(cfg: dict):
    sp = cfg.get("text_splitter", {}) or {}
    strategy = sp.get("strategy", "chinese")
    if strategy == "chinese":
        return ChineseTextSplitter(
            chunk_size=sp.get("chunk_size", 300),
            chunk_overlap=sp.get("chunk_overlap", 50),
            separators=sp.get("chinese_separators") or sp.get("separators"),
            keep_separator=sp.get("keep_separator", True),
            min_chunk_size=sp.get("min_chunk_size", 32),
        )
    return RecursiveTextSplitter(
        chunk_size=sp.get("chunk_size", 512),
        chunk_overlap=sp.get("chunk_overlap", 64),
    )


def build(
    input_dir: str,
    output_path: Optional[str] = None,
    backend: Optional[str] = None,
    limit: Optional[int] = None,
    llm: Optional[LocalLLM] = None,
    config_path: str = "config/config.yaml",
    store=None,
) -> Dict:
    """构建知识图谱。

    Args:
        store: 已构造好的 KGStore（多租户场景由路由传入用户专属 store）。
            传入时忽略 output_path/backend，**不会**再按配置创建全局库——
            之前 API 路由虽然构造了租户 pipeline，但抽取结果全部写进了
            全局 data/kg.db，租户隔离被破坏。
    """
    cfg = load_config(config_path)
    kg_cfg = cfg.get("knowledge_graph", {}) or {}

    target = resolve_path(input_dir)
    if not target.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    docs = load_directory(
        target,
        pdf_engine=cfg["document_loader"].get("pdf_engine", "pdfplumber"),
        recursive=True,
        encoding=cfg["document_loader"].get("encoding", "utf-8"),
    )
    if limit is not None and limit > 0:
        docs = docs[:limit]
    if not docs:
        logger.warning("目录中没有可处理的文档: %s", input_dir)
        return {"entities": 0, "relations": 0, "documents": 0, "triples": 0}

    splitter = get_splitter(cfg)
    chunks = []
    for d in docs:
        try:
            parts = splitter.split_text(d.content, metadata=d.metadata)
        except Exception:
            parts = []
        for p in parts:
            chunks.append(
                {
                    "text": p.text,
                    "source_doc": d.metadata.get("source") or d.metadata.get("filepath") or d.metadata.get("filename", ""),
                    "chunk_id": str(p.metadata.get("id") or f"{p.metadata.get('source', 'doc')}-{p.index}"),
                    "metadata": p.metadata,
                }
            )

    if store is None:
        store_kg_cfg = kg_cfg
        if output_path:
            store_kg_cfg = {**store_kg_cfg, "backend": backend or kg_cfg.get("backend", "sqlite"), "sqlite_path": output_path}
        if backend:
            store_kg_cfg = {**store_kg_cfg, "backend": backend}
        store = create_kg_store(store_kg_cfg)

    extractor = None
    if llm is not None:
        extractor_cfg = kg_cfg.get("extractor", {}) or {}
        extractor = KGExtractor(
            llm=llm,
            max_entities_per_chunk=int(extractor_cfg.get("max_entities_per_chunk", 20)),
            max_relations_per_chunk=int(extractor_cfg.get("max_relations_per_chunk", 30)),
        )

    entities, relations, triples = [], [], []
    if extractor is not None and chunks:
        entities, relations, triples = extractor.extract_from_chunks(chunks)
    if entities:
        store.upsert_entities(entities)
    if relations:
        store.upsert_relations(relations)

    counts = store.count()
    return {
        "entities": counts["entities"],
        "relations": counts["relations"],
        "triples": len(triples),
        "documents": len(docs),
        "chunks": len(chunks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="构建知识图谱")
    parser.add_argument("--input", required=True, help="输入目录")
    parser.add_argument("--output", default=None, help="SQLite 路径（覆盖配置）")
    parser.add_argument("--backend", choices=["sqlite", "neo4j"], default=None)
    parser.add_argument("--limit", type=int, default=None, help="限制处理文档数")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--use-llm", action="store_true", help="使用 LLM 抽取（需要本地模型）")
    parser.add_argument("--print-only", action="store_true", help="仅打印摘要，不写库")
    args = parser.parse_args()

    llm = None
    if args.use_llm:
        cfg = load_config(args.config)
        try:
            llm = LocalLLM(
                model_name=cfg["llm"]["model_name"],
                cache_dir=cfg["llm"].get("cache_dir"),
                local_files_only=cfg["llm"].get("local_files_only", False),
            )
        except Exception as exc:
            logger.warning("LLM 加载失败，将跳过抽取：%s", exc)
            llm = None

    if args.print_only:
        cfg = load_config(args.config)
        target = resolve_path(args.input)
        docs = load_directory(target, pdf_engine=cfg["document_loader"].get("pdf_engine", "pdfplumber"))
        if args.limit:
            docs = docs[: args.limit]
        summary = {
            "documents": len(docs),
            "input_dir": str(target),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    summary = build(
        input_dir=args.input,
        output_path=args.output,
        backend=args.backend,
        limit=args.limit,
        llm=llm,
        config_path=args.config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())