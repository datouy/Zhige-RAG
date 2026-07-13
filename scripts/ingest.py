"""命令行入库脚本：将指定文档或目录写入向量库。

用法：
    python scripts/ingest.py path/to/file.pdf
    python scripts/ingest.py path/to/folder --recursive
    python scripts/ingest.py data/raw --config config/config.yaml
    python scripts/ingest.py data/raw --version  # 同时保存版本到 data/versions/
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# 允许从项目根目录运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.document_loader import load_directory, load_document
from src.embeddings import EmbeddingModel
from src.text_splitter import ChineseTextSplitter, RecursiveTextSplitter
from src.utils import apply_env_overrides, ensure_dir, get_logger, load_config, merge_dict
from src.vector_store import ChromaStore

# 版本管理功能
import scripts.version_manager as version_manager

logger = get_logger("ingest")


def _compute_doc_hash(text: str) -> str:
    """计算文档内容的 SHA256。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_splitter(cfg: dict):
    sp_cfg = cfg.get("text_splitter", {})
    strategy = sp_cfg.get("strategy", "chinese")
    if strategy == "chinese":
        return ChineseTextSplitter(
            chunk_size=sp_cfg.get("chunk_size", 256),
            chunk_overlap=sp_cfg.get("chunk_overlap", 32),
            separators=sp_cfg.get("chinese_separators") or sp_cfg.get("separators"),
            keep_separator=sp_cfg.get("keep_separator", True),
            min_chunk_size=sp_cfg.get("min_chunk_size", 32),
        )
    return RecursiveTextSplitter(
        chunk_size=sp_cfg.get("chunk_size", 512),
        chunk_overlap=sp_cfg.get("chunk_overlap", 64),
    )


def ingest_path(
    target: Path,
    cfg: dict,
    splitter,
    enable_version: bool = False,
) -> int:
    """入库单个文件或目录。

    Args:
        target: 文件或目录路径。
        cfg: 配置字典。
        splitter: 文本分块器。
        enable_version: 是否保存版本到 data/versions/。
    """
    dl_cfg = cfg.get("document_loader", {})

    if target.is_file():
        docs = load_document(
            target,
            pdf_engine=dl_cfg.get("pdf_engine", "pdfplumber"),
            encoding=dl_cfg.get("encoding", "utf-8"),
            clean=dl_cfg.get("clean_text", True),
        )
    elif target.is_dir():
        docs = load_directory(
            target,
            pdf_engine=dl_cfg.get("pdf_engine", "pdfplumber"),
            recursive=True,
            encoding=dl_cfg.get("encoding", "utf-8"),
        )
    else:
        logger.warning("路径不存在: %s", target)
        return 0

    if not docs:
        logger.warning("未解析到任何文档")
        return 0

    # 分块（每个 Document 可能产生多个 Chunk）
    all_chunks: list = []
    doc_chunks_map: dict = {}  # doc.source -> list of serialized chunks

    for doc in docs:
        doc_chunks = splitter.split_text(doc.content, metadata=doc.metadata)
        all_chunks.extend(doc_chunks)
        # 保存版本用的序列化格式
        doc_chunks_map[doc.metadata.get("source", target.name)] = [
            {"text": ck.text, "metadata": dict(ck.metadata)}
            for ck in doc_chunks
        ]

    logger.info("共生成 %d 个分块", len(all_chunks))

    # 保存版本（如果启用）
    if enable_version:
        for doc in docs:
            source_name = doc.metadata.get("source", target.name)
            chunks_for_version = doc_chunks_map.get(source_name, [])
            if chunks_for_version:
                doc_hash = _compute_doc_hash(doc.content)
                version = version_manager.save_version(
                    doc_name=source_name,
                    chunks=chunks_for_version,
                    doc_hash=doc_hash,
                    created_by="ingest",
                )
                logger.info("文档 %s 已保存版本 %s", source_name, version)

    # 加载 embedding & 向量库
    embed = EmbeddingModel(
        model_name=cfg["embedding"]["model_name"],
        device=cfg["embedding"].get("device", "auto"),
        batch_size=cfg["embedding"].get("batch_size", 32),
        max_seq_length=cfg["embedding"].get("max_seq_length", 512),
        normalize=cfg["embedding"].get("normalize_embeddings", True),
        cache_dir=cfg["embedding"].get("cache_dir"),
        local_files_only=cfg["embedding"].get("local_files_only", False),
    )
    store = ChromaStore(
        persist_directory=cfg["vector_store"]["persist_directory"],
        collection_name=cfg["vector_store"].get("collection_name", "chinese_rag_kb"),
        embedding_model=embed,
        distance_fn=cfg["vector_store"].get("distance_fn", "cosine"),
    )

    n = store.add_chunks(all_chunks)
    logger.info("✅ 入库完成，新增/覆盖 %d 条", n)
    return n


def main():
    parser = argparse.ArgumentParser(description="RAG 知识库入库脚本")
    parser.add_argument("target", help="待入库文件或目录路径")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--recursive", action="store_true", help="递归遍历目录")
    parser.add_argument("--version", action="store_true", help="保存文档版本到 data/versions/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_env_overrides(cfg)
    splitter = build_splitter(cfg)

    ensure_dir(cfg["vector_store"]["persist_directory"])
    paths = cfg.get("paths", {}) or {}
    raw_docs_dir = paths.get("raw_docs") or paths.get("raw_docs_dir") or "data/raw"
    ensure_dir(raw_docs_dir)

    target = Path(args.target)
    if not target.exists():
        logger.error("目标路径不存在: %s", target)
        sys.exit(1)

    n = ingest_path(target, cfg, splitter, enable_version=args.version)
    print(f"\n入库完成：{n} 条分块")


if __name__ == "__main__":
    main()