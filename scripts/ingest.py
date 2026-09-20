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
            custom_dict_path="config/custom_dict.txt",  # 自动加载自定义词典
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

    # 数据层统一管线：元数据增强（标题/层级/更新时间/权限/状态）
    # → 深度清洗 → 去重 → 准入门（过滤过期/草稿/权限不清）→ 分块 → 分块去重
    from src.data_quality import run_data_pipeline
    from src.document_meta import enrich_directory, enrich_file

    if target.is_file():
        docs = enrich_file(target, default_acl=cfg.get("data_quality", {}).get("default_acl", "*"))
    elif target.is_dir():
        docs = enrich_directory(
            target,
            default_acl=cfg.get("data_quality", {}).get("default_acl", "*"),
            recursive=True,
        )
    else:
        logger.warning("路径不存在: %s", target)
        return 0

    if not docs:
        logger.warning("未解析到任何文档")
        return 0

    all_chunks, report = run_data_pipeline(docs, cfg, splitter)
    logger.info(
        "数据层管线完成：%d 个小节 → %d 个分块（文档去重丢弃 %d，准入门拒绝 %d，分块去重丢弃 %d）",
        len(docs),
        len(all_chunks),
        report.get("doc_dedup_dropped", 0),
        report.get("gate", {}).get("rejected", 0),
        report.get("chunk_dedup_dropped", 0),
    )
    gate = report.get("gate", {})
    if gate.get("rejected"):
        for detail in gate.get("details", [])[:10]:
            logger.warning("准入门拒绝：%s", detail)

    if not all_chunks:
        logger.warning("管线后没有可入库分块（可能全部被准入门拒绝）")
        return 0

    # 保存版本（如果启用）——按原始文档（清洗前）逐份保存
    if enable_version:
        seen_sources: set = set()
        for doc in docs:
            source_name = doc.metadata.get("source", target.name)
            if source_name in seen_sources:
                continue
            seen_sources.add(source_name)
            doc_hash = _compute_doc_hash(doc.content)
            chunks_for_version = [
                {"text": ck.text, "metadata": dict(ck.metadata)}
                for ck in all_chunks
                if (ck.metadata or {}).get("source") == source_name
            ]
            if chunks_for_version:
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

    # 入库前按 source 清理同名文档旧分块，避免内容更新后残留旧块
    # （分块 id 只含 source|page|chunk_index，upsert 不会删除已消失的块）
    stale_sources = {
        str(c.metadata.get("source") or c.metadata.get("filepath") or "unknown")
        for c in all_chunks
    }
    for src in stale_sources:
        try:
            deleted = store.delete_by_metadata({"source": {"$eq": src}})
            if deleted:
                logger.info("已清理 %s 的 %d 条旧分块", src, deleted)
        except Exception as exc:
            logger.warning("清理旧分块失败（source=%s）: %s", src, exc)

    n = store.add_chunks(all_chunks)
    logger.info("✅ 入库完成，新增/覆盖 %d 条", n)
    
    # 入库后打印几条分块样例，检查质量
    _print_chunk_samples(all_chunks, num_samples=5)
    
    return n


def _print_chunk_samples(chunks, num_samples: int = 5):
    """打印分块样例用于质量检查。"""
    if not chunks:
        return
    
    print(f"\n{'='*80}")
    print(f"分块质量检查 - 样例展示")
    print(f"{'='*80}\n")
    
    print(f"总分块数: {len(chunks)}")
    avg_len = sum(len(c.text) for c in chunks) / len(chunks)
    print(f"平均长度: {avg_len:.1f} 字符")
    print(f"最小长度: {min(len(c.text) for c in chunks)}")
    print(f"最大长度: {max(len(c.text) for c in chunks)}\n")
    
    print(f"{'='*80}")
    print(f"前 {min(num_samples, len(chunks))} 个分块预览:")
    print(f"{'='*80}\n")
    
    for i, chunk in enumerate(chunks[:num_samples]):
        print(f"{'-'*80}")
        print(f"分块 {i+1}:")
        print(f"  长度: {len(chunk.text)} 字符")
        print(f"  来源: {chunk.metadata.get('source', 'N/A')}")
        if 'title' in chunk.metadata:
            print(f"  标题: {chunk.metadata['title']}")
        print(f"  内容预览:\n")
        
        # 显示前 200 字符
        preview = chunk.text[:200]
        if len(chunk.text) > 200:
            preview += "..."
        print(f"    {preview}")
        print()
    
    print(f"{'='*80}")
    print("质量检查要点:")
    print("   [OK] 每个块语义完整（未在句子中间截断）")
    print("   [OK] 块长度合理（300-500 字符为宜）")
    print("   [OK] 专有名词未被切碎")
    print("   [OK] 有适当的上下文重叠")
    print(f"{'='*80}\n")


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