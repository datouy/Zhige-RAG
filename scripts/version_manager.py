"""文档版本管理 CLI 工具。

功能：
- 查看某文档所有版本
- 查看某版本详情
- 对比两个版本的差异（chunk-level diff）
- 列出所有已版本化文档
- 清理旧版本（保留最近 N 个）

用法：
    python scripts/version_manager.py list rag-intro.md
    python scripts/version_manager.py show rag-intro.md v2
    python scripts/version_manager.py diff rag-intro.md v1 v2
    python scripts/version_manager.py list-all
    python scripts/version_manager.py prune rag-intro.md --keep 3
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, get_logger, resolve_path

logger = get_logger("version_manager")

VERSIONS_ROOT = "data/versions"
SIMILARITY_THRESHOLD = 0.6


# ----------------------------------------------------------------------
# 版本存储路径
# ----------------------------------------------------------------------
def _doc_versions_root(doc_name: str) -> Path:
    root = resolve_path(VERSIONS_ROOT)
    safe_name = Path(doc_name).name
    return root / safe_name


def _version_dir(doc_name: str, version: str) -> Path:
    return _doc_versions_root(doc_name) / version


# ----------------------------------------------------------------------
# 版本号管理
# ----------------------------------------------------------------------
def _get_latest_version(doc_name: str) -> str | None:
    """获取某文档最新版本号（如 v10），不存在则返回 None。"""
    root = _doc_versions_root(doc_name)
    if not root.exists():
        return None
    versions = []
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith("v"):
            versions.append(d.name)
    if not versions:
        return None
    versions.sort(key=lambda x: int(x[1:]))
    return versions[-1]


def _next_version(doc_name: str) -> str:
    """获取下一个版本号。首次为 v1。"""
    latest = _get_latest_version(doc_name)
    if latest is None:
        return "v1"
    return f"v{int(latest[1:]) + 1}"


def _parse_version_order(v: str) -> int:
    """将版本号转为整数用于排序。"""
    return int(v[1:]) if v.startswith("v") else 0


# ----------------------------------------------------------------------
# 版本保存
# ----------------------------------------------------------------------
def save_version(
    doc_name: str,
    chunks: list,
    doc_hash: str,
    created_by: str = "ingest",
) -> str:
    """保存文档版本，返回版本号。

    Args:
        doc_name: 文档名（如 rag-intro.md）
        chunks: 分块列表，每项含 text 和 metadata
        doc_hash: 文档原始内容的 SHA256
        created_by: 创建来源（ingest / update）

    Returns:
        版本号字符串（如 "v1", "v2"）
    """
    doc_root = _doc_versions_root(doc_name)
    ensure_dir(doc_root)

    # 检查是否已有相同 hash 的版本
    latest_v = _get_latest_version(doc_name)
    if latest_v:
        meta_path = _version_dir(doc_name, latest_v) / "metadata.json"
        if meta_path.exists():
            prev_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if prev_meta.get("doc_hash") == doc_hash:
                logger.info("文档内容未变（hash=%s），跳过新建版本", doc_hash[:8])
                return latest_v

    # 创建新版本
    version = _next_version(doc_name)
    vdir = _version_dir(doc_name, version)
    ensure_dir(vdir)

    total_chars = sum(len(c.get("text", "")) for c in chunks)

    metadata = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "doc_hash": doc_hash,
        "num_chunks": len(chunks),
        "total_chars": total_chars,
        "created_by": created_by,
        "source_file": doc_name,
    }

    chunks_file = vdir / "chunks.json"
    meta_file = vdir / "metadata.json"

    chunks_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("已保存版本 %s：%s (%d chunks, %d chars)", version, doc_name, len(chunks), total_chars)
    return version


# ----------------------------------------------------------------------
# 版本查询
# ----------------------------------------------------------------------
def list_versions(doc_name: str) -> list[dict]:
    """列出某文档所有版本，返回版本信息列表。"""
    root = _doc_versions_root(doc_name)
    if not root.exists():
        return []

    results = []
    for vdir in sorted(root.iterdir(), key=lambda x: _parse_version_order(x.name)):
        if not vdir.is_dir() or not vdir.name.startswith("v"):
            continue
        meta_file = vdir / "metadata.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            results.append(meta)
    return results


def list_all_documents() -> list[str]:
    """列出所有已版本化的文档名。"""
    root = resolve_path(VERSIONS_ROOT)
    if not root.exists():
        return []
    return [d.name for d in root.iterdir() if d.is_dir()]


def get_version_detail(doc_name: str, version: str) -> dict | None:
    """获取某版本的详细信息。"""
    vdir = _version_dir(doc_name, version)
    meta_file = vdir / "metadata.json"
    chunks_file = vdir / "chunks.json"

    if not meta_file.exists() or not chunks_file.exists():
        return None

    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))

    return {
        "metadata": metadata,
        "chunks": chunks,
    }


# ----------------------------------------------------------------------
# Chunk-Level Diff
# ----------------------------------------------------------------------
def _compute_chunks_diff(old_chunks: list, new_chunks: list) -> dict:
    """计算两个版本之间的 chunk 级别差异。

    使用 SequenceMatcher 按文本相似度匹配：
    - ratio >= SIMILARITY_THRESHOLD → modified
    - ratio < SIMILARITY_THRESHOLD 且 old 有匹配 → removed
    - ratio < SIMILARITY_THRESHOLD 且 new 有匹配 → added
    """
    old_texts = [c.get("text", "") for c in old_chunks]
    new_texts = [c.get("text", "") for c in new_chunks]

    matcher = difflib.SequenceMatcher(None, old_texts, new_texts)
    opcodes = matcher.get_opcodes()

    changes = []
    added = 0
    removed = 0
    modified = 0
    unchanged = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            unchanged += (i2 - i1)
        elif tag == "replace":
            for idx in range(i1, i2):
                old_text = old_texts[idx]
                new_text = new_texts[j1 + (idx - i1)] if j1 + (idx - i1) < j2 else ""
                ratio = difflib.SequenceMatcher(None, old_text, new_text).ratio()
                changes.append({
                    "type": "modified",
                    "chunk_id": idx,
                    "from_text": old_text,
                    "to_text": new_text,
                    "similarity": round(ratio, 3),
                })
                modified += 1
        elif tag == "delete":
            for idx in range(i1, i2):
                changes.append({
                    "type": "removed",
                    "chunk_id": idx,
                    "from_text": old_texts[idx],
                })
                removed += 1
        elif tag == "insert":
            for idx in range(j1, j2):
                changes.append({
                    "type": "added",
                    "chunk_id": None,
                    "to_text": new_texts[idx],
                })
                added += 1

    return {
        "summary": {
            "added": added,
            "removed": removed,
            "modified": modified,
            "unchanged": unchanged,
        },
        "changes": changes,
    }


def compute_diff(doc_name: str, from_version: str, to_version: str) -> dict | None:
    """对比两个版本，返回 diff 结果。"""
    old_data = get_version_detail(doc_name, from_version)
    new_data = get_version_detail(doc_name, to_version)

    if old_data is None or new_data is None:
        return None

    diff_result = _compute_chunks_diff(old_data["chunks"], new_data["chunks"])

    return {
        "doc": doc_name,
        "from_version": from_version,
        "to_version": to_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **diff_result,
    }


# ----------------------------------------------------------------------
# 版本清理
# ----------------------------------------------------------------------
def prune_versions(doc_name: str, keep: int = 3) -> int:
    """删除旧版本，只保留最近的 N 个版本。

    Returns:
        删除的版本数量。
    """
    versions = list_versions(doc_name)
    if len(versions) <= keep:
        logger.info("版本数量 %d <= keep=%d，无需清理", len(versions), keep)
        return 0

    versions.sort(key=lambda x: _parse_version_order(x["version"]))
    to_delete = versions[:-keep]
    deleted = 0

    for v_meta in to_delete:
        vdir = _version_dir(doc_name, v_meta["version"])
        if vdir.exists():
            shutil.rmtree(vdir)
            logger.info("已删除旧版本 %s", v_meta["version"])
            deleted += 1

    return deleted


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------
def _cli_list(doc_name: str):
    """列出文档所有版本。"""
    versions = list_versions(doc_name)
    if not versions:
        print(f"未找到文档 {doc_name} 的版本记录。")
        return

    print(f"\n文档: {doc_name}")
    print(f"{'版本':<8} {'时间':<28} {'分块数':<8} {'字符数':<10} {'来源':<10}")
    print("-" * 70)
    for v in versions:
        ts = v.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        print(f"{v['version']:<8} {ts:<28} {v['num_chunks']:<8} {v['total_chars']:<10} {v.get('created_by', ''):<10}")


def _cli_show(doc_name: str, version: str):
    """显示版本详情。"""
    data = get_version_detail(doc_name, version)
    if data is None:
        print(f"未找到 {doc_name} 的版本 {version}。")
        sys.exit(1)

    meta = data["metadata"]
    chunks = data["chunks"]

    print(f"\n{'='*60}")
    print(f"文档: {doc_name}")
    print(f"版本: {meta['version']}")
    print(f"时间: {meta.get('timestamp', 'N/A')}")
    print(f"Hash: {meta.get('doc_hash', 'N/A')}")
    print(f"分块数: {meta['num_chunks']}")
    print(f"字符数: {meta['total_chars']}")
    print(f"来源: {meta.get('created_by', 'N/A')}")
    print(f"{'='*60}")
    print(f"\n分块预览 (前 5 个):")
    for i, ck in enumerate(chunks[:5]):
        text = ck.get("text", "")[:100]
        print(f"  [{i}] {text}...")


def _cli_diff(doc_name: str, from_ver: str, to_ver: str):
    """显示版本差异。"""
    result = compute_diff(doc_name, from_ver, to_ver)
    if result is None:
        print(f"无法比较 {doc_name} 的版本 {from_ver} 和 {to_ver}。")
        sys.exit(1)

    summary = result["summary"]
    print(f"\n{'='*60}")
    print(f"文档: {result['doc']}")
    print(f"对比: {result['from_version']} → {result['to_version']}")
    print(f"{'='*60}")
    print(f"统计: 新增 {summary['added']} | 删除 {summary['removed']} | 修改 {summary['modified']} | 不变 {summary['unchanged']}")
    print(f"{'='*60}")

    if result["changes"]:
        print("\n详细变更:")
        for i, change in enumerate(result["changes"], 1):
            print(f"\n--- 变更 {i} ({change['type']}) ---")
            if change["type"] == "modified":
                print(f"Chunk ID: {change['chunk_id']}")
                print(f"相似度: {change['similarity']}")
                print(f"旧内容: {change['from_text'][:150]}...")
                print(f"新内容: {change['to_text'][:150]}...")
            elif change["type"] == "added":
                print(f"新内容: {change['to_text'][:150]}...")
            elif change["type"] == "removed":
                print(f"Chunk ID: {change['chunk_id']}")
                print(f"旧内容: {change['from_text'][:150]}...")

    # 输出 JSON 格式
    print(f"\n--- JSON Output ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cli_list_all():
    """列出所有已版本化文档。"""
    docs = list_all_documents()
    if not docs:
        print("尚无任何已版本化的文档。")
        return

    print(f"\n已版本化文档列表 ({len(docs)} 个):")
    print(f"{'文档名':<40} {'版本数':<8} {'最新版本':<12}")
    print("-" * 65)
    for doc_name in sorted(docs):
        versions = list_versions(doc_name)
        version_count = len(versions)
        latest = versions[-1]["version"] if versions else "N/A"
        print(f"{doc_name:<40} {version_count:<8} {latest:<12}")


def _cli_prune(doc_name: str, keep: int):
    """清理旧版本。"""
    deleted = prune_versions(doc_name, keep)
    print(f"已删除 {deleted} 个旧版本，保留最近 {keep} 个。")


def main():
    parser = argparse.ArgumentParser(description="文档版本管理工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # list
    p_list = subparsers.add_parser("list", help="列出某文档的所有版本")
    p_list.add_argument("doc_name", help="文档名（如 rag-intro.md）")

    # show
    p_show = subparsers.add_parser("show", help="显示某版本的详情")
    p_show.add_argument("doc_name", help="文档名")
    p_show.add_argument("version", help="版本号（如 v1）")

    # diff
    p_diff = subparsers.add_parser("diff", help="对比两个版本的差异")
    p_diff.add_argument("doc_name", help="文档名")
    p_diff.add_argument("from_version", help="起始版本")
    p_diff.add_argument("to_version", help="目标版本")

    # list-all
    subparsers.add_parser("list-all", help="列出所有已版本化文档")

    # prune
    p_prune = subparsers.add_parser("prune", help="清理旧版本")
    p_prune.add_argument("doc_name", help="文档名")
    p_prune.add_argument("--keep", type=int, default=3, help="保留最近 N 个版本（默认 3）")

    args = parser.parse_args()

    if args.command == "list":
        _cli_list(args.doc_name)
    elif args.command == "show":
        _cli_show(args.doc_name, args.version)
    elif args.command == "diff":
        _cli_diff(args.doc_name, args.from_version, args.to_version)
    elif args.command == "list-all":
        _cli_list_all()
    elif args.command == "prune":
        _cli_prune(args.doc_name, args.keep)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
