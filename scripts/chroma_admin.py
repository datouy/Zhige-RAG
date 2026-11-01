"""Chroma 持久化索引的检查、备份与故障隔离工具。

示例（执行写操作前请先停止 API 进程）：

    python scripts/chroma_admin.py health
    python scripts/chroma_admin.py backup --output data/backups/chroma.zip
    python scripts/chroma_admin.py export --collection chinese_rag_kb --output data/export.jsonl
    python scripts/chroma_admin.py drop-collection --collection broken_name --confirm
    python scripts/chroma_admin.py quarantine --confirm
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.deps import get_runtime_config  # noqa: E402
from src.utils import resolve_path  # noqa: E402
from src.vector_store import get_shared_chroma_client, probe_collections  # noqa: E402


def _persist_dir() -> Path:
    cfg = get_runtime_config()
    return resolve_path(cfg["vector_store"]["persist_directory"])


def _client():
    path = _persist_dir()
    return get_shared_chroma_client(str(path))


def _iter_collection_rows(collection: Any, batch_size: int = 500) -> Iterator[Dict[str, Any]]:
    offset = 0
    while True:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        if not ids:
            break
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for index, chunk_id in enumerate(ids):
            yield {
                "id": chunk_id,
                "document": documents[index] if index < len(documents) else "",
                "metadata": metadatas[index] if index < len(metadatas) else {},
            }
        offset += len(ids)
        if len(ids) < batch_size:
            break


def cmd_health(_args: argparse.Namespace) -> int:
    report = probe_collections(_client())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "healthy" else 1


def cmd_backup(args: argparse.Namespace) -> int:
    source = _persist_dir()
    if not source.exists():
        print(f"Chroma 目录不存在: {source}", file=sys.stderr)
        return 1
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_base = output.with_suffix("") if output.suffix == ".zip" else output
    created = shutil.make_archive(str(archive_base), "zip", root_dir=str(source.parent), base_dir=source.name)
    print(created)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    client = _client()
    try:
        collection = client.get_collection(args.collection)
    except Exception as exc:
        print(f"无法打开 collection {args.collection}: {exc}", file=sys.stderr)
        return 1
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in _iter_collection_rows(collection, batch_size=args.batch_size):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    print(json.dumps({"collection": args.collection, "rows": rows, "output": str(output)}, ensure_ascii=False))
    return 0


def cmd_drop_collection(args: argparse.Namespace) -> int:
    if not args.confirm:
        print("拒绝执行：删除 collection 需要显式传入 --confirm", file=sys.stderr)
        return 2
    client = _client()
    try:
        client.delete_collection(args.collection)
    except Exception as exc:
        print(f"删除 collection 失败: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"dropped": args.collection}, ensure_ascii=False))
    return 0


def cmd_quarantine(args: argparse.Namespace) -> int:
    if not args.confirm:
        print("拒绝执行：隔离整个目录需要显式传入 --confirm", file=sys.stderr)
        return 2
    source = _persist_dir()
    if not source.exists():
        print(f"Chroma 目录不存在: {source}", file=sys.stderr)
        return 1
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = source.with_name(f"{source.name}.quarantine-{stamp}")
    source.rename(target)
    if args.create_empty:
        source.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"quarantined": str(source), "moved_to": str(target)}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chroma 索引检查与恢复工具")
    sub = parser.add_subparsers(dest="command", required=True)

    health = sub.add_parser("health", help="只读检查所有 collection")
    health.set_defaults(func=cmd_health)

    backup = sub.add_parser("backup", help="将整个 Chroma 持久化目录打包为 zip")
    backup.add_argument("--output", required=True, help="输出 zip 路径")
    backup.set_defaults(func=cmd_backup)

    export = sub.add_parser("export", help="导出单个 collection 为 JSONL")
    export.add_argument("--collection", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--batch-size", type=int, default=500)
    export.set_defaults(func=cmd_export)

    drop = sub.add_parser("drop-collection", help="删除无法读取的 collection")
    drop.add_argument("--collection", required=True)
    drop.add_argument("--confirm", action="store_true")
    drop.set_defaults(func=cmd_drop_collection)

    quarantine = sub.add_parser("quarantine", help="隔离整个持久化目录并可选创建空目录")
    quarantine.add_argument("--confirm", action="store_true")
    quarantine.add_argument("--create-empty", action="store_true")
    quarantine.set_defaults(func=cmd_quarantine)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
