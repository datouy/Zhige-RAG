"""Copy and verify a local Chroma workspace under its cross-process write lock."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _manifest(path: Path) -> dict:
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    collections = []
    for collection in client.list_collections():
        collections.append(
            {
                "name": collection.name,
                "count": int(collection.count()),
                "metadata": dict(collection.metadata or {}),
            }
        )
    collections.sort(key=lambda item: item["name"])
    return {"collections": collections, "chunks": sum(item["count"] for item in collections)}


def main() -> int:
    parser = argparse.ArgumentParser(description="迁移本地 Chroma 工作区")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    manifest_path = Path(args.manifest).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("目标目录存在且非空，拒绝覆盖")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(source / ".chroma-write.lock"), timeout=120)
    with lock:
        if destination.exists():
            destination.rmdir()
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(".chroma-write.lock"),
        )

    source_manifest = _manifest(source)
    destination_manifest = _manifest(destination)
    if source_manifest != destination_manifest:
        raise RuntimeError("Chroma collection/chunk 清单不一致")
    digest = hashlib.sha256(
        json.dumps(destination_manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    report = {
        "status": "verified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination": str(destination),
        "collections": destination_manifest["collections"],
        "chunks": destination_manifest["chunks"],
        "manifest_sha256": digest,
        "rollback": "Keep VECTOR_STORE_PERSIST_DIR at the untouched source; switch back on any collection/count probe failure.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
