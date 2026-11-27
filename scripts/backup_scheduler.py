"""Scheduled, verified and optionally offsite workspace backups."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backup_service import (  # noqa: E402
    BACKUP_ROOT,
    create_backup,
    verify_backup,
)
from src.utils import get_logger, resolve_path  # noqa: E402

logger = get_logger("backup_scheduler")


def _run_once(remote_dir: Path | None, retention_days: int) -> dict:
    result = create_backup()
    local = Path(result["path"])
    manifest = verify_backup(local)
    copied = None
    if remote_dir is not None:
        remote_dir.mkdir(parents=True, exist_ok=True)
        copied_path = remote_dir / local.name
        temporary = copied_path.with_name(copied_path.name + ".partial")
        digest_before = result["sha256"]
        shutil.copy2(local, temporary)
        os.replace(temporary, copied_path)
        verify_backup(copied_path)
        digest_hash = hashlib.sha256()
        with open(copied_path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest_hash.update(block)
        digest = digest_hash.hexdigest()
        if digest != digest_before:
            copied_path.unlink(missing_ok=True)
            raise RuntimeError("异地备份 SHA256 与本地不一致")
        copied = str(copied_path)

    cutoff = time.time() - max(0, retention_days) * 86400
    if retention_days > 0:
        for directory in (BACKUP_ROOT, remote_dir):
            if directory is None or not directory.exists():
                continue
            for path in directory.glob("zhige-backup-*"):
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)

    report = {
        **result,
        "verified": True,
        "manifest_created_at": manifest.get("created_at"),
        "offsite_path": copied,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    status_file = BACKUP_ROOT / "backup-status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="知阁自动备份")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=86400)
    parser.add_argument("--retention-days", type=int, default=14)
    parser.add_argument("--remote-dir", default=os.getenv("ZHIGE_BACKUP_REMOTE_DIR", ""))
    args = parser.parse_args()
    remote_dir = resolve_path(args.remote_dir) if args.remote_dir.strip() else None
    data_root = resolve_path(os.getenv("RAG_DATA_DIR", "data"))
    lock = FileLock(str(data_root / ".backup-scheduler.lock"), timeout=0)
    while True:
        try:
            with lock:
                result = _run_once(remote_dir, args.retention_days)
                print(json.dumps(result, ensure_ascii=False))
        except Exception:  # noqa: BLE001
            logger.exception("自动备份失败")
            if args.once:
                raise
        if args.once:
            return 0
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
