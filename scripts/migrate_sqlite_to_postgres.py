"""Copy relational application data from SQLite to PostgreSQL.

The script takes a consistent SQLite snapshot first, refuses a non-empty
target unless explicitly allowed, runs all inserts in one target transaction,
and verifies row counts before writing a migration manifest. Chroma vectors
remain in their versioned local directory; this script does not pretend to
migrate them into pgvector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import Boolean, DateTime, Float, Integer, MetaData, create_engine, func, inspect, select

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.models import Base  # noqa: E402


def _convert(value: Any, column: Any) -> Any:
    if value is None:
        return None
    if isinstance(column.type, DateTime) and isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    if isinstance(column.type, Boolean):
        return bool(value)
    if isinstance(column.type, Integer):
        return int(value)
    if isinstance(column.type, Float):
        return float(value)
    return value


def _snapshot(source: Path, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source), timeout=30) as src:
        with sqlite3.connect(output) as dst:
            src.backup(dst)
            dst.execute("PRAGMA wal_checkpoint(FULL)")
    digest = hashlib.sha256()
    with open(output, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def migrate(
    *,
    sqlite_path: Path,
    database_url: str,
    snapshot_dir: Path,
    allow_nonempty: bool = False,
) -> Dict[str, Any]:
    if not sqlite_path.is_file():
        raise FileNotFoundError(sqlite_path)
    if not database_url.startswith("postgresql"):
        raise ValueError("目标 DATABASE_URL 必须是 PostgreSQL")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = snapshot_dir / f"sqlite-before-postgres-{stamp}.db"
    snapshot_sha256 = _snapshot(sqlite_path, snapshot)

    source_engine = create_engine(f"sqlite:///{sqlite_path}")
    target_engine = create_engine(database_url, pool_pre_ping=True)
    source_meta = MetaData()
    source_meta.reflect(bind=source_engine)
    Base.metadata.create_all(bind=target_engine)
    inspector = inspect(target_engine)
    existing_tables = set(inspector.get_table_names())
    counts: Dict[str, int] = {}

    with target_engine.begin() as target:
        for table in Base.metadata.sorted_tables:
            if table.name not in source_meta.tables:
                continue
            source_table = source_meta.tables[table.name]
            existing = int(
                target.execute(select(func.count()).select_from(table)).scalar_one()
            )
            if existing and not allow_nonempty:
                raise RuntimeError(
                    f"目标表 {table.name} 已有 {existing} 行；迁移中止，未写入任何后续表"
                )
            source_columns = {col.name for col in source_table.columns}
            target_columns = [col for col in table.columns if col.name in source_columns]
            batch = []
            source_count = 0
            with source_engine.connect() as source_connection:
                rows = source_connection.execute(select(source_table)).mappings()
                for row in rows:
                    batch.append(
                        {col.name: _convert(row[col.name], col) for col in target_columns}
                    )
                    source_count += 1
                    if len(batch) >= 500:
                        target.execute(table.insert(), batch)
                        batch.clear()
            if batch:
                target.execute(table.insert(), batch)
            target_count = int(
                target.execute(select(func.count()).select_from(table)).scalar_one()
            )
            if target_count != source_count:
                raise RuntimeError(
                    f"表 {table.name} 搬运行数不一致: source={source_count}, target={target_count}"
                )
            counts[table.name] = target_count

    source_engine.dispose()
    target_engine.dispose()
    report = {
        "status": "migrated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rows": counts,
        "source_snapshot": str(snapshot),
        "source_snapshot_sha256": snapshot_sha256,
        "target_table_count": len(existing_tables),
        "vector_backend": "chroma-unchanged",
        "rollback": "Keep existing DB_PATH and restart with DATABASE_URL unset; the pre-migration snapshot is also restorable.",
    }
    manifest = snapshot_dir / f"migration-{stamp}.json"
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["manifest"] = str(manifest)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="SQLite → PostgreSQL 关系数据迁移")
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--allow-nonempty", action="store_true")
    args = parser.parse_args()
    report = migrate(
        sqlite_path=Path(args.sqlite).resolve(),
        database_url=args.database_url,
        snapshot_dir=Path(args.snapshot_dir).resolve(),
        allow_nonempty=args.allow_nonempty,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
