"""完整备份包校验与 SQLite/原始文档恢复演练。"""
from __future__ import annotations

import sqlite3
import tarfile
from pathlib import Path

from scripts.backup_restore import create_backup, restore_backup


def _make_db(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS sample (value TEXT NOT NULL)")
        conn.execute("DELETE FROM sample")
        conn.execute("INSERT INTO sample(value) VALUES (?)", (value,))
        conn.commit()
    finally:
        conn.close()


def _db_value(path: Path) -> str:
    conn = sqlite3.connect(path)
    try:
        return str(conn.execute("SELECT value FROM sample").fetchone()[0])
    finally:
        conn.close()


def test_full_local_backup_and_restore_drill(tmp_path):
    root = tmp_path / "project"
    db = root / "data" / "users.db"
    raw = root / "data" / "raw"
    kg = root / "data" / "kg-test.db"
    chroma = root / "data" / "chroma"
    raw.mkdir(parents=True)
    (raw / "policy.md").write_text("备份前内容", encoding="utf-8")
    chroma.mkdir(parents=True)
    (chroma / "index.bin").write_bytes(b"index-v1")
    _make_db(db, "before")
    _make_db(kg, "graph-before")

    archive = tmp_path / "backup.tar.gz"
    result = create_backup(
        archive,
        root=root,
        backends={
            "sqlite_db": str(db),
            "raw_dir": str(raw),
            "chroma_dir": str(chroma),
            "vector": "chroma",
            "kg": "sqlite",
        },
        allow_live=True,
    )
    assert result["bytes"] > 0
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert "manifest.json" in names
    assert "checksums.sha256" in names
    assert "sqlite/users.db" in names
    assert "raw/policy.md" in names
    assert "vector/chroma/index.bin" in names

    _make_db(db, "after")
    (raw / "policy.md").write_text("备份后内容", encoding="utf-8")
    (chroma / "index.bin").write_bytes(b"index-v2")

    restored = restore_backup(
        archive,
        root=root,
        force=True,
        replace_data=True,
        sqlite_db=db,
        raw_dir=raw,
        chroma_dir=chroma,
        kg_data_dir=root / "data",
    )
    assert restored["status"] == "restored"
    assert _db_value(db) == "before"
    assert (raw / "policy.md").read_text(encoding="utf-8") == "备份前内容"
    assert (chroma / "index.bin").read_bytes() == b"index-v1"
    assert db.with_suffix(".db.pre-restore").exists()


def test_restore_requires_explicit_force(tmp_path):
    import pytest

    from scripts.backup_restore import BackupError

    root = tmp_path / "project"
    db = root / "data" / "users.db"
    _make_db(db, "value")
    archive = tmp_path / "backup.tar.gz"
    create_backup(
        archive,
        root=root,
        backends={
            "sqlite_db": str(db),
            "raw_dir": str(root / "data" / "raw"),
            "vector": "chroma",
            "kg": "sqlite",
        },
        allow_live=True,
    )
    with pytest.raises(BackupError, match="--force"):
        restore_backup(archive, root=root)
