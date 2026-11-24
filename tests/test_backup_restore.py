"""备份/恢复演练：创建备份、暂存恢复、完整性校验与取消。"""
from pathlib import Path

import src.backup_service as bs
from src.db.database import SessionLocal, init_db
from src.db.models import User


def test_create_stage_and_cancel_backup(monkeypatch, tmp_path):
    init_db()
    db = SessionLocal()
    try:
        db.add(User(username="drill", email="drill@x.com", is_active=True, role="viewer"))
        db.commit()
    finally:
        db.close()

    monkeypatch.setattr(bs, "BACKUP_ROOT", tmp_path / "backups")
    monkeypatch.setattr(bs, "RESTORE_ROOT", tmp_path / "restore_staging")
    monkeypatch.setattr(bs, "PENDING_FILE", tmp_path / ".restore_pending.json")
    monkeypatch.setattr(bs, "STATUS_FILE", tmp_path / ".restore_status.json")
    monkeypatch.setattr(bs, "_runtime_paths", lambda: {
        "raw": tmp_path / "raw",
        "chroma": tmp_path / "chroma",
        "eval": tmp_path / "eval",
        "logs": tmp_path / "logs",
    })

    result = bs.create_backup()
    assert Path(result["path"]).exists()
    assert result["file_count"] >= 1
    assert len(result["sha256"]) == 64

    pending = bs.stage_restore(Path(result["path"]), "drill.zip")
    assert pending["manifest"]["app"] == "zhige"
    assert bs.pending_restore() is not None

    bs.cancel_pending_restore()
    assert bs.pending_restore() is None
