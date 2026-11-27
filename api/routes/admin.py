"""管理员概览、质量治理、审计、评估与工作区备份。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.backup_service import (
    BACKUP_ROOT,
    BackupError,
    cancel_pending_restore,
    create_backup,
    last_restore_status,
    pending_restore,
    stage_restore,
)
from src.db.database import check_connection, get_db
from src.db.models import (
    DocumentRecord,
    FeedbackRecord,
    KnowledgeBase,
    KnowledgeGap,
    MCPServer,
    UsageLog,
    User,
)
from src.middleware.audit import AuditAction, get_audit_logger
from src.team_auth import require_admin_user
from src.utils import get_logger, resolve_path

logger = get_logger("api.routes.admin")
router = APIRouter()


class FeedbackResolveBody(BaseModel):
    resolved: bool = True
    resolution: str = Field("", max_length=2000)
    create_gap: bool = False


class KnowledgeGapUpdateBody(BaseModel):
    status: str = Field(..., pattern="^(open|in_progress|resolved|dismissed)$")
    resolution: str = Field("", max_length=4000)
    assigned_to: Optional[str] = Field(None, max_length=36)


def _json_load(raw: Optional[str], fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _feedback_dict(row: FeedbackRecord) -> Dict[str, Any]:
    return {
        "id": row.id,
        "user_id": row.user_id,
        "query": row.query,
        "answer": row.answer or "",
        "rating": row.rating,
        "correction": row.correction or "",
        "question_type": row.question_type or "",
        "citation_issue": bool(row.citation_issue),
        "missing_source": bool(row.missing_source),
        "resolved": bool(row.resolved),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _gap_dict(row: KnowledgeGap) -> Dict[str, Any]:
    return {
        "id": row.id,
        "user_id": row.user_id,
        "kb_id": row.kb_id,
        "session_id": row.session_id,
        "feedback_id": row.feedback_id,
        "query": row.query,
        "answer": row.answer or "",
        "reason": row.reason,
        "detail": row.detail or "",
        "status": row.status,
        "resolution": row.resolution or "",
        "assigned_to": row.assigned_to,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


@router.get("/api/v1/admin/overview", tags=["管理"])
def admin_overview(
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    from src.ingest_jobs import stats as job_stats

    unresolved_feedback = (
        db.query(func.count(FeedbackRecord.id))
        .filter(FeedbackRecord.rating == "not_helpful", FeedbackRecord.resolved.is_(False))
        .scalar()
        or 0
    )
    open_gaps = (
        db.query(func.count(KnowledgeGap.id))
        .filter(KnowledgeGap.status.in_(["open", "in_progress"]))
        .scalar()
        or 0
    )
    failed_docs = (
        db.query(func.count(DocumentRecord.id))
        .filter(DocumentRecord.index_status == "failed")
        .scalar()
        or 0
    )
    pending_mcp = (
        db.query(func.count(MCPServer.id))
        .filter(MCPServer.approval_status == "pending")
        .scalar()
        or 0
    )
    return {
        "users": {
            "total": int(db.query(func.count(User.id)).scalar() or 0),
            "active": int(db.query(func.count(User.id)).filter(User.is_active.is_(True)).scalar() or 0),
        },
        "knowledge": {
            "kbs": int(db.query(func.count(KnowledgeBase.id)).scalar() or 0),
            "documents": int(db.query(func.count(DocumentRecord.id)).scalar() or 0),
            "chunks": int(db.query(func.coalesce(func.sum(DocumentRecord.chunk_count), 0)).scalar() or 0),
            "failed_documents": int(failed_docs),
        },
        "quality": {
            "unresolved_feedback": int(unresolved_feedback),
            "open_gaps": int(open_gaps),
        },
        "security": {
            "pending_mcp_approvals": int(pending_mcp),
            "restore_pending": bool(pending_restore()),
        },
        "ingest": job_stats(),
        "settings": {
            "authenticated_as": current_user.username,
            "database_ok": bool(check_connection()),
        },
    }


@router.get("/api/v1/admin/quality", tags=["管理"])
def admin_quality(
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    feedback = (
        db.query(FeedbackRecord)
        .filter(FeedbackRecord.rating == "not_helpful")
        .order_by(FeedbackRecord.resolved.asc(), FeedbackRecord.created_at.desc())
        .limit(limit)
        .all()
    )
    gaps = (
        db.query(KnowledgeGap)
        .filter(KnowledgeGap.status.in_(["open", "in_progress"]))
        .order_by(KnowledgeGap.created_at.desc())
        .limit(limit)
        .all()
    )
    total_feedback = int(db.query(func.count(FeedbackRecord.id)).scalar() or 0)
    bad_feedback = int(
        db.query(func.count(FeedbackRecord.id))
        .filter(FeedbackRecord.rating == "not_helpful")
        .scalar()
        or 0
    )
    return {
        "feedback": [_feedback_dict(row) for row in feedback],
        "knowledge_gaps": [_gap_dict(row) for row in gaps],
        "stats": {
            "feedback_total": total_feedback,
            "not_helpful": bad_feedback,
            "bad_rate": round(bad_feedback / total_feedback, 4) if total_feedback else 0.0,
            "unresolved_feedback": sum(1 for row in feedback if not row.resolved),
            "open_gaps": len(gaps),
        },
    }


@router.patch("/api/v1/feedback/{feedback_id}/resolve", tags=["管理"])
def resolve_feedback(
    feedback_id: str,
    body: FeedbackResolveBody,
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    row = db.query(FeedbackRecord).filter(FeedbackRecord.id == feedback_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="反馈不存在")
    row.resolved = body.resolved
    gap = None
    if body.create_gap:
        gap = (
            db.query(KnowledgeGap)
            .filter(KnowledgeGap.feedback_id == row.id)
            .first()
        )
        if gap is None:
            gap = KnowledgeGap(
                user_id=row.user_id,
                feedback_id=row.id,
                query=row.query,
                answer=row.answer,
                reason="feedback",
                detail=body.resolution or "管理员根据差评创建",
                status="open",
            )
            db.add(gap)
    db.commit()
    if gap is not None:
        db.refresh(gap)
    try:
        get_audit_logger().log(
            AuditAction.FEEDBACK_RESOLVE,
            user_id=current_user.id,
            resource=row.id,
            details={"resolved": bool(row.resolved), "gap_id": gap.id if gap else None},
            db=db,
        )
    except Exception:
        logger.debug("反馈处理审计写入失败", exc_info=True)
    return {
        "feedback": _feedback_dict(row),
        "knowledge_gap": _gap_dict(gap) if gap is not None else None,
    }


@router.get("/api/v1/knowledge-gaps", tags=["管理"])
def list_knowledge_gaps(
    status: Optional[str] = Query(None, pattern="^(open|in_progress|resolved|dismissed)$"),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    query = db.query(KnowledgeGap)
    if status:
        query = query.filter(KnowledgeGap.status == status)
    rows = query.order_by(KnowledgeGap.created_at.desc()).limit(limit).all()
    return {"items": [_gap_dict(row) for row in rows], "total": len(rows)}


@router.patch("/api/v1/knowledge-gaps/{gap_id}", tags=["管理"])
def update_knowledge_gap(
    gap_id: str,
    body: KnowledgeGapUpdateBody,
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    row = db.query(KnowledgeGap).filter(KnowledgeGap.id == gap_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="知识缺口不存在")
    if body.assigned_to:
        assignee = db.query(User).filter(User.id == body.assigned_to, User.is_active.is_(True)).first()
        if assignee is None:
            raise HTTPException(status_code=404, detail="指派用户不存在或已停用")
    row.status = body.status
    row.resolution = body.resolution.strip() or None
    row.assigned_to = body.assigned_to
    db.commit()
    db.refresh(row)
    try:
        get_audit_logger().log(
            AuditAction.KNOWLEDGE_GAP_UPDATE,
            user_id=current_user.id,
            resource=row.id,
            details={"status": row.status, "assigned_to": row.assigned_to},
            db=db,
        )
    except Exception:
        logger.debug("知识缺口审计写入失败", exc_info=True)
    return _gap_dict(row)


@router.get("/api/v1/admin/audit", tags=["管理"])
def admin_audit(
    limit: int = Query(100, ge=1, le=500),
    action: Optional[str] = Query(None, max_length=80),
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    query = db.query(UsageLog).filter(UsageLog.method == "AUDIT")
    if action:
        query = query.filter(UsageLog.endpoint == action)
    rows = query.order_by(UsageLog.created_at.desc()).limit(limit).all()
    items = []
    for row in rows:
        details = _json_load(row.details_json, {})
        items.append(
            {
                "id": row.id,
                "action": row.endpoint,
                "user_id": row.user_id,
                "resource": row.resource or "",
                "success": row.status_code == 200,
                "details": details.get("details", {}) if isinstance(details, dict) else {},
                "error": row.error_message or "",
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )
    return {"items": items, "total": len(items)}


def _safe_json_preview(path: Path, max_chars: int = 12000) -> Any:
    try:
        if path.stat().st_size > max_chars * 8:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@router.get("/api/v1/admin/evaluations", tags=["管理"])
def admin_evaluations(
    current_user: User = Depends(require_admin_user),
) -> Dict[str, Any]:
    from api.deps import get_runtime_config

    cfg = get_runtime_config()
    eval_dir = resolve_path(cfg.get("paths", {}).get("eval", "data/eval"))
    reports: List[Dict[str, Any]] = []
    if eval_dir.is_dir():
        files = sorted(
            [
                path
                for path in eval_dir.rglob("*.json")
                if path.is_file() and path.name != "index.json"
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:20]
        for path in files:
            reports.append(
                {
                    "name": path.name,
                    "path": str(path.relative_to(eval_dir)),
                    "size": path.stat().st_size,
                    "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    "summary": _safe_json_preview(path),
                }
            )
    return {"reports": reports, "total": len(reports)}


@router.get("/api/v1/admin/backup", tags=["管理"])
def backup_status(
    current_user: User = Depends(require_admin_user),
) -> Dict[str, Any]:
    files = sorted(
        (
            path
            for pattern in ("zhige-backup-*.zip", "zhige-backup-*.zip.enc")
            for path in BACKUP_ROOT.glob(pattern)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if BACKUP_ROOT.exists() else []
    status_file = BACKUP_ROOT / "backup-status.json"
    scheduler_status = None
    if status_file.is_file():
        try:
            scheduler_status = _safe_json_preview(status_file)
        except Exception:
            scheduler_status = {"status": "invalid"}
    return {
        "pending": pending_restore(),
        "last": last_restore_status(),
        "scheduler": scheduler_status,
        "backups": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "created_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "encrypted": path.name.endswith(".enc"),
                "download_url": f"/api/v1/admin/backup/{path.name}",
            }
            for path in files[:20]
        ],
    }


@router.post("/api/v1/admin/backup", status_code=201, tags=["管理"])
def create_workspace_backup(
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    try:
        result = create_backup()
        get_audit_logger().log(
            AuditAction.BACKUP_EXPORT,
            user_id=current_user.id,
            resource=result["filename"],
            details={"size": result["size"], "sha256": result["sha256"]},
            db=db,
        )
        return {
            **result,
            "download_url": f"/api/v1/admin/backup/{result['filename']}",
        }
    except BackupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("创建备份失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="创建备份失败") from exc


@router.get("/api/v1/admin/backup/{filename}", tags=["管理"])
def download_workspace_backup(
    filename: str,
    current_user: User = Depends(require_admin_user),
) -> FileResponse:
    allowed = filename.startswith("zhige-backup-") and (
        filename.endswith(".zip") or filename.endswith(".zip.enc")
    )
    if not allowed:
        raise HTTPException(status_code=404, detail="备份不存在")
    path = (BACKUP_ROOT / Path(filename).name).resolve()
    if not path.is_relative_to(BACKUP_ROOT.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="备份不存在")
    return FileResponse(
        path,
        filename=path.name,
        media_type=(
            "application/octet-stream"
            if path.name.endswith(".enc")
            else "application/zip"
        ),
        headers={"Cache-Control": "no-store"},
    )


def _persist_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(destination, "wb") as stream:
            while True:
                block = upload.file.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > max_bytes:
                    raise BackupError("恢复包超过允许大小")
                stream.write(block)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return size


@router.post("/api/v1/admin/restore", status_code=202, tags=["管理"])
async def stage_workspace_restore(
    file: UploadFile = File(...),
    confirm: str = Form(...),
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if confirm.strip().upper() != "RESTORE":
        raise HTTPException(status_code=400, detail="请输入 RESTORE 确认恢复")
    backup_name = (file.filename or "").lower()
    if not backup_name.endswith((".zip", ".zip.enc")):
        raise HTTPException(status_code=400, detail="恢复包必须是 ZIP 或 .zip.enc 文件")
    from src.backup_service import MAX_ARCHIVE_BYTES

    temp_dir = resolve_path("data/.restore_upload")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{current_user.id[:8]}-{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
    try:
        await run_in_threadpool(
            _persist_upload,
            file,
            temp_path,
            MAX_ARCHIVE_BYTES,
        )
        pending = await run_in_threadpool(
            stage_restore,
            temp_path,
            file.filename or "backup.zip",
        )
        get_audit_logger().log(
            AuditAction.BACKUP_RESTORE,
            user_id=current_user.id,
            resource=pending["token"],
            details={"filename": pending.get("original_name"), "status": "staged"},
            db=db,
        )
        return {
            "status": "staged",
            "requires_restart": True,
            "token": pending["token"],
            "filename": pending.get("original_name"),
            "message": "恢复包校验通过；重启知阁后将在数据库初始化前应用",
        }
    except BackupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@router.delete("/api/v1/admin/restore", tags=["管理"])
def cancel_workspace_restore(
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    cancelled = cancel_pending_restore()
    try:
        get_audit_logger().log(
            AuditAction.BACKUP_RESTORE,
            user_id=current_user.id,
            resource="restore:pending",
            details={"status": "cancelled" if cancelled else "idle"},
            db=db,
        )
    except Exception:
        logger.debug("取消恢复审计写入失败", exc_info=True)
    return {"cancelled": cancelled, "status": "cancelled" if cancelled else "idle"}
