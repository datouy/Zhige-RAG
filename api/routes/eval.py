"""评估路由。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.deps import get_runtime_config, get_runtime_pipeline
from src.db.database import get_db
from src.db.models import User
from src.kb_service import KBError, get_accessible_kb
from src.middleware.audit import AuditAction, get_audit_logger
from src.middleware.identity import get_current_user
from src.utils import get_logger, resolve_path

logger = get_logger("api.routes.eval")

router = APIRouter()


def _eval_path(raw: str) -> Path:
    """把配置中的 data/eval 相对路径映射到当前工作区评估目录。"""
    cfg = get_runtime_config()
    eval_root = resolve_path(cfg.get("paths", {}).get("eval", "data/eval"))
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    parts = candidate.parts
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "eval":
        candidate = Path(*parts[2:])
    return (eval_root / candidate).resolve()


@router.post("/api/v1/eval/run", tags=["评估"])
def run_evaluation(
    kb_id: Optional[str] = Query(None, description="要评估的知识库；缺省为默认知识库"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """运行评估（使用已配置的评估集）。需要认证。

    必须是 ``def``（同步端点）：``evaluate()`` 会循环执行全量
    ``answer_with_timing``，分钟级阻塞。之前写成 ``async def`` 直接调用，
    会把整个事件循环（包括 /api/health 探针）卡死，k8s 会判死重启。
    FastAPI 自动把 ``def`` 端点调度到线程池，不会阻塞事件循环。
    """
    user: User = current_user  # type: ignore[assignment]
    try:
        from scripts.evaluate import (
            archive_report,
            evaluate as run_eval,
            load_eval_dataset,
        )

        cfg = get_runtime_config()
        dataset_path = _eval_path(
            cfg.get("evaluation", {}).get("dataset_path", "data/eval/eval_set.jsonl")
        )

        if not dataset_path.exists():
            raise HTTPException(
                status_code=404, detail=f"评估集不存在: {dataset_path}"
            )

        kb_payload = None
        owner_id = user.id
        if kb_id:
            try:
                kb, _ = get_accessible_kb(db, user, kb_id, "viewer")
            except KBError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            owner_id = kb.user_id
            kb_payload = {
                "id": kb.id,
                "name": kb.name,
                "collection_name": kb.collection_name,
                "owner_id": kb.user_id,
            }

        # P1.4：复用单例；显式知识库时按该库 collection 评估。
        pipeline = get_runtime_pipeline(user_id=owner_id, kb=kb_payload)
        items = load_eval_dataset(dataset_path)
        summary = run_eval(pipeline, items)
        report_path = _eval_path(
            cfg.get("evaluation", {}).get("report_path", "data/eval/report.json")
        )
        archive = archive_report(summary, report_path)
        try:
            get_audit_logger().log(
                AuditAction.EVALUATION_RUN,
                user_id=user.id,
                resource=kb_id or "default",
                details={
                    "kb_id": kb_id,
                    "archive_path": archive.get("archive_path"),
                    "samples": summary.get("samples", 0),
                    "retrieval_hit_rate": summary.get("retrieval_hit_rate"),
                },
                db=db,
            )
        except Exception:
            logger.debug("评估审计写入失败", exc_info=True)
        return {
            **summary,
            "kb": ({"id": kb_payload["id"], "name": kb_payload["name"]} if kb_payload else None),
            "archive_path": archive.get("archive_path"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("评估失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
