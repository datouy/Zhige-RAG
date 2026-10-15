"""反馈层路由：用户反馈 + 差评导出评估集（持续迭代闭环）。

端点：
- POST /api/v1/feedback          提交一次问答反馈（helpful / not_helpful + 纠错）
- GET  /api/v1/feedback          查询反馈列表（本人；管理员可看全部）
- GET  /api/v1/feedback/stats    反馈统计（差评率、差评 top 问题类型）
- POST /api/v1/eval/export-feedback
                                把未处理的差评（含纠错文本）导出为评估集
                                jsonl——bad case 直接变成回归测试资产。

迭代闭环：问答（带 verification 报告）→ 用户反馈 → 差评入库 →
导出评估集 → scripts/evaluate 回归 → 修订切块/检索/入库策略。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.db.models import FeedbackRecord, User
from src.middleware.identity import get_current_user
from src.utils import get_logger, resolve_path

logger = get_logger("api.routes.feedback")

router = APIRouter()


def _auto_route_feedback(body: "FeedbackBody"):
    """差评自动分流：用 d1 ``choice`` 原语把反馈归因到处理环节。

    仅在 ``decision.enabled`` 且 ``decision.feedback_route.enabled`` 打开时生效；
    未启用 / 决策失败 / 置信度不足时返回 ``None``，调用方保持"未分类"原行为。
    """
    try:
        from api.deps import get_decision_provider, get_runtime_config

        dcfg = (get_runtime_config().get("decision", {}) or {}).get("feedback_route", {}) or {}
        if not dcfg.get("enabled"):
            return None
        provider = get_decision_provider()
        if provider is None:
            return None
        from src.feedback_router import classify_feedback

        return classify_feedback(
            body.query,
            body.answer,
            body.correction,
            provider,
            min_confidence=float(dcfg.get("min_confidence", 0.5)),
        )
    except Exception as exc:  # noqa: BLE001 - 分流失败绝不能影响反馈提交
        logger.warning("反馈自动分流失败（已跳过）: %s", exc)
        return None


class FeedbackBody(BaseModel):
    """POST /api/v1/feedback 请求体。"""

    query: str = Field(..., min_length=1, description="被评价的问题")
    answer: str = Field("", description="当时的回答（可空）")
    sources_json: str = Field("", description="引用来源 JSON 字符串（可空）")
    rating: str = Field(..., description="helpful / not_helpful")
    correction: str = Field("", description="纠正/期望的答案（差评时强烈建议填写）")
    question_type: str = Field("", description="问题分类标签（可选，便于归因）")


@router.post("/api/v1/feedback", tags=["反馈"])
def submit_feedback(
    body: FeedbackBody,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """提交一次问答反馈。需要认证。

    rating 必须是 helpful / not_helpful；not_helpful 会被反馈统计聚合，
    带纠错文本的差评可通过 /eval/export-feedback 变成评估集样本。
    """
    user: User = current_user
    rating = body.rating.strip().lower()
    if rating not in {"helpful", "not_helpful"}:
        raise HTTPException(status_code=422, detail="rating 必须是 helpful 或 not_helpful")

    # 差评自动分流（d1 choice 原语）：前端没填 question_type 时自动归因，
    # 让 /feedback/stats 的"差评按类型归因"真正可用；失败静默跳过。
    question_type: Optional[str] = (body.question_type or "").strip()[:50] or None
    auto_route: Optional[Dict[str, Any]] = None
    if question_type is None and rating == "not_helpful":
        route = _auto_route_feedback(body)
        if route is not None:
            question_type = route.category[:50]
            auto_route = route.to_dict()

    rec = FeedbackRecord(
        user_id=user.id,
        query=body.query.strip()[:4000],
        answer=(body.answer or "")[:8000],
        sources_json=(body.sources_json or "")[:8000],
        rating=rating,
        correction=(body.correction or "").strip()[:8000] or None,
        question_type=question_type,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    logger.info("反馈已记录 user=%s rating=%s id=%s type=%s", user.id, rating, rec.id, question_type)
    resp: Dict[str, Any] = {"id": rec.id, "status": "ok", "question_type": question_type}
    if auto_route is not None:
        resp["auto_route"] = auto_route
    return resp


@router.get("/api/v1/feedback", tags=["反馈"])
def list_feedback(
    rating: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """查询反馈记录。普通用户只见本人反馈；管理员可见全部。需要认证。"""
    user: User = current_user
    limit = max(1, min(int(limit), 200))
    q = db.query(FeedbackRecord)
    if not user.is_admin:
        q = q.filter(FeedbackRecord.user_id == user.id)
    if rating:
        q = q.filter(FeedbackRecord.rating == rating.strip().lower())
    rows = q.order_by(FeedbackRecord.created_at.desc()).limit(limit).all()
    return {
        "items": [
            {
                "id": r.id,
                "query": r.query,
                "answer": (r.answer or "")[:500],
                "rating": r.rating,
                "correction": r.correction,
                "question_type": r.question_type,
                "resolved": bool(r.resolved),
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None,
            }
            for r in rows
        ],
        "total": len(rows),
    }


@router.get("/api/v1/feedback/stats", tags=["反馈"])
def feedback_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """反馈统计：好评/差评数、差评率、未处理差评、按问题类型归因。需要认证。"""
    user: User = current_user
    q = db.query(FeedbackRecord)
    if not user.is_admin:
        q = q.filter(FeedbackRecord.user_id == user.id)
    rows = q.all()
    total = len(rows)
    bad = [r for r in rows if r.rating == "not_helpful"]
    good = total - len(bad)
    by_type: Dict[str, int] = {}
    for r in bad:
        key = r.question_type or "未分类"
        by_type[key] = by_type.get(key, 0) + 1
    return {
        "total": total,
        "helpful": good,
        "not_helpful": len(bad),
        "bad_rate": round(len(bad) / total, 4) if total else 0.0,
        "unresolved_bad": sum(1 for r in bad if not r.resolved),
        "bad_by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
    }


class ExportResult(BaseModel):
    path: str
    exported: int


@router.post("/api/v1/eval/export-feedback", tags=["反馈"])
def export_feedback_evalset(
    only_with_correction: bool = True,
    limit: int = 200,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ExportResult:
    """把差评导出为评估集（jsonl：{question, reference_answer, source}）。

    只处理未导出过的差评（resolved=False）；导出后标记 resolved=True，
    保证同一 bad case 只进一次回归集。需要认证；普通用户仅导出自己的。
    """
    user: User = current_user
    q = db.query(FeedbackRecord).filter(
        FeedbackRecord.rating == "not_helpful",
        FeedbackRecord.resolved.is_(False),  # type: ignore[union-attr]
    )
    if not user.is_admin:
        q = q.filter(FeedbackRecord.user_id == user.id)
    if only_with_correction:
        q = q.filter(FeedbackRecord.correction.isnot(None), FeedbackRecord.correction != "")  # type: ignore[union-attr]
    rows = q.order_by(FeedbackRecord.created_at.asc()).limit(max(1, min(int(limit), 500))).all()
    if not rows:
        return ExportResult(path="", exported=0)

    out_path = resolve_path(f"data/eval/feedback_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "question": r.query,
                        "reference_answer": r.correction or "",
                        "source": f"feedback:{r.id}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    for r in rows:
        r.resolved = True
    db.commit()
    logger.info("差评导出评估集 %d 条 -> %s", len(rows), out_path)
    return ExportResult(path=str(out_path), exported=len(rows))


class MemoryBody(BaseModel):
    """长期记忆管理请求体。"""

    key: str = Field(..., min_length=1, max_length=200)
    value: str = Field(..., min_length=1, max_length=2000)


@router.get("/api/v1/memory", tags=["反馈"])
def list_memory(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """列出当前用户的长期记忆。需要认证。"""
    from src.db.models import LongTermMemory

    rows = (
        db.query(LongTermMemory)
        .filter(LongTermMemory.user_id == current_user.id)
        .order_by(LongTermMemory.updated_at.desc())
        .all()
    )
    return {
        "items": [
            {"key": r.mem_key, "value": r.mem_value, "source": r.source}
            for r in rows
        ],
        "total": len(rows),
    }


@router.post("/api/v1/memory", tags=["反馈"])
def upsert_memory(
    body: MemoryBody,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """手动写入/更新一条长期记忆。需要认证。"""
    from datetime import timedelta

    from src.db.models import LongTermMemory

    row = (
        db.query(LongTermMemory)
        .filter(LongTermMemory.user_id == current_user.id, LongTermMemory.mem_key == body.key.strip())
        .first()
    )
    if row is None:
        row = LongTermMemory(user_id=current_user.id, mem_key=body.key.strip(), mem_value=body.value.strip(), source="user")
        db.add(row)
    else:
        row.mem_value = body.value.strip()
        row.updated_at = datetime.utcnow() + timedelta(seconds=1)  # 保证排序可见更新
    db.commit()
    return {"status": "ok", "key": body.key.strip()}


@router.delete("/api/v1/memory/{key}", tags=["反馈"])
def delete_memory(
    key: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """删除一条长期记忆。需要认证。"""
    from src.db.models import LongTermMemory

    n = (
        db.query(LongTermMemory)
        .filter(LongTermMemory.user_id == current_user.id, LongTermMemory.mem_key == key)
        .delete()
    )
    db.commit()
    return {"status": "ok", "deleted": n}
