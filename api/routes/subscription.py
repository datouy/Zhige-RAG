"""订阅管理路由 - 查询套餐、当前用量、升级。"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from src.db.database import get_db
from src.db.models import SUBSCRIPTION_TIERS, User
from src.middleware.auth import get_current_user
from src.utils import get_logger

from api.quota import CHAT_ENDPOINT, queries_today

logger = get_logger("api.routes.subscription")

router = APIRouter(prefix="/api/v1/subscription", tags=["订阅管理"])


class TierLimits(BaseModel):
    """套餐限制详情。"""
    max_chunks: int
    max_queries_per_day: int
    available_models: list[str]
    reranker: bool
    max_kb: int


class TierInfo(BaseModel):
    """套餐信息。"""
    name: str
    display_name: str
    limits: TierLimits


class UsageInfo(BaseModel):
    """当前用量。"""
    chunks_used: int
    chunks_limit: int
    queries_today: int
    queries_limit: int


class SubscriptionResponse(BaseModel):
    """订阅信息响应。"""
    current_tier: str
    usage: UsageInfo
    tiers: dict


class UpgradeRequest(BaseModel):
    """升级请求。"""
    target_tier: str


# 套餐展示名称映射
TIER_DISPLAY_NAMES = {
    "free": "免费版",
    "pro": "专业版",
    "enterprise": "企业版",
}


@router.get("/tiers")
def list_tiers() -> dict:
    """列出所有可用套餐。"""
    tiers = {}
    for name, limits in SUBSCRIPTION_TIERS.items():
        tiers[name] = {
            "name": name,
            "display_name": TIER_DISPLAY_NAMES.get(name, name),
            "limits": limits,
        }
    return {"tiers": tiers}


@router.get("/current", response_model=SubscriptionResponse)
def get_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SubscriptionResponse:
    """获取当前用户的订阅信息（真实用量）。

    P5 审计修复：之前 chunks_used / queries_today 永远返回 0（TODO 未完成）。
    现在从租户向量库统计已用分块，从 UsageLog 统计今日问答次数。
    """
    chunks_used = 0
    try:
        from api.deps import get_runtime_config
        from src.factories import TenantAwareFactory

        store = TenantAwareFactory.get_chroma_store(current_user.id, get_runtime_config())
        chunks_used = store.count()
    except Exception as exc:  # noqa: BLE001
        logger.warning("统计向量库用量失败: %s", exc)

    queries_today_used = queries_today(db, current_user.id)

    return SubscriptionResponse(
        current_tier=current_user.subscription_tier,
        usage=UsageInfo(
            chunks_used=chunks_used,
            chunks_limit=current_user.max_chunks,
            queries_today=queries_today_used,
            queries_limit=current_user.max_queries_per_day,
        ),
        tiers=SUBSCRIPTION_TIERS,
    )


@router.post("/upgrade")
def upgrade_tier(
    req: UpgradeRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """申请升级套餐（记录请求，实际支付由外部系统处理）。"""
    if req.target_tier not in SUBSCRIPTION_TIERS:
        raise HTTPException(status_code=400, detail=f"无效的套餐：{req.target_tier}")

    if req.target_tier == current_user.subscription_tier:
        raise HTTPException(status_code=400, detail="您已经是这个套餐了")

    # TODO: 对接支付系统
    return {
        "message": f"升级到 {TIER_DISPLAY_NAMES.get(req.target_tier, req.target_tier)} 申请已记录",
        "target_tier": req.target_tier,
        "status": "pending_payment",
    }
