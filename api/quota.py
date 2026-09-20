"""订阅配额校验与用量记录。

P5 审计修复：``User.max_chunks`` / ``User.max_queries_per_day`` 与
``UsageLog`` 表此前从未被任何代码读取执行，free 用户可无限上传与提问。
本模块把套餐限额真正接入 ingest / chat 路径。

约定：
- 限额字段为 ``-1`` 或 ``None`` 表示不限制（enterprise）。
- 用量以 ``UsageLog`` 表为准：每次问答成功后写入一条记录。
"""
from __future__ import annotations

from datetime import datetime, time as dt_time
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.db.models import User, UsageLog
from src.utils import get_logger

logger = get_logger("api.quota")

# 计入"每日提问次数"的端点标识（UsageLog.endpoint）
CHAT_ENDPOINT = "/api/v1/chat"


def _today_start() -> datetime:
    now = datetime.utcnow()
    return datetime.combine(now.date(), dt_time.min)


def queries_today(db: Session, user_id: str, endpoint: str = CHAT_ENDPOINT) -> int:
    """统计用户今天的调用次数。"""
    return (
        db.query(UsageLog)
        .filter(
            UsageLog.user_id == user_id,
            UsageLog.endpoint == endpoint,
            UsageLog.created_at >= _today_start(),
        )
        .count()
    )


def check_query_quota(db: Session, user: User, endpoint: str = CHAT_ENDPOINT) -> None:
    """校验每日问答配额，超限抛 429。"""
    limit = user.max_queries_per_day
    if limit is None or limit < 0:  # 不限制
        return
    used = queries_today(db, user.id, endpoint=endpoint)
    if used >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"今日问答次数已达套餐上限（{limit} 次/天），请升级套餐或明天再试",
        )


def record_usage(
    db: Session,
    user_id: str,
    endpoint: str,
    method: str = "POST",
    status_code: int = 200,
    latency_ms: int = 0,
    tokens_used: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """写入一条用量日志。失败只记 warning，不影响主流程。"""
    try:
        db.add(
            UsageLog(
                user_id=user_id,
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                latency_ms=int(latency_ms),
                tokens_used=int(tokens_used),
                error_message=error_message,
            )
        )
        db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("写入 UsageLog 失败: %s", exc)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass


def check_chunk_quota(user: User, current_chunks: int, new_chunks: int) -> None:
    """校验知识库分块总量配额，超限抛 413。"""
    limit = user.max_chunks
    if limit is None or limit < 0:  # 不限制
        return
    if current_chunks + new_chunks > limit:
        raise HTTPException(
            status_code=413,
            detail=(
                f"知识库容量已达套餐上限：现有 {current_chunks} 块，本次新增 "
                f"{new_chunks} 块将超过上限 {limit} 块，请删除部分文档或升级套餐"
            ),
        )
