"""认证依赖：从 JWT Token 中解析当前用户。

默认是**单用户本地模式**：不要求登录，所有请求自动落到一个本地用户上。

为什么这么设计
--------------
知阁定位是"本地开源 RAG 工具"，不是 SaaS。个人开发者与本地私有化的中小企业，
要的是**拉起来就能用** —— 先注册一个账号再能提问，纯属多余的门槛。

关闭认证（默认）
    - 无需注册登录，打开即用；
    - 所有数据归属固定 id 的"本地用户"，不会串到别人那里；
    - 给 enterprise 级额度（这是你自己的机器，不该被配额挡住）。

打开认证
    把 ``auth.enabled`` 设为 true（或 ``AUTH_ENABLED=1``），即恢复完整的
    JWT 认证与多租户隔离 —— 适合"一台服务器多人共用"的场景。
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.auth.jwt_handler import decode_token
from src.db.database import get_db
from src.db.models import User
from src.utils import get_logger

logger = get_logger("middleware.auth")

# auto_error=False：未启用认证时不会有 Authorization 头，不能让它直接 403
security = HTTPBearer(auto_error=False)

# 本地单用户模式的固定身份。id 写死，保证数据始终落在同一个"本地用户"下
# （换机器/重装后只要库还在，历史文档与知识库都还在）。
LOCAL_USER_ID = "00000000-0000-0000-0000-000000000001"
LOCAL_USERNAME = "local"
LOCAL_USER_EMAIL = "local@zhige.local"


def auth_enabled() -> bool:
    """是否要求登录。

    优先级：环境变量 ``AUTH_ENABLED`` > ``config.yaml`` 的 ``auth.enabled`` > 默认 False。
    """
    env = os.getenv("AUTH_ENABLED")
    if env is not None and env.strip() != "":
        return env.strip().lower() in ("1", "true", "yes", "on")
    try:
        from src.utils import load_config

        cfg = load_config("config/config.yaml") or {}
        return bool((cfg.get("auth") or {}).get("enabled", False))
    except Exception:  # noqa: BLE001
        return False


def get_local_user(db: Session) -> User:
    """取（或创建）本地默认用户。"""
    user = db.query(User).filter(User.id == LOCAL_USER_ID).first()
    if user is not None:
        return user

    user = User(
        id=LOCAL_USER_ID,
        username=LOCAL_USERNAME,
        email=LOCAL_USER_EMAIL,
        password_hash="!",  # 本地用户不走密码登录，占位即可
        is_active=True,
        is_admin=True,
        subscription_tier="enterprise",
        max_chunks=-1,
        max_queries_per_day=-1,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("已创建本地默认用户：无需登录，所有数据归于此用户")
    return user


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """解析当前用户。

    - 未启用认证（默认）：返回本地默认用户，忽略是否携带 token；
    - 已启用认证：校验 Bearer access token。
    """
    if not auth_enabled():
        return get_local_user(db)

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少认证信息",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 已过期或无效",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 Token 类型",
        )

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在或已禁用",
        )

    return user
