"""身份依赖。

默认本地模式沿用固定用户、零登录门槛；设置 ``ZHIGE_AUTH_MODE=team`` 后，
同一个依赖改为校验 HttpOnly Cookie 或不透明 Bearer 令牌。业务路由无需
感知模式差异。
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.db.models import User
from src.utils import get_logger

logger = get_logger("middleware.identity")

# 固定的本地用户身份。id 写死，保证数据始终落在同一个"本地用户"下。
LOCAL_USER_ID = "00000000-0000-0000-0000-000000000001"
LOCAL_USERNAME = "local"
LOCAL_USER_EMAIL = "local@zhige.local"


def get_local_user(db: Session) -> User:
    """取（或创建）本地默认用户。"""
    user = db.query(User).filter(User.id == LOCAL_USER_ID).first()
    if user is not None:
        return user

    user = User(
        id=LOCAL_USER_ID,
        username=LOCAL_USERNAME,
        email=LOCAL_USER_EMAIL,
        is_active=True,
        is_admin=True,
        role="admin",
        password_hash="",  # 兼容历史 NOT NULL 约束；本地模式不用于登录
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("已创建本地默认用户：无需登录，所有数据归于此用户")
    return user


async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    """返回当前用户；团队模式缺少或过期凭证时返回 401。"""
    from src.team_auth import request_token, team_mode, user_from_token

    if not team_mode():
        return get_local_user(db)

    user = user_from_token(db, request_token(request))
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return user
