"""本地身份依赖：所有请求都归属唯一的本地用户。

知阁定位是**本地开源知识库工具**，不提供注册与登录 —— 拉起来就能用，
"先注册一个账号再能提问" 纯属多余的门槛。

那为什么还需要这个模块？因为数据表（文档、知识库、会话、反馈、长期记忆）
都用 ``user_id`` 标记归属。这里给所有请求固定一个本地用户身份，
**保持归属关系明确**，同时用户完全不需要任何账号。

设计要点
--------
- ``LOCAL_USER_ID`` 写死，保证换机器/重启后历史数据仍落在同一个身份下；
- 本地用户给予 enterprise 级额度（这是你自己的机器，不该被配额挡住）；
- 首次访问时自动建行，之后直接复用。
"""
from __future__ import annotations

from fastapi import Depends
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
        subscription_tier="enterprise",
        max_chunks=-1,
        max_queries_per_day=-1,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("已创建本地默认用户：无需登录，所有数据归于此用户")
    return user


async def get_current_user(db: Session = Depends(get_db)) -> User:
    """当前用户 —— 即本地用户。

    路由统一用 ``Depends(get_current_user)`` 拿身份，因此这里没有任何
    token / 会话 / 密码相关的判断：请求进来就是本地用户。
    """
    return get_local_user(db)
