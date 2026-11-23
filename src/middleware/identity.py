"""HTTP / WebSocket 身份依赖。

两种部署模式：

- ``enterprise``（默认）：必须登录，支持 HttpOnly Cookie 与 Bearer Token；
- ``local``：保留零登录体验，但只允许回环地址访问。

本地模式不是“默认管理员绕过”：一旦服务从非回环地址访问，认证依赖会直接
拒绝。Docker / K8s 必须使用 enterprise 模式。
"""
from __future__ import annotations

import os
from typing import Callable, Optional
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.db.models import AuthSession, User
from src.security import (
    AUTH_SESSION_COOKIE,
    AuthError,
    auth_mode,
    effective_role,
    extract_bearer_token,
    is_local_auth,
    is_loopback_host,
    resolve_session,
    role_allowed,
)
from src.utils import get_logger

logger = get_logger("middleware.identity")

LOCAL_USER_ID = "00000000-0000-0000-0000-000000000001"
LOCAL_USERNAME = "local"
LOCAL_USER_EMAIL = "local@zhige.local"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def get_local_user(db: Session) -> User:
    """取（或创建）本地模式用户；该用户只允许从回环地址访问。"""
    user = db.query(User).filter(User.id == LOCAL_USER_ID).first()
    if user is None:
        user = User(
            id=LOCAL_USER_ID,
            username=LOCAL_USERNAME,
            email=LOCAL_USER_EMAIL,
            department="",
            role="admin",
            is_active=True,
            is_admin=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("已创建回环地址专用本地用户")
        return user

    changed = False
    if str(user.role or "") != "admin":
        user.role = "admin"
        changed = True
    if not bool(user.is_admin):
        user.is_admin = True
        changed = True
    if not bool(user.is_active):
        user.is_active = True
        changed = True
    if changed:
        db.commit()
        db.refresh(user)
    return user


def _local_host_allowed(host: Optional[str]) -> bool:
    if is_loopback_host(host):
        return True
    allow_testclient = str(os.getenv("AUTH_ALLOW_TESTCLIENT", "0")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return allow_testclient and str(host or "").lower() == "testclient"


def _cookie_csrf_ok(request: Request) -> bool:
    """Cookie 认证的写请求若带 Origin，必须与 Host 同源。"""
    origin = request.headers.get("origin")
    if not origin:
        return True
    expected = request.headers.get("host", "")
    try:
        return urlsplit(origin).netloc.lower() == expected.lower()
    except Exception:
        return False


def _http_auth_token(request: Request) -> tuple[Optional[str], str]:
    bearer = extract_bearer_token(request.headers.get("authorization"))
    if bearer:
        return bearer, "bearer"
    cookie = request.cookies.get(AUTH_SESSION_COOKIE)
    if cookie:
        return cookie, "cookie"
    return None, "none"


async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    """返回已认证用户；企业模式没有有效会话时返回 401。"""
    try:
        mode = auth_mode()
    except AuthError as exc:
        logger.error("认证配置错误: %s", exc)
        raise HTTPException(status_code=503, detail="认证配置错误") from exc

    if mode == "local":
        host = request.client.host if request.client else None
        if not _local_host_allowed(host):
            logger.warning("拒绝非回环地址使用 local 认证模式: %s", host)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="本地无登录模式仅允许回环地址；远程访问请启用企业认证",
            )
        return get_local_user(db)

    token, source = _http_auth_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if source == "cookie" and request.method.upper() not in _SAFE_METHODS:
        if not _cookie_csrf_ok(request):
            raise HTTPException(status_code=403, detail="跨站请求被拒绝")
    try:
        user, session = resolve_session(db, token)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    request.state.auth_session = session
    request.state.auth_source = source
    return user


async def authenticate_websocket(websocket, db: Session) -> tuple[User, Optional[AuthSession]]:
    """在 accept 前完成 WebSocket 认证；失败关闭码为 4401。"""
    if is_local_auth():
        host = websocket.client.host if websocket.client else None
        if not _local_host_allowed(host):
            await websocket.close(code=4403, reason="local auth is restricted to loopback")
            raise AuthError("本地无登录模式仅允许回环地址")
        return get_local_user(db), None

    token = extract_bearer_token(websocket.headers.get("authorization"))
    token = token or websocket.cookies.get(AUTH_SESSION_COOKIE)
    token = token or websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4401, reason="authentication required")
        raise AuthError("authentication required")
    try:
        user, session = resolve_session(db, token)
    except AuthError as exc:
        await websocket.close(code=4401, reason=str(exc)[:100])
        raise
    return user, session


def require_roles(*allowed_roles: str) -> Callable[..., User]:
    """生成角色依赖；admin 始终拥有访问权。"""

    async def dependency(current_user: User = Depends(get_current_user)) -> User:
        if not role_allowed(effective_role(current_user), allowed_roles):
            raise HTTPException(status_code=403, detail="当前角色无权执行此操作")
        return current_user

    return dependency


require_admin = require_roles("admin")
require_auditor = require_roles("auditor")
require_kb_admin = require_roles("kb_admin")


__all__ = [
    "LOCAL_USER_ID",
    "authenticate_websocket",
    "get_current_user",
    "get_local_user",
    "require_admin",
    "require_auditor",
    "require_kb_admin",
    "require_roles",
]
