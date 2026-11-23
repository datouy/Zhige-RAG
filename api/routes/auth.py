"""企业身份与管理路由。

- 登录 / 登出 / 当前用户 / 修改自己的密码
- 管理员创建、停用、调整角色与重置密码

会话 Token 仅返回一次；浏览器同时写入 HttpOnly、SameSite=Strict Cookie。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.db.models import AuthSession, User
from src.middleware.identity import get_current_user, require_admin
from src.rate_limit import get_real_client_ip, limiter
from src.security import (
    AUTH_SESSION_COOKIE,
    USER_ROLES,
    AuthError,
    authenticate,
    cookie_secure,
    create_session,
    create_user,
    effective_role,
    password_policy_errors,
    revoke_session,
    revoke_user_sessions,
    session_ttl_minutes,
    set_password,
)
from src.utils import get_logger

logger = get_logger("api.routes.auth")
router = APIRouter()


class LoginBody(BaseModel):
    username: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=256)


class ChangePasswordBody(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=12, max_length=256)


class CreateUserBody(BaseModel):
    username: str = Field(..., min_length=2, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=12, max_length=256)
    role: str = Field("member", pattern=r"^(member|kb_admin|admin|auditor)$")
    department: str = Field("", max_length=100)


class UpdateUserBody(BaseModel):
    role: Optional[str] = Field(None, pattern=r"^(member|kb_admin|admin|auditor)$")
    department: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None


class ResetPasswordBody(BaseModel):
    new_password: str = Field(..., min_length=12, max_length=256)


def _user_dict(user: User) -> Dict[str, Any]:
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "department": user.department or "",
        "role": effective_role(user),
        "is_active": bool(user.is_active),
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "updated_at": user.updated_at.isoformat() if user.updated_at else None,
    }


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=AUTH_SESSION_COOKIE,
        value=token,
        max_age=session_ttl_minutes() * 60,
        httponly=True,
        secure=cookie_secure(),
        samesite="strict",
        path="/",
    )


@router.post("/api/v1/auth/login", tags=["认证"])
@limiter.limit("10/minute")
async def login(
    request: Request,
    response: Response,
    body: LoginBody,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    try:
        user = authenticate(db, body.username, body.password)
        token, session = create_session(
            db,
            user,
            ip_address=get_real_client_ip(request),
            user_agent=request.headers.get("user-agent", ""),
        )
    except AuthError as exc:
        logger.warning("登录失败 username=%s ip=%s", body.username, get_real_client_ip(request))
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    _set_auth_cookie(response, token)
    logger.info("用户登录成功 user=%s role=%s", user.username, effective_role(user))
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": max(0, int((session.expires_at - session.created_at).total_seconds())),
        "user": _user_dict(user),
    }


@router.post("/api/v1/auth/logout", tags=["认证"])
async def logout(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    session = getattr(request.state, "auth_session", None)
    if isinstance(session, AuthSession):
        revoke_session(db, session)
    response.delete_cookie(AUTH_SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/api/v1/auth/me", tags=["认证"])
async def me(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    return {"user": _user_dict(current_user)}


@router.post("/api/v1/auth/change-password", tags=["认证"])
async def change_password(
    body: ChangePasswordBody,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    try:
        authenticate(db, current_user.username, body.current_password)
        set_password(db, current_user, body.new_password)
        revoked = revoke_user_sessions(db, current_user.id)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response.delete_cookie(AUTH_SESSION_COOKIE, path="/")
    return {"ok": True, "revoked_sessions": revoked, "message": "密码已修改，请重新登录"}


@router.get("/api/v1/admin/users", tags=["用户管理"])
async def list_users(
    limit: int = 200,
    offset: int = 0,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    users = db.query(User).order_by(User.created_at.asc()).offset(offset).limit(limit).all()
    return {"users": [_user_dict(user) for user in users], "total": db.query(User).count()}


@router.post("/api/v1/admin/users", status_code=201, tags=["用户管理"])
async def create_account(
    body: CreateUserBody,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    try:
        user = create_user(
            db,
            username=body.username.strip(),
            email=body.email.strip().lower(),
            password=body.password,
            role=body.role,
            department=body.department.strip(),
        )
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("管理员 %s 创建用户 %s role=%s", current_user.username, user.username, body.role)
    return {"user": _user_dict(user)}


@router.patch("/api/v1/admin/users/{user_id}", tags=["用户管理"])
async def update_account(
    user_id: str,
    body: UpdateUserBody,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user.id == current_user.id and body.is_active is False:
        raise HTTPException(status_code=400, detail="不能停用当前登录的管理员账号")
    if body.role is not None:
        if body.role not in USER_ROLES:
            raise HTTPException(status_code=400, detail="未知角色")
        user.role = body.role
        user.is_admin = body.role == "admin"
    if body.department is not None:
        user.department = body.department.strip()
    if body.is_active is not None:
        user.is_active = bool(body.is_active)
        if not user.is_active:
            revoke_user_sessions(db, user.id)
    db.commit()
    db.refresh(user)
    logger.info("管理员 %s 更新用户 %s", current_user.username, user.username)
    return {"user": _user_dict(user)}


@router.post("/api/v1/admin/users/{user_id}/reset-password", tags=["用户管理"])
async def reset_password(
    user_id: str,
    body: ResetPasswordBody,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    errors = password_policy_errors(body.new_password, user.username)
    if errors:
        raise HTTPException(status_code=400, detail="；".join(errors))
    try:
        set_password(db, user, body.new_password)
        revoked = revoke_user_sessions(db, user.id)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.warning("管理员 %s 重置用户 %s 密码", current_user.username, user.username)
    return {"ok": True, "revoked_sessions": revoked}


__all__ = ["router"]
