"""可选团队模式：登录、成员角色、知识库共享与 MCP 审批。"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.db.models import KnowledgeBase, KnowledgeBaseMember, MCPServer, User
from src.middleware.audit import AuditAction, get_audit_logger
from src.middleware.identity import get_current_user
from src.team_auth import (
    SESSION_COOKIE,
    account_locked,
    auth_mode,
    authenticate_credentials,
    create_auth_session,
    find_active_user,
    reset_login_lockout,
    effective_role,
    hash_password,
    normalize_role,
    request_token,
    require_admin_user,
    revoke_session,
    revoke_user_sessions,
    team_mode,
    user_dict,
    user_from_token,
)
from src.utils import get_logger

logger = get_logger("api.routes.team")
router = APIRouter()


def _cookie_secure() -> bool:
    return os.getenv("ZHIGE_COOKIE_SECURE", "0").strip().lower() in ("1", "true", "yes", "on")


def _set_session_cookie(response: Response, token: str) -> None:
    max_age = max(1, int(os.getenv("ZHIGE_SESSION_DAYS", "7"))) * 86400
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=max_age,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        path="/",
    )


class BootstrapBody(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    email: str = Field(..., min_length=3, max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(..., min_length=8, max_length=128)
    bootstrap_token: str = ""


class LoginBody(BaseModel):
    login: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=128)


class UserCreateBody(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    email: str = Field(..., min_length=3, max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(..., min_length=8, max_length=128)
    department: str = Field("", max_length=100)
    role: str = Field("viewer", pattern="^(viewer|editor|admin)$")


class UserUpdateBody(BaseModel):
    department: Optional[str] = Field(None, max_length=100)
    role: Optional[str] = Field(None, pattern="^(viewer|editor|admin)$")
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8, max_length=128)


class MemberBody(BaseModel):
    user_id: Optional[str] = None
    login: Optional[str] = None
    role: str = Field("viewer", pattern="^(viewer|editor)$")


class MCPDecisionBody(BaseModel):
    decision: str = Field(..., pattern="^(approved|rejected)$")
    reason: str = Field("", max_length=1000)


def _find_user(db: Session, login: str) -> Optional[User]:
    value = (login or "").strip()
    if not value:
        return None
    return db.query(User).filter((User.username == value) | (User.email == value)).first()


def _owned_or_admin_kb(db: Session, actor: User, kb_id: str) -> KnowledgeBase:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if kb is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if effective_role(actor) != "admin" and kb.user_id != actor.id:
        raise HTTPException(status_code=403, detail="只有知识库所有者或团队管理员可以管理共享")
    return kb


@router.get("/api/v1/team/status", tags=["团队"])
def team_status(request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """认证模式探测。团队模式未登录时返回 200，供前端展示登录页。"""
    configured = team_mode()
    credentialed = db.query(User).filter(User.password_hash.isnot(None), User.password_hash != "").count()
    authenticated = False
    user = None
    if configured:
        current = user_from_token(db, request_token(request))
        if current is not None:
            authenticated = True
            user = user_dict(current)
    return {
        "auth_mode": auth_mode(),
        "team_mode": configured,
        "authenticated": authenticated if configured else True,
        "bootstrap_available": configured and credentialed == 0,
        "bootstrap_token_required": bool(os.getenv("ZHIGE_BOOTSTRAP_TOKEN", "").strip()),
        "user": user,
    }


@router.post("/api/v1/team/bootstrap", tags=["团队"])
def bootstrap_admin(
    body: BootstrapBody,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not team_mode():
        raise HTTPException(status_code=409, detail="当前为本地模式；请设置 ZHIGE_AUTH_MODE=team")
    required_token = os.getenv("ZHIGE_BOOTSTRAP_TOKEN", "").strip()
    if required_token and body.bootstrap_token != required_token:
        raise HTTPException(status_code=403, detail="初始化令牌无效")
    if db.query(User).filter(User.password_hash.isnot(None), User.password_hash != "").count():
        raise HTTPException(status_code=409, detail="管理员已经初始化，请直接登录")

    username = body.username.strip()
    email = str(body.email).strip().lower()
    # 首个管理员接管历史本地用户，继续看到原有知识库与文档。
    user = db.query(User).filter(User.username == "local").first()
    if user is None:
        user = User(username=username, email=email)
        db.add(user)
    else:
        conflict = db.query(User).filter(
            User.id != user.id,
            (User.username == username) | (User.email == email),
        ).first()
        if conflict is not None:
            raise HTTPException(status_code=409, detail="用户名或邮箱已被其他账号使用")
        user.username = username
        user.email = email
    user.password_hash = hash_password(body.password)
    user.role = "admin"
    user.is_admin = True
    user.is_active = True
    db.commit()
    db.refresh(user)

    token, _ = create_auth_session(db, user, user_agent=request.headers.get("user-agent"))
    _set_session_cookie(response, token)
    try:
        get_audit_logger().log(
            AuditAction.CONFIG_CHANGE,
            user_id=user.id,
            resource="team:bootstrap",
            details={"username": user.username},
            db=db,
        )
    except Exception:
        logger.debug("初始化审计写入失败", exc_info=True)
    return {"user": user_dict(user), "token": token, "expires_in": None}


@router.post("/api/v1/team/login", tags=["团队"])
def login(
    body: LoginBody,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not team_mode():
        raise HTTPException(status_code=409, detail="当前为本地模式，无需登录")
    existing = find_active_user(db, body.login)
    if existing is not None and account_locked(existing):
        raise HTTPException(status_code=423, detail="账号已锁定，请稍后再试")
    user = authenticate_credentials(db, body.login, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="账号或密码错误")
    token, _ = create_auth_session(db, user, user_agent=request.headers.get("user-agent"))
    _set_session_cookie(response, token)
    return {"user": user_dict(user), "token": token}


@router.post("/api/v1/team/logout", tags=["团队"])
def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> Dict[str, Any]:
    revoked = revoke_session(db, request_token(request))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"status": "ok", "revoked": revoked}


@router.get("/api/v1/team/me", tags=["团队"])
def me(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    return {"user": user_dict(current_user), "auth_mode": auth_mode()}


@router.get("/api/v1/team/users", tags=["团队"])
def list_team_users(
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    rows = db.query(User).order_by(User.created_at.asc()).all()
    return {"users": [user_dict(row) for row in rows], "total": len(rows)}


@router.post("/api/v1/team/users", status_code=201, tags=["团队"])
def create_team_user(
    body: UserCreateBody,
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not team_mode():
        raise HTTPException(status_code=409, detail="团队用户仅在 ZHIGE_AUTH_MODE=team 下可用")
    username = body.username.strip()
    email = str(body.email).strip().lower()
    if db.query(User).filter((User.username == username) | (User.email == email)).first():
        raise HTTPException(status_code=409, detail="用户名或邮箱已存在")
    role = normalize_role(body.role)
    user = User(
        username=username,
        email=email,
        department=body.department.strip(),
        role=role,
        is_admin=role == "admin",
        is_active=True,
        password_hash=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    try:
        get_audit_logger().log(
            AuditAction.CONFIG_CHANGE,
            user_id=current_user.id,
            resource=f"team:user:{user.id}",
            details={
                "operation": "create_user",
                "username": user.username,
                "role": user.role,
            },
            db=db,
        )
    except Exception:
        logger.debug("创建成员审计写入失败", exc_info=True)
    return user_dict(user)


@router.patch("/api/v1/team/users/{user_id}", tags=["团队"])
def update_team_user(
    user_id: str,
    body: UserUpdateBody,
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    was_active_admin = (
        bool(user.is_active)
        and effective_role(user) == "admin"
    )
    removes_admin = body.is_active is False or (
        body.role is not None and normalize_role(body.role) != "admin"
    )
    if was_active_admin and removes_admin:
        other_admins = (
            db.query(User)
            .filter(
                User.id != user.id,
                User.is_active.is_(True),
                (User.role == "admin") | (User.is_admin.is_(True)),
            )
            .count()
        )
        if other_admins == 0:
            raise HTTPException(status_code=409, detail="必须至少保留一个启用中的管理员")
    if user.id == current_user.id and (body.is_active is False or body.role in {"viewer", "editor"}):
        raise HTTPException(status_code=409, detail="不能降级或停用当前管理员自己")
    if body.department is not None:
        user.department = body.department.strip()
    if body.role is not None:
        user.role = normalize_role(body.role)
        user.is_admin = user.role == "admin"
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.password:
        user.password_hash = hash_password(body.password)
    if body.password or body.is_active is True:
        reset_login_lockout(user)
    revoke_credentials = bool(body.password) or body.is_active is False
    db.commit()
    if revoke_credentials:
        revoke_user_sessions(db, user.id)
    db.refresh(user)
    try:
        get_audit_logger().log(
            AuditAction.CONFIG_CHANGE,
            user_id=current_user.id,
            resource=f"team:user:{user.id}",
            details={
                "operation": "update_user",
                "username": user.username,
                "role": user.role,
                "active": bool(user.is_active),
                "password_reset": bool(body.password),
                "sessions_revoked": revoke_credentials,
            },
            db=db,
        )
    except Exception:
        logger.debug("更新成员审计写入失败", exc_info=True)
    return user_dict(user)


@router.get("/api/v1/team/kbs/{kb_id}/members", tags=["团队"])
def list_kb_members(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    kb = _owned_or_admin_kb(db, current_user, kb_id)
    rows = (
        db.query(KnowledgeBaseMember, User)
        .join(User, User.id == KnowledgeBaseMember.user_id)
        .filter(KnowledgeBaseMember.kb_id == kb.id)
        .order_by(KnowledgeBaseMember.created_at.asc())
        .all()
    )
    owner = db.query(User).filter(User.id == kb.user_id).first()
    members = []
    if owner is not None:
        members.append({**user_dict(owner), "access_role": "owner", "member_id": None})
    members.extend(
        {**user_dict(user), "access_role": member.role, "member_id": member.id}
        for member, user in rows
    )
    return {"kb_id": kb.id, "members": members}


@router.post("/api/v1/team/kbs/{kb_id}/members", status_code=201, tags=["团队"])
def share_kb(
    kb_id: str,
    body: MemberBody,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not team_mode():
        raise HTTPException(status_code=409, detail="知识库共享仅在团队模式开放")
    kb = _owned_or_admin_kb(db, current_user, kb_id)
    target = db.query(User).filter(User.id == body.user_id).first() if body.user_id else _find_user(db, body.login or "")
    if target is None or not target.is_active:
        raise HTTPException(status_code=404, detail="目标用户不存在或已停用")
    if target.id == kb.user_id:
        raise HTTPException(status_code=409, detail="所有者无需重复共享")
    row = db.query(KnowledgeBaseMember).filter(
        KnowledgeBaseMember.kb_id == kb.id, KnowledgeBaseMember.user_id == target.id
    ).first()
    if row is None:
        row = KnowledgeBaseMember(kb_id=kb.id, user_id=target.id, role=body.role)
        db.add(row)
    else:
        row.role = body.role
    db.commit()
    db.refresh(row)
    try:
        get_audit_logger().log(
            AuditAction.KB_SHARE,
            user_id=current_user.id,
            resource=kb.id,
            details={"member": target.id, "role": row.role},
            db=db,
        )
    except Exception:
        logger.debug("共享审计写入失败", exc_info=True)
    return {"id": row.id, "kb_id": kb.id, "user_id": target.id, "role": row.role}


@router.delete("/api/v1/team/kbs/{kb_id}/members/{member_id}", tags=["团队"])
def unshare_kb(
    kb_id: str,
    member_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    kb = _owned_or_admin_kb(db, current_user, kb_id)
    row = db.query(KnowledgeBaseMember).filter(
        KnowledgeBaseMember.id == member_id, KnowledgeBaseMember.kb_id == kb.id
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="成员不存在")
    shared_user_id = row.user_id
    db.delete(row)
    db.commit()
    try:
        get_audit_logger().log(
            AuditAction.KB_SHARE,
            user_id=current_user.id,
            resource=kb.id,
            details={"member": shared_user_id, "operation": "unshare"},
            db=db,
        )
    except Exception:
        logger.debug("取消共享审计写入失败", exc_info=True)
    return {"deleted": member_id}


@router.get("/api/v1/team/mcp-servers", tags=["团队"])
def team_mcp_servers(
    status: Optional[str] = None,
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    query = db.query(MCPServer)
    if status:
        query = query.filter(MCPServer.approval_status == status)
    rows = query.order_by(MCPServer.created_at.desc()).limit(200).all()
    return {
        "servers": [
            {
                "id": row.id,
                "user_id": row.user_id,
                "name": row.name,
                "command": row.command,
                "enabled": bool(row.enabled),
                "approval_status": row.approval_status,
                "last_error": row.last_error,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ],
        "total": len(rows),
    }


@router.post("/api/v1/team/mcp-servers/{server_id}/decision", tags=["团队"])
def decide_mcp_server(
    server_id: str,
    body: MCPDecisionBody,
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    row = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="MCP 服务不存在")
    row.approval_status = body.decision
    row.approved_by = current_user.id
    row.approved_at = datetime.utcnow()
    if body.decision == "rejected":
        row.enabled = False
        row.last_error = body.reason.strip()[:1000] or "管理员已拒绝"
    db.commit()
    try:
        from api.routes.agent import reset_agent_cache

        reset_agent_cache(row.user_id)
    except Exception:
        pass
    try:
        get_audit_logger().log(
            AuditAction.MCP_APPROVE,
            user_id=current_user.id,
            resource=row.id,
            details={"decision": body.decision, "reason": body.reason[:200]},
            db=db,
        )
    except Exception:
        logger.debug("MCP 审批审计写入失败", exc_info=True)
    return {"id": row.id, "approval_status": row.approval_status, "enabled": bool(row.enabled)}
