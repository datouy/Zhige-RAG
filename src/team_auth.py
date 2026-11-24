"""可选团队认证：口令、会话令牌、角色和知识库权限。

默认 ``ZHIGE_AUTH_MODE=local`` 时不启用这些入口；团队模式只需要切换环境
变量即可保护现有 API。口令使用标准库 PBKDF2，数据库只保存不透明令牌的
SHA-256 摘要，避免任何可直接重放的凭证落盘。
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta
from typing import Iterable, Optional

from fastapi import Depends, HTTPException, Request, WebSocket
from sqlalchemy.orm import Session

from src.db.models import AuthSession, User
from src.utils import get_logger

logger = get_logger("team_auth")

SESSION_COOKIE = "zhige_session"
ROLE_LEVEL = {"viewer": 1, "editor": 2, "admin": 3}
PBKDF2_ITERATIONS = int(os.getenv("ZHIGE_PASSWORD_ITERATIONS", "390000"))
SESSION_DAYS = max(1, int(os.getenv("ZHIGE_SESSION_DAYS", "7")))
LOGIN_MAX_ATTEMPTS = max(1, int(os.getenv("ZHIGE_LOGIN_MAX_ATTEMPTS", "5")))
LOGIN_LOCKOUT_MINUTES = max(1, int(os.getenv("ZHIGE_LOGIN_LOCKOUT_MINUTES", "15")))


def auth_mode() -> str:
    return "team" if os.getenv("ZHIGE_AUTH_MODE", "local").strip().lower() == "team" else "local"


def team_mode() -> bool:
    return auth_mode() == "team"


def normalize_role(role: Optional[str], *, is_admin: bool = False) -> str:
    value = (role or "").strip().lower()
    if is_admin:
        return "admin"
    return value if value in ROLE_LEVEL else "viewer"


def effective_role(user: User) -> str:
    """本地模式永远具备管理员能力；团队模式尊重持久化角色。"""
    if not team_mode():
        return "admin"
    return normalize_role(getattr(user, "role", None), is_admin=bool(user.is_admin))


def has_role(user: User, minimum: str) -> bool:
    return ROLE_LEVEL[effective_role(user)] >= ROLE_LEVEL[minimum]


def hash_password(password: str) -> str:
    if len(password or "") < 8:
        raise ValueError("密码至少需要 8 个字符")
    if len(password.encode("utf-8")) > 1024:
        raise ValueError("密码过长")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: Optional[str]) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = str(encoded or "").split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            (password or "").encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(actual.hex(), digest_hex)
    except (TypeError, ValueError, OverflowError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_auth_session(
    db: Session,
    user: User,
    *,
    user_agent: Optional[str] = None,
) -> tuple[str, AuthSession]:
    raw = secrets.token_urlsafe(32)
    row = AuthSession(
        user_id=user.id,
        token_hash=_token_hash(raw),
        user_agent=(user_agent or "")[:255] or None,
        expires_at=datetime.utcnow() + timedelta(days=SESSION_DAYS),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return raw, row


def revoke_session(db: Session, raw_token: Optional[str]) -> int:
    if not raw_token:
        return 0
    now = datetime.utcnow()
    count = (
        db.query(AuthSession)
        .filter(AuthSession.token_hash == _token_hash(raw_token), AuthSession.revoked_at.is_(None))
        .update({AuthSession.revoked_at: now}, synchronize_session=False)
    )
    db.commit()
    return int(count)


def revoke_user_sessions(db: Session, user_id: str) -> int:
    """吊销一个用户的全部活动会话，用于停用账号或更换口令。"""
    now = datetime.utcnow()
    count = (
        db.query(AuthSession)
        .filter(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
        .update({AuthSession.revoked_at: now}, synchronize_session=False)
    )
    db.commit()
    return int(count)


def user_from_token(db: Session, raw_token: Optional[str]) -> Optional[User]:
    if not raw_token:
        return None
    now = datetime.utcnow()
    session = (
        db.query(AuthSession)
        .filter(
            AuthSession.token_hash == _token_hash(raw_token),
            AuthSession.revoked_at.is_(None),
            AuthSession.expires_at > now,
        )
        .first()
    )
    if session is None:
        return None
    user = (
        db.query(User)
        .filter(User.id == session.user_id, User.is_active.is_(True))
        .first()
    )
    if user is None:
        return None
    session.last_seen_at = now
    db.commit()
    return user


def request_token(request: Request) -> Optional[str]:
    authorization = request.headers.get("Authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return request.cookies.get(SESSION_COOKIE)


def find_active_user(db: Session, login: str) -> Optional[User]:
    """按用户名或邮箱查找启用中的账号（不校验口令）。"""
    login = (login or "").strip()
    if not login:
        return None
    return (
        db.query(User)
        .filter(
            (User.username == login) | (User.email == login),
            User.is_active.is_(True),
        )
        .first()
    )


def account_locked(user: User) -> bool:
    """账号是否处于登录失败锁定期内。"""
    locked_until = getattr(user, "locked_until", None)
    if locked_until is None:
        return False
    if locked_until.tzinfo is not None:
        locked_until = locked_until.replace(tzinfo=None)
    return datetime.utcnow() < locked_until


def record_login_failure(db: Session, user: User) -> None:
    """记录一次失败登录；达到阈值后锁定账号一段时间。"""
    attempts = (getattr(user, "failed_login_attempts", 0) or 0) + 1
    user.failed_login_attempts = attempts
    if attempts >= LOGIN_MAX_ATTEMPTS:
        user.locked_until = datetime.utcnow() + timedelta(minutes=LOGIN_LOCKOUT_MINUTES)
        user.failed_login_attempts = 0
        logger.warning(
            "账号 %s 连续登录失败 %d 次，已锁定 %d 分钟",
            user.username, LOGIN_MAX_ATTEMPTS, LOGIN_LOCKOUT_MINUTES,
        )
    db.commit()


def record_login_success(db: Session, user: User) -> None:
    """登录成功后清空失败计数与锁定状态。"""
    user.failed_login_attempts = 0
    user.locked_until = None
    db.commit()


def reset_login_lockout(user: User) -> None:
    """管理员重置口令或重新启用账号时清空锁定状态。"""
    user.failed_login_attempts = 0
    user.locked_until = None


def authenticate_credentials(db: Session, login: str, password: str) -> Optional[User]:
    login = (login or "").strip()
    if not login or not password:
        return None
    user = find_active_user(db, login)
    if user is None:
        return None
    if account_locked(user):
        return None
    if not verify_password(password, user.password_hash):
        record_login_failure(db, user)
        return None
    record_login_success(db, user)
    return user


def websocket_user(websocket: WebSocket, db: Session) -> User:
    if not team_mode():
        from src.middleware.identity import get_local_user

        return get_local_user(db)
    raw = websocket.cookies.get(SESSION_COOKIE)
    authorization = websocket.headers.get("authorization", "")
    if not raw and authorization.lower().startswith("bearer "):
        raw = authorization[7:].strip()
    user = user_from_token(db, raw)
    if user is None:
        raise PermissionError("团队模式需要登录")
    return user


def role_dependency(minimum: str):
    from src.middleware.identity import get_current_user

    async def _dependency(user: User = Depends(get_current_user)) -> User:
        if not has_role(user, minimum):
            raise HTTPException(status_code=403, detail=f"需要 {minimum} 权限")
        return user

    return _dependency


require_editor = role_dependency("editor")
require_admin_user = role_dependency("admin")


def user_dict(user: User, *, include_role: bool = True) -> dict:
    payload = {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "department": user.department or "",
        "is_admin": bool(user.is_admin),
        "is_active": bool(user.is_active),
    }
    if include_role:
        payload["role"] = effective_role(user)
    return payload


def can_manage_roles(actor: User, roles: Iterable[str]) -> bool:
    return effective_role(actor) == "admin" and all(role in ROLE_LEVEL for role in roles)
