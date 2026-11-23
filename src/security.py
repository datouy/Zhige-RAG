"""企业模式身份、密码和可撤销会话的安全原语。

设计目标：

- 认证不引入新的 Web 框架依赖，密码哈希使用标准库 scrypt；
- 浏览器使用 HttpOnly 会话 Cookie，API 客户端可使用 Bearer Token；
- Token 数据库只存摘要，支持登出、停用账号和强制过期；
- 本地无登录模式只能由回环地址使用，企业部署默认强制认证。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from sqlalchemy.orm import Session

from src.db.models import AuthCredential, AuthSession, User
from src.utils import get_logger

logger = get_logger("security")

AUTH_SESSION_COOKIE = "zhige_session"
AUTH_MODES = {"local", "enterprise"}
USER_ROLES = {"member", "kb_admin", "admin", "auditor"}
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_LEN = 32


class AuthError(RuntimeError):
    """认证失败或认证配置不安全。"""


def _now() -> datetime:
    return datetime.utcnow()


def _security_config() -> Dict[str, Any]:
    """读取 security 配置；环境变量优先，避免配置文件里的弱默认值生效。"""
    cfg: Dict[str, Any] = {}
    try:
        # 延迟导入，避免 api.deps → security → api.deps 的模块初始化环。
        from api.deps import get_runtime_config

        cfg = dict((get_runtime_config().get("security") or {}))
    except Exception:
        cfg = {}

    env_map = {
        "AUTH_MODE": "mode",
        "AUTH_SESSION_TTL_MINUTES": "session_ttl_minutes",
        "AUTH_MAX_FAILED_ATTEMPTS": "max_failed_attempts",
        "AUTH_LOCK_MINUTES": "lock_minutes",
        "AUTH_COOKIE_SECURE": "cookie_secure",
        "AUTH_ALLOW_TESTCLIENT": "allow_testclient",
    }
    for env_name, key in env_map.items():
        value = os.getenv(env_name)
        if value is not None and value != "":
            cfg[key] = value
    return cfg


def auth_mode() -> str:
    mode = str(_security_config().get("mode") or "enterprise").strip().lower()
    if mode not in AUTH_MODES:
        raise AuthError(f"不支持的 AUTH_MODE={mode!r}，仅允许 local / enterprise")
    return mode


def is_local_auth() -> bool:
    return auth_mode() == "local"


def effective_role(user: User) -> str:
    role = str(getattr(user, "role", "") or "").strip().lower()
    if role in USER_ROLES:
        return role
    return "admin" if bool(getattr(user, "is_admin", False)) else "member"


def is_admin(user: User) -> bool:
    return effective_role(user) == "admin"


def is_loopback_host(host: Optional[str]) -> bool:
    value = str(host or "").strip().lower()
    if value.startswith("::ffff:"):
        value = value.split("::ffff:", 1)[1]
    return value in {"127.0.0.1", "::1", "localhost"}


def password_policy_errors(password: str, username: str = "") -> list[str]:
    errors: list[str] = []
    value = str(password or "")
    if len(value) < 12:
        errors.append("密码至少需要 12 个字符")
    if len(value) > 256:
        errors.append("密码不能超过 256 个字符")
    categories = sum(
        [
            any(ch.islower() for ch in value),
            any(ch.isupper() for ch in value),
            any(ch.isdigit() for ch in value),
            any(not ch.isalnum() for ch in value),
        ]
    )
    if value and categories < 3:
        errors.append("密码需至少包含大小写字母、数字、符号中的三类")
    if username and value.lower() == username.lower():
        errors.append("密码不能与用户名相同")
    return errors


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_LEN,
    )
    return "scrypt${}${}${}${}${}".format(
        _SCRYPT_N,
        _SCRYPT_R,
        _SCRYPT_P,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n_s, r_s, p_s, salt_s, digest_s = str(stored).split("$", 5)
        if scheme != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_s.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_s.encode("ascii"))
        actual = hashlib.scrypt(
            str(password).encode("utf-8"),
            salt=salt,
            n=int(n_s),
            r=int(r_s),
            p=int(p_s),
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def hash_token(token: str) -> str:
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def generate_session_token() -> str:
    return "zhg_" + secrets.token_urlsafe(40)


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    value = str(authorization or "").strip()
    if not value:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _int_cfg(key: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(_security_config().get(key, default)))
    except (TypeError, ValueError):
        return default


def session_ttl_minutes() -> int:
    return _int_cfg("session_ttl_minutes", 480, 5)


def max_failed_attempts() -> int:
    return _int_cfg("max_failed_attempts", 5, 1)


def lock_minutes() -> int:
    return _int_cfg("lock_minutes", 15, 1)


def cookie_secure() -> bool:
    raw = _security_config().get("cookie_secure", False)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def authenticate(db: Session, username: str, password: str) -> User:
    normalized = str(username or "").strip()
    if not normalized:
        raise AuthError("用户名或密码错误")
    user = (
        db.query(User)
        .filter((User.username == normalized) | (User.email == normalized))
        .first()
    )
    credential = db.get(AuthCredential, user.id) if user is not None else None

    # 对不存在的用户也执行一次固定成本哈希，降低用户名枚举的时序差异。
    if user is None or credential is None:
        verify_password(password, hash_password("invalid-placeholder"))
        raise AuthError("用户名或密码错误")
    if not bool(user.is_active):
        raise AuthError("账号已停用")
    now = _now()
    if credential.locked_until and credential.locked_until > now:
        raise AuthError("登录失败次数过多，请稍后再试")

    if not verify_password(password, credential.password_hash):
        credential.failed_attempts = int(credential.failed_attempts or 0) + 1
        if credential.failed_attempts >= max_failed_attempts():
            credential.locked_until = now + timedelta(minutes=lock_minutes())
            credential.failed_attempts = 0
        db.commit()
        raise AuthError("用户名或密码错误")

    credential.failed_attempts = 0
    credential.locked_until = None
    db.commit()
    db.refresh(user)
    return user


def set_password(
    db: Session,
    user: User,
    password: str,
    *,
    reset_failures: bool = True,
) -> AuthCredential:
    errors = password_policy_errors(password, user.username)
    if errors:
        raise AuthError("；".join(errors))
    credential = db.get(AuthCredential, user.id)
    if credential is None:
        credential = AuthCredential(
            user_id=user.id, password_hash=hash_password(password)
        )
        db.add(credential)
    else:
        credential.password_hash = hash_password(password)
        credential.password_changed_at = _now()
        if reset_failures:
            credential.failed_attempts = 0
            credential.locked_until = None
    db.commit()
    db.refresh(credential)
    return credential


def create_user(
    db: Session,
    *,
    username: str,
    email: str,
    password: str,
    role: str = "member",
    department: str = "",
) -> User:
    role = str(role or "member").strip().lower()
    if role not in USER_ROLES:
        raise AuthError(f"未知角色：{role}")
    if db.query(User).filter(User.username == username).first() is not None:
        raise AuthError("用户名已存在")
    if email and db.query(User).filter(User.email == email).first() is not None:
        raise AuthError("邮箱已存在")
    user = User(
        username=username,
        email=email,
        department=department,
        role=role,
        is_admin=(role == "admin"),
        is_active=True,
    )
    db.add(user)
    db.flush()
    try:
        set_password(db, user, password)
    except Exception:
        db.rollback()
        raise
    db.refresh(user)
    return user


def create_session(
    db: Session,
    user: User,
    *,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Tuple[str, AuthSession]:
    if not bool(user.is_active):
        raise AuthError("账号已停用")
    token = generate_session_token()
    now = _now()
    session = AuthSession(
        user_id=user.id,
        token_hash=hash_token(token),
        expires_at=now + timedelta(minutes=session_ttl_minutes()),
        ip_address=str(ip_address or "")[:128] or None,
        user_agent=str(user_agent or "")[:512] or None,
        created_at=now,
        last_seen_at=now,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return token, session


def resolve_session(db: Session, token: str) -> Tuple[User, AuthSession]:
    session = (
        db.query(AuthSession)
        .filter(AuthSession.token_hash == hash_token(token))
        .first()
    )
    if session is None or bool(session.revoked):
        raise AuthError("登录状态无效")
    if session.expires_at <= _now():
        session.revoked = True
        db.commit()
        raise AuthError("登录已过期")
    user = db.get(User, session.user_id)
    if user is None or not bool(user.is_active):
        session.revoked = True
        db.commit()
        raise AuthError("账号已停用")
    session.last_seen_at = _now()
    db.commit()
    return user, session


def revoke_session(db: Session, session: AuthSession) -> None:
    session.revoked = True
    db.commit()


def revoke_user_sessions(db: Session, user_id: str) -> int:
    count = (
        db.query(AuthSession)
        .filter(AuthSession.user_id == user_id, AuthSession.revoked.is_(False))
        .update({AuthSession.revoked: True}, synchronize_session=False)
    )
    db.commit()
    return int(count or 0)


def _bootstrap_password() -> str:
    password = str(os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "") or "")
    password_file = str(os.getenv("BOOTSTRAP_ADMIN_PASSWORD_FILE", "") or "")
    if password_file:
        password = Path(password_file).read_text(encoding="utf-8").strip()
    return password


def ensure_bootstrap_admin(db: Session) -> Optional[User]:
    """企业模式首次启动创建管理员；不会使用默认弱口令。"""
    if not is_local_auth():
        existing = (
            db.query(User)
            .join(AuthCredential, AuthCredential.user_id == User.id)
            .filter(User.role == "admin", User.is_active.is_(True))
            .first()
        )
        if existing is not None:
            return existing

        username = str(os.getenv("BOOTSTRAP_ADMIN_USERNAME", "admin") or "admin").strip()
        password = _bootstrap_password()
        if not password:
            raise AuthError(
                "企业模式没有可用管理员。请设置 BOOTSTRAP_ADMIN_PASSWORD 或 "
                "BOOTSTRAP_ADMIN_PASSWORD_FILE 后重启；拒绝使用默认弱口令。"
            )
        errors = password_policy_errors(password, username)
        if errors:
            raise AuthError("管理员初始密码不符合策略：" + "；".join(errors))

        user = db.query(User).filter(User.username == username).first()
        if user is None:
            user = create_user(
                db,
                username=username,
                email=f"{username}@zhige.local",
                password=password,
                role="admin",
                department=str(os.getenv("BOOTSTRAP_ADMIN_DEPARTMENT", "") or ""),
            )
        else:
            user.role = "admin"
            user.is_admin = True
            user.is_active = True
            db.commit()
            set_password(db, user, password)
        logger.warning(
            "已创建企业模式初始管理员 %s。首次登录后请立即修改密码并删除初始密码环境变量。",
            username,
        )
        return user
    return None


def role_allowed(role: str, allowed: Iterable[str]) -> bool:
    role = str(role or "member").strip().lower()
    return role == "admin" or role in {str(item).lower() for item in allowed}


__all__ = [
    "AUTH_SESSION_COOKIE",
    "AuthError",
    "authenticate",
    "auth_mode",
    "cookie_secure",
    "create_session",
    "create_user",
    "effective_role",
    "ensure_bootstrap_admin",
    "extract_bearer_token",
    "generate_session_token",
    "hash_password",
    "hash_token",
    "is_admin",
    "is_local_auth",
    "is_loopback_host",
    "password_policy_errors",
    "resolve_session",
    "revoke_session",
    "revoke_user_sessions",
    "role_allowed",
    "session_ttl_minutes",
    "set_password",
    "verify_password",
]
