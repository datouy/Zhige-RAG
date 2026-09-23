"""认证路由：注册、登录、Token 刷新、获取用户信息。"""
from __future__ import annotations
import hashlib
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.db.database import get_db
from src.db.models import User, RefreshToken
from src.auth.jwt_handler import (
    create_access_token,
    create_refresh_token,
    decode_token,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
from src.auth.password import hash_password, verify_password
from src.middleware.auth import get_current_user
from src.middleware.audit import get_audit_logger, AuditAction

router = APIRouter(prefix="/api/v1/auth", tags=["认证"])

# Rate limiter instance
limiter = Limiter(key_func=get_remote_address)


def _get_client_ip(req: Request) -> str:
    """获取客户端 IP 地址。"""
    forwarded = req.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = req.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    if req.client:
        return req.client.host
    return "unknown"


class RegisterRequest(BaseModel):
    """注册请求。"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱")
    password: str = Field(..., min_length=8, max_length=128, description="密码")


class LoginRequest(BaseModel):
    """登录请求。"""
    login: str = Field(..., description="用户名或邮箱")
    password: str = Field(..., description="密码")


class RefreshRequest(BaseModel):
    """刷新 Token 请求。"""
    refresh_token: str = Field(..., description="Refresh Token")


class TokenResponse(BaseModel):
    """Token 响应。"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800


class UserResponse(BaseModel):
    """用户信息响应。"""
    id: str
    username: str
    email: str
    subscription_tier: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("3/minute")
def register(request: Request, req_data: RegisterRequest, db: Session = Depends(get_db)) -> UserResponse:
    """注册新用户（限流：每分钟 3 次）。"""
    # 检查用户名是否存在
    existing = db.query(User).filter(User.username == req_data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="用户名已存在")

    # 检查邮箱是否存在
    existing_email = db.query(User).filter(User.email == req_data.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="邮箱已被注册")

    # 创建用户
    user = User(
        username=req_data.username,
        email=req_data.email,
        password_hash=hash_password(req_data.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # 审计日志
    audit = get_audit_logger()
    audit.log_auth(
        user_id=user.id,
        action=AuditAction.REGISTER,
        ip=_get_client_ip(request),
        success=True,
        db=db,
    )

    return user


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(request: Request, req_data: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """用户登录，返回 Access Token 和 Refresh Token（限流：每分钟 5 次）。"""
    # 支持用户名或邮箱登录
    user = db.query(User).filter(
        (User.username == req_data.login) | (User.email == req_data.login)
    ).first()

    if not user or not verify_password(req_data.password, user.password_hash):
        # 审计日志 - 登录失败
        audit = get_audit_logger()
        audit.log_auth(
            user_id=None,
            action=AuditAction.LOGIN_FAILED,
            ip=_get_client_ip(request),
            success=False,
            error_message="用户名或密码错误",
            db=db,
        )
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if not user.is_active:
        # 审计日志 - 账户被禁用
        audit = get_audit_logger()
        audit.log_auth(
            user_id=user.id,
            action=AuditAction.LOGIN_FAILED,
            ip=_get_client_ip(request),
            success=False,
            error_message="账户已被禁用",
            db=db,
        )
        raise HTTPException(status_code=403, detail="账户已被禁用")

    # 生成 Token
    access_token = create_access_token(user.id)
    refresh_token_str = create_refresh_token(user.id)

    # 存储 refresh token hash（用于吊销）
    token_record = RefreshToken(
        user_id=user.id,
        token_hash=hashlib.sha256(refresh_token_str.encode()).hexdigest(),
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db.add(token_record)
    db.commit()

    # 审计日志 - 登录成功
    audit = get_audit_logger()
    audit.log_auth(
        user_id=user.id,
        action=AuditAction.LOGIN_SUCCESS,
        ip=_get_client_ip(request),
        success=True,
        db=db,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token_str,
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(req: RefreshRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """使用 Refresh Token 刷新 Access Token。

    安全（P5 审计修复）：除 JWT 签名校验外，必须查库确认该 token 存在、
    未吊销且未过期，否则一律 401。之前的实现从不校验吊销状态——logout
    之后泄露的 refresh token 仍能无限换新 token，吊销机制形同虚设。
    """
    payload = decode_token(req.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="无效的 Refresh Token")

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="用户不存在或已禁用")

    # 查库校验：token 必须存在、未吊销、未过期
    token_hash = hashlib.sha256(req.refresh_token.encode()).hexdigest()
    record = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash,
        RefreshToken.revoked == False,  # noqa: E712
    ).first()
    if record is None:
        # 从未签发过或已被吊销（logout / 轮换）→ 拒绝
        raise HTTPException(status_code=401, detail="Refresh Token 已失效，请重新登录")
    if record.expires_at and record.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Refresh Token 已过期，请重新登录")

    # 轮换：吊销旧 token，签发新 token
    record.revoked = True

    # 生成新 Token
    access_token = create_access_token(user.id)
    new_refresh = create_refresh_token(user.id)
    new_record = RefreshToken(
        user_id=user.id,
        token_hash=hashlib.sha256(new_refresh.encode()).hexdigest(),
        expires_at=datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(new_record)
    db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh,
    )


@router.get("/status")
def auth_status() -> dict:
    """告知前端当前是否需要登录（**无需认证**即可访问）。

    本地单用户模式下返回 ``auth_required=false``，前端直接进入主界面 ——
    这就是"拉起来就能用"的关键一环。
    """
    from src.middleware.auth import auth_enabled

    required = auth_enabled()
    return {
        "auth_required": required,
        "mode": "multi_user" if required else "local_single_user",
        "hint": "已启用多用户认证，请登录" if required else "本地模式：无需登录，数据保存在本机",
    }


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """获取当前登录用户信息。"""
    return current_user


@router.post("/logout")
def logout(
    req: Request,
    req_data: RefreshRequest,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> dict:
    """注销，吊销 Refresh Token。"""
    # 解析 token 获取用户 ID
    payload = decode_token(req_data.refresh_token)
    user_id = payload.get("sub") if payload else None

    token_hash = hashlib.sha256(req_data.refresh_token.encode()).hexdigest()
    token = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash,
        RefreshToken.revoked == False,
    ).first()
    if token:
        token.revoked = True
        db.commit()

    # 审计日志
    audit = get_audit_logger()
    audit.log_auth(
        user_id=user_id,
        action=AuditAction.LOGOUT,
        ip=_get_client_ip(req),
        success=True,
        db=db,
    )

    return {"message": "已成功注销"}


@router.delete("/me")
def delete_me(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """注销（删除）当前账号：停用用户并清理其全部租户数据。

    P5 审计修复的落地：之前 ``TenantAwareFactory.cleanup_user`` 是死代码，
    用户删除流程根本不存在。本端点停用账号（保留审计记录）并删除该用户的
    Chroma collection、KG 数据与内存缓存。
    """
    from api.deps import get_runtime_config
    from src.factories import TenantAwareFactory

    current_user.is_active = False
    db.commit()

    cleanup_ok = True
    try:
        TenantAwareFactory.cleanup_user(current_user.id, get_runtime_config())
    except Exception as exc:  # noqa: BLE001
        cleanup_ok = False
        get_audit_logger().log_auth(
            user_id=current_user.id,
            action=AuditAction.LOGOUT,
            ip=_get_client_ip(request),
            success=False,
            error_message=f"cleanup_user 失败: {exc}",
            db=db,
        )

    # 同步清理 Agent 路由中按用户缓存的 tools/agent（其内部持有已删除的
    # 租户向量库引用）
    try:
        from api.routes.agent import reset_agent_cache

        reset_agent_cache(current_user.id)
    except Exception:  # noqa: BLE001
        pass

    return {
        "message": "账号已注销",
        "user_id": current_user.id,
        "data_cleaned": cleanup_ok,
    }
