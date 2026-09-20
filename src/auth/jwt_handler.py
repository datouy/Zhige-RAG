"""JWT Token 处理工具。

提供 Access Token（30分钟有效期）和 Refresh Token（7天有效期）的创建与验证。
"""
from __future__ import annotations
import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional
import jwt

_DEFAULT_SECRET = "change-me-in-production-minimum-256-bits-random-string"
SECRET_KEY = os.getenv("JWT_SECRET", _DEFAULT_SECRET)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

if SECRET_KEY == _DEFAULT_SECRET:
    # 使用默认弱密钥时任何人都能伪造 token。开发环境允许启动，但必须醒目提示；
    # 生产环境（显式要求强密钥）可通过 REQUIRE_STRONG_JWT=1 直接拒绝启动。
    logging.getLogger(__name__).warning(
        "JWT_SECRET 未设置，正在使用不安全的默认密钥——仅可用于本地开发！"
        "请通过环境变量 JWT_SECRET 设置随机密钥，例如：%s",
        secrets.token_urlsafe(32),
    )
    if os.getenv("REQUIRE_STRONG_JWT", "").lower() in ("1", "true", "yes"):
        raise RuntimeError("生产环境必须设置 JWT_SECRET 环境变量（REQUIRE_STRONG_JWT=1）")


def create_access_token(user_id: str) -> str:
    """创建 Access Token（30分钟有效期）。"""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """创建 Refresh Token（7天有效期）。"""
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """解码 Token，返回 payload 或 None（过期/无效）。"""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
