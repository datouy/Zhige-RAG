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

# 兼容两种环境变量名：``JWT_SECRET`` 是规范名（.env.example / k8s Secret /
# 文档都用它）；``SECRET_KEY`` 是历史别名。docker-compose 早期只传 SECRET_KEY，
# 而代码只读 JWT_SECRET —— 两边对不上，生产环境会静默回落到下面的默认弱密钥，
# 任何人都能离线伪造任意用户的 token 接管账户。这里保留别名并持续告警，
# 避免已部署环境因改名立刻启动失败，同时提示迁移。
_ALIAS_ENV_USED = not os.getenv("JWT_SECRET") and bool(os.getenv("SECRET_KEY"))
SECRET_KEY = os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY") or _DEFAULT_SECRET
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

_log = logging.getLogger(__name__)

_REQUIRE_STRONG = os.getenv("REQUIRE_STRONG_JWT", "").lower() in ("1", "true", "yes")
# 部署模板里的占位符长度往往能骗过"≥32 字符"检查（如 k8s/secret.yaml），
# 因此单独识别。
_SECRET_IS_PLACEHOLDER = (
    "CHANGE_ME" in SECRET_KEY.upper() or "CHANGE-ME" in SECRET_KEY.upper()
)

if SECRET_KEY == _DEFAULT_SECRET:
    # 用默认弱密钥时任何人都能伪造 token。开发环境允许启动但必须醒目提示；
    # 生产环境（显式要求强密钥）直接拒绝启动。
    _log.warning(
        "JWT_SECRET 未设置，正在使用不安全的默认密钥——仅可用于本地开发！"
        "请通过环境变量 JWT_SECRET 设置随机密钥，例如：%s",
        secrets.token_urlsafe(32),
    )
elif _SECRET_IS_PLACEHOLDER:
    _log.warning(
        "JWT_SECRET 看起来仍是部署模板中的占位符（含 CHANGE_ME），请替换为随机密钥。"
        '生成方式：python -c "import secrets; print(secrets.token_urlsafe(32))"'
    )
elif len(SECRET_KEY) < 32:
    _log.warning("JWT_SECRET 长度不足 32 字符（当前 %d），存在被暴力破解的风险。", len(SECRET_KEY))

if _ALIAS_ENV_USED:
    _log.warning("检测到环境变量 SECRET_KEY，请迁移到规范的 JWT_SECRET（当前两者均可识别）。")

if _REQUIRE_STRONG and (
    SECRET_KEY == _DEFAULT_SECRET or _SECRET_IS_PLACEHOLDER or len(SECRET_KEY) < 32
):
    raise RuntimeError(
        "生产环境必须设置强 JWT_SECRET（至少 32 字符的随机串，且不能是部署模板占位符）。"
        "已设置 REQUIRE_STRONG_JWT=1，拒绝启动。"
    )


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
