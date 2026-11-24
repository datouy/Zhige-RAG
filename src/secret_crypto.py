"""密钥加密存储（Fernet）。

用于把 ``config/local_overrides.yaml`` 里的 API Key 以密文落盘，避免明文
密钥进入普通配置文件、Git 或日志。设计约束：

- 密钥来源优先级：环境变量 ``ZHIGE_SECRETS_KEY``（urlsafe base64 的 32 字节）
  > 本地密钥文件 ``data/.secrets_key``（首次自动生成，POSIX 下收紧到 0600）。
- 未安装 ``cryptography`` 或拿不到密钥时 ``get_fernet()`` 返回 ``None``，
  调用方（``settings_service``）据此**拒绝落盘明文**，而不是退化为明文写入。
- 密文带 ``enc:v1:`` 前缀，方便与占位符 ``${...}`` 和明文区分，便于迁移。
"""
from __future__ import annotations

import base64
import os
from typing import Optional

from src.utils import get_logger, resolve_path

logger = get_logger("secret_crypto")

PREFIX = "enc:v1:"
_SECRETS_KEY_FILE = os.getenv("ZHIGE_SECRETS_KEY_FILE", "data/.secrets_key")

_fernet = None
_fernet_resolved = False


def _env_key() -> Optional[bytes]:
    raw = os.getenv("ZHIGE_SECRETS_KEY", "").strip()
    if not raw:
        return None
    try:
        decoded = base64.urlsafe_b64decode(raw.encode("ascii"))
        if len(decoded) == 32:
            return base64.urlsafe_b64encode(decoded)
        logger.warning("ZHIGE_SECRETS_KEY 长度不是 32 字节，已忽略")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ZHIGE_SECRETS_KEY 解析失败，已忽略：%s", exc)
    return None


def _file_key() -> Optional[bytes]:
    try:
        path = resolve_path(_SECRETS_KEY_FILE)
        if not path.exists():
            return None
        key = path.read_bytes().strip()
        decoded = base64.urlsafe_b64decode(key)
        if len(decoded) == 32:
            return key
    except Exception:
        return None
    return None


def _create_key_file() -> Optional[bytes]:
    try:
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        path = resolve_path(_SECRETS_KEY_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key + b"\n")
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        logger.info("已生成本地密钥文件 %s", path.name)
        return key
    except Exception as exc:  # noqa: BLE001
        logger.warning("生成本地密钥文件失败：%s", exc)
        return None


def get_fernet():
    """返回进程级单例 Fernet；不可用时返回 ``None``。"""
    global _fernet, _fernet_resolved
    if _fernet_resolved:
        return _fernet
    _fernet_resolved = True
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        logger.warning("cryptography 未安装，密钥加密不可用")
        return None
    key = _env_key() or _file_key() or _create_key_file()
    if key is None:
        return None
    try:
        _fernet = Fernet(key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fernet 初始化失败：%s", exc)
        _fernet = None
    return _fernet


def is_encrypted(value: str) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def is_placeholder(value: str) -> bool:
    return isinstance(value, str) and value.startswith("${")


def encrypt_secret(value: str) -> Optional[str]:
    """加密单个密钥；占位符 / 已加密值原样返回；不可加密时返回 ``None``。

    返回 ``None`` 表示"无法安全落盘"，调用方应丢弃该字段而不是写明文。
    """
    if not value or is_placeholder(value) or is_encrypted(value):
        return value
    fernet = get_fernet()
    if fernet is None:
        return None
    return PREFIX + fernet.encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(value: str) -> str:
    """解密密文；明文 / 占位符原样返回，解密失败时返回原值（不抛异常）。"""
    if not is_encrypted(value):
        return value
    fernet = get_fernet()
    if fernet is None:
        return value
    try:
        return fernet.decrypt(value[len(PREFIX):].encode("ascii")).decode("utf-8")
    except Exception:  # noqa: BLE001
        logger.warning("密钥解密失败，已返回原值")
        return value


__all__ = [
    "PREFIX",
    "decrypt_secret",
    "encrypt_secret",
    "get_fernet",
    "is_encrypted",
    "is_placeholder",
]
