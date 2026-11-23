"""模型凭据管理。

支持两种企业部署方式：

- ``env``：凭据只来自环境变量 / K8s Secret / Vault 注入，接口拒绝落盘；
- ``file``：用 Fernet 加密写入本地文件，主密钥仍由环境变量提供；
- ``auto``：环境变量优先；未提供主密钥时退化为只读 env。

无论哪种模式，``config/local_overrides.yaml`` 都不允许出现明文 ``api_key``。
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import get_logger, resolve_path

logger = get_logger("secrets_store")

SECRET_ENV_ALIASES = {
    "llm.openai.api_key": ("OPENAI_API_KEY",),
    "embedding.openai.api_key": ("EMBEDDING_OPENAI_API_KEY", "OPENAI_API_KEY"),
    "llm.ollama.api_key": ("OLLAMA_API_KEY",),
    "embedding.ollama.api_key": ("EMBEDDING_OLLAMA_API_KEY",),
    "reranker.api_key": ("RERANKER_API_KEY",),
}


class SecretStoreError(RuntimeError):
    """凭据存储配置或写入失败。"""


def secret_backend() -> str:
    mode = str(os.getenv("SECRETS_BACKEND", "auto") or "auto").strip().lower()
    if mode not in {"auto", "env", "file"}:
        raise SecretStoreError("SECRETS_BACKEND 仅支持 auto / env / file")
    return mode


def secrets_path() -> Path:
    return resolve_path(os.getenv("SECRETS_FILE", "data/secrets.enc"))


def _fernet_key() -> Optional[bytes]:
    raw = str(os.getenv("ZHIGE_SECRETS_KEY", "") or "").strip()
    if raw:
        try:
            decoded = base64.urlsafe_b64decode(raw.encode("ascii"))
            if len(decoded) == 32:
                return raw.encode("ascii")
        except Exception:
            pass
        raise SecretStoreError("ZHIGE_SECRETS_KEY 必须是 Fernet 的 32 字节 base64url 密钥")

    passphrase = str(os.getenv("ZHIGE_SECRETS_PASSPHRASE", "") or "")
    if not passphrase:
        return None
    salt = str(os.getenv("ZHIGE_SECRETS_SALT", "zhige-local-secret-store-v1") or "").encode(
        "utf-8"
    )
    digest = hashlib.scrypt(
        passphrase.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32
    )
    return base64.urlsafe_b64encode(digest)


def _fernet():
    try:
        from cryptography.fernet import Fernet
    except Exception as exc:  # noqa: BLE001
        raise SecretStoreError("file 凭据后端需要安装 cryptography") from exc
    key = _fernet_key()
    if not key:
        raise SecretStoreError(
            "file 凭据后端需要 ZHIGE_SECRETS_KEY 或 ZHIGE_SECRETS_PASSPHRASE"
        )
    return Fernet(key)


def _env_secret(name: str) -> Optional[str]:
    for env_name in SECRET_ENV_ALIASES.get(name, ()):
        value = os.getenv(env_name)
        if value:
            return str(value)
    return None


def _read_file() -> Dict[str, str]:
    path = secrets_path()
    if not path.exists():
        return {}
    try:
        token = path.read_bytes()
        plain = _fernet().decrypt(token)
        data = json.loads(plain.decode("utf-8"))
        return {str(k): str(v) for k, v in (data or {}).items()}
    except SecretStoreError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SecretStoreError("加密凭据文件损坏或主密钥不匹配") from exc


def _write_file(values: Dict[str, str]) -> None:
    path = secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _fernet().encrypt(
        json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


def get_secret(name: str) -> Optional[str]:
    """环境变量优先；file 后端再读取加密文件。"""
    env_value = _env_secret(name)
    if env_value:
        return env_value
    mode = secret_backend()
    if mode == "env":
        return None
    if mode == "file" or (mode == "auto" and _fernet_key()):
        try:
            return _read_file().get(name)
        except SecretStoreError:
            logger.error("读取加密凭据失败", exc_info=True)
            return None
    return None


def set_secret(name: str, value: str) -> bool:
    """写入加密凭据；env 后端拒绝写入，返回 False。"""
    if not value:
        return False
    if _env_secret(name):
        raise SecretStoreError(
            f"{name} 已由环境变量/Secret Manager 提供，不能再写入本地凭据文件"
        )
    mode = secret_backend()
    if mode == "env":
        return False
    if mode == "auto" and not _fernet_key():
        return False
    values = _read_file()
    values[name] = str(value)
    _write_file(values)
    return True


def delete_secret(name: str) -> bool:
    mode = secret_backend()
    if mode == "env":
        return False
    if mode == "auto" and not _fernet_key():
        return False
    values = _read_file()
    existed = name in values
    values.pop(name, None)
    _write_file(values)
    return existed


def has_secret(name: str) -> bool:
    return bool(get_secret(name))


def hydrate_config_secrets(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把可管理凭据注入运行时配置；不修改原对象。"""
    out = copy.deepcopy(cfg or {})
    targets = [
        ("llm", "openai"),
        ("llm", "ollama"),
        ("embedding", "openai"),
        ("embedding", "ollama"),
    ]
    for section_name, backend in targets:
        section = ((out.get(section_name) or {}).get(backend) or {})
        secret_name = f"{section_name}.{backend}.api_key"
        value = get_secret(secret_name)
        if value:
            section["api_key"] = value
    reranker = out.get("reranker")
    if isinstance(reranker, dict):
        value = get_secret("reranker.api_key")
        if value:
            reranker["api_key"] = value
    return out


def strip_plaintext_secrets(value: Any, path: tuple[str, ...] = ()) -> tuple[Any, Dict[str, str]]:
    """递归移除明文 api_key，并返回学到的凭据映射。"""
    found: Dict[str, str] = {}
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            current_path = path + (str(key),)
            if key == "api_key" and isinstance(item, str) and item.strip():
                # 只有 llm/embedding/reranker 这种可映射路径才保留
                name = ".".join(current_path)
                if name in SECRET_ENV_ALIASES or name == "reranker.api_key":
                    found[name] = item.strip()
                continue
            out[key], nested = strip_plaintext_secrets(item, current_path)
            found.update(nested)
        return out, found
    if isinstance(value, list):
        items = []
        for item in value:
            cleaned, nested = strip_plaintext_secrets(item, path)
            items.append(cleaned)
            found.update(nested)
        return items, found
    return value, found


__all__ = [
    "SECRET_ENV_ALIASES",
    "SecretStoreError",
    "delete_secret",
    "get_secret",
    "has_secret",
    "hydrate_config_secrets",
    "secret_backend",
    "set_secret",
    "strip_plaintext_secrets",
]
