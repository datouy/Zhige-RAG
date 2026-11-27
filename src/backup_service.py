"""知阁工作区备份与安全恢复。

备份包只包含运行数据与去掉秘密字段的本地配置覆盖。环境变量、API Key、
Cookie 与浏览器数据不会单独打包；账号口令摘要随数据库保留，用于恢复团队
账号，备份文件必须按敏感数据保管。恢复包先写入隔离暂存区，由管理员确认后
在下次服务启动、数据库初始化之前应用。
"""
from __future__ import annotations

import hashlib
import json
import os
import base64
import secrets
import shutil
import sqlite3
import stat
import subprocess
import time
import uuid
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from filelock import FileLock

from src.utils import get_logger, resolve_path

logger = get_logger("backup_service")

SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = resolve_path(os.getenv("RAG_DATA_DIR", "data"))
BACKUP_ROOT = DATA_ROOT / "backups"
RESTORE_ROOT = DATA_ROOT / ".restore_staging"
PENDING_FILE = DATA_ROOT / ".restore_pending.json"
STATUS_FILE = DATA_ROOT / ".restore_status.json"
MAX_ARCHIVE_BYTES = int(os.getenv("ZHIGE_BACKUP_MAX_MB", "2048")) * 1024 * 1024
MAX_EXTRACTED_BYTES = int(os.getenv("ZHIGE_RESTORE_MAX_MB", "4096")) * 1024 * 1024
MAX_ENTRIES = int(os.getenv("ZHIGE_RESTORE_MAX_ENTRIES", "200000"))
_ENC_MAGIC = b"ZHIGE-AESGCM-V1\n"
_ENC_CHUNK = 1024 * 1024


class BackupError(ValueError):
    """备份或恢复包校验失败，消息可以直接返回给管理员。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encryption_enabled() -> bool:
    return os.getenv("ZHIGE_BACKUP_ENCRYPTION", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "required",
    }


def _load_backup_key(*, create_if_missing: bool = False) -> bytes:
    encoded = os.getenv("ZHIGE_BACKUP_KEY", "").strip()
    if encoded:
        try:
            key = base64.urlsafe_b64decode(encoded.encode("ascii"))
        except Exception as exc:  # noqa: BLE001
            raise BackupError("ZHIGE_BACKUP_KEY 不是有效的 base64url 密钥") from exc
        if len(key) != 32:
            raise BackupError("ZHIGE_BACKUP_KEY 解码后必须为 32 字节")
        return key

    configured_file = os.getenv("ZHIGE_BACKUP_KEY_FILE", "").strip()
    key_path = resolve_path(configured_file) if configured_file else DATA_ROOT / ".backup-key"
    if key_path.is_file():
        try:
            text = key_path.read_text(encoding="ascii").strip()
            key = base64.urlsafe_b64decode(text.encode("ascii"))
        except Exception as exc:  # noqa: BLE001
            raise BackupError("备份密钥文件格式无效") from exc
        if len(key) != 32:
            raise BackupError("备份密钥必须为 32 字节")
        return key

    if not create_if_missing:
        raise BackupError("未配置备份加密密钥")
    key = secrets.token_bytes(32)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(base64.urlsafe_b64encode(key).decode("ascii"), encoding="ascii")
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    logger.warning("已生成本地备份密钥，请立即迁移到 Secret 管理器并做独立保管")
    return key


def backup_key_id() -> str:
    return hashlib.sha256(_load_backup_key(create_if_missing=True)).hexdigest()[:12]


def encrypt_file(source: Path, destination: Path, key: Optional[bytes] = None) -> Dict[str, Any]:
    """Encrypt a ZIP with chunked AES-256-GCM for bounded memory use."""
    key = key or _load_backup_key(create_if_missing=True)
    aead = AESGCM(key)
    nonce_prefix = secrets.token_bytes(8)
    partial = destination.with_name(destination.name + ".partial")
    try:
        with open(source, "rb") as src, open(partial, "wb") as dst:
            dst.write(_ENC_MAGIC)
            dst.write(nonce_prefix)
            counter = 0
            while True:
                block = src.read(_ENC_CHUNK)
                if not block:
                    break
                nonce = nonce_prefix + counter.to_bytes(4, "big")
                aad = _ENC_MAGIC + counter.to_bytes(8, "big")
                dst.write(aead.encrypt(nonce, block, aad))
                counter += 1
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {
        "encrypted": True,
        "algorithm": "AES-256-GCM",
        "key_id": hashlib.sha256(key).hexdigest()[:12],
        "chunk_bytes": _ENC_CHUNK,
    }


def decrypt_file(source: Path, destination: Path, key: Optional[bytes] = None) -> None:
    """Authenticate and decrypt a chunked backup file."""
    key = key or _load_backup_key(create_if_missing=False)
    aead = AESGCM(key)
    partial = destination.with_name(destination.name + ".partial")
    try:
        with open(source, "rb") as src:
            magic = src.read(len(_ENC_MAGIC))
            if magic != _ENC_MAGIC:
                raise BackupError("不是受支持的知阁加密备份")
            nonce_prefix = src.read(8)
            if len(nonce_prefix) != 8:
                raise BackupError("加密备份头部不完整")
            with open(partial, "wb") as dst:
                counter = 0
                while True:
                    encrypted = src.read(_ENC_CHUNK + 16)
                    if not encrypted:
                        break
                    nonce = nonce_prefix + counter.to_bytes(4, "big")
                    aad = _ENC_MAGIC + counter.to_bytes(8, "big")
                    try:
                        dst.write(aead.decrypt(nonce, encrypted, aad))
                    except Exception as exc:  # noqa: BLE001
                        raise BackupError("备份解密或完整性认证失败") from exc
                    counter += 1
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise BackupError(f"无法读取恢复状态：{exc}") from exc
    if not isinstance(payload, dict):
        raise BackupError("恢复状态格式无效")
    return payload


def _database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def _database_is_postgres() -> bool:
    return _database_url().startswith(("postgresql", "postgres"))


def _sqlite_path() -> Path:
    from src.db.database import DB_PATH

    return Path(DB_PATH).resolve()


def _snapshot_sqlite(destination: Path) -> None:
    source = _sqlite_path()
    if not source.exists():
        raise BackupError(f"数据库文件不存在：{source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Windows 的盘符路径不能可靠地放进 SQLite URI；普通连接配合只读式
    # ``backup`` API 仍是 SQLite 官方支持的一致性快照方式。
    with closing(sqlite3.connect(str(source), timeout=30)) as src:
        with closing(sqlite3.connect(destination)) as dst:
            src.backup(dst)
            dst.execute("PRAGMA wal_checkpoint(FULL)")


def _pg_connection_args() -> tuple[List[str], Dict[str, str]]:
    from sqlalchemy.engine import make_url

    try:
        url = make_url(_database_url())
    except Exception as exc:  # noqa: BLE001
        raise BackupError(f"DATABASE_URL 无效：{exc}") from exc
    args: List[str] = []
    env = dict(os.environ)
    if url.host:
        args.extend(["--host", str(url.host)])
    if url.port:
        args.extend(["--port", str(url.port)])
    if url.username:
        args.extend(["--username", str(url.username)])
    if url.database:
        args.extend(["--dbname", str(url.database)])
    if url.password:
        env["PGPASSWORD"] = str(url.password)
    return args, env


def _snapshot_postgres(destination: Path) -> None:
    executable = shutil.which("pg_dump")
    if not executable:
        raise BackupError("PostgreSQL 备份需要安装 pg_dump，并把它加入 PATH")
    args, env = _pg_connection_args()
    command = [
        executable,
        "--format=custom",
        "--no-owner",
        "--no-privileges",
        "--file",
        str(destination),
        *args,
    ]
    completed = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        timeout=int(os.getenv("ZHIGE_BACKUP_TIMEOUT_SECONDS", "1800")),
        check=False,
    )
    if completed.returncode != 0:
        destination.unlink(missing_ok=True)
        raise BackupError(f"pg_dump 失败：{(completed.stderr or completed.stdout)[-1000:]}")


def _clean_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(word in lowered for word in ("api_key", "password", "secret", "token")):
                if isinstance(item, str) and item.startswith("${"):
                    cleaned[key] = item
                continue
            cleaned[key] = _clean_secrets(item)
        return cleaned
    if isinstance(value, list):
        return [_clean_secrets(item) for item in value]
    return value


def _safe_local_overrides(destination: Path) -> bool:
    from src.settings_service import overrides_path

    source = overrides_path()
    if not source.exists():
        return False
    try:
        import yaml

        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        safe = _clean_secrets(payload)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "# 备份时已移除 API Key、密码、令牌等秘密字段\n"
            + yaml.safe_dump(safe, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("本地配置脱敏失败，已跳过：%s", exc)
        return False


def _runtime_paths() -> Dict[str, Path]:
    try:
        from api.deps import get_runtime_config

        cfg = get_runtime_config()
    except Exception:
        cfg = {}
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    vector = cfg.get("vector_store", {}) if isinstance(cfg, dict) else {}
    return {
        "raw": resolve_path(paths.get("raw_docs", "data/raw")),
        "chroma": resolve_path(
            vector.get("persist_directory")
            or paths.get("chroma_db", "data/chroma_db")
        ),
        "eval": resolve_path(paths.get("eval", "data/eval")),
        "logs": resolve_path(paths.get("logs", "logs")),
    }


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        yield path


def _write_zip_member(
    archive: zipfile.ZipFile,
    source: Path,
    arcname: str,
    *,
    timeout: Optional[float] = None,
) -> None:
    """Add a file while tolerating brief Windows sharing locks from active readers."""
    deadline = time.monotonic() + float(
        timeout if timeout is not None else os.getenv("ZHIGE_BACKUP_READ_RETRY_SECONDS", "30")
    )
    while True:
        try:
            archive.write(source, arcname)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def _add_tree(
    archive: zipfile.ZipFile,
    source: Path,
    arc_root: str,
    manifest_files: List[Dict[str, Any]],
    *,
    exclude_names: Optional[set[str]] = None,
) -> int:
    if not source.exists():
        return 0
    count = 0
    for path in _iter_files(source):
        if exclude_names and path.name in exclude_names:
            continue
        relative = path.relative_to(source).as_posix()
        arcname = f"{arc_root.rstrip('/')}/{relative}"
        _write_zip_member(archive, path, arcname)
        manifest_files.append(
            {
                "path": arcname,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        count += 1
    return count


def _add_chroma_tree(
    archive: zipfile.ZipFile,
    source: Path,
    manifest_files: List[Dict[str, Any]],
) -> int:
    """Copy the local Chroma directory while blocking application writes."""
    if not source.exists():
        return 0
    lock = FileLock(
        str(source / ".chroma-write.lock"),
        timeout=float(os.getenv("ZHIGE_BACKUP_LOCK_TIMEOUT", "120") or 120),
    )
    with lock:
        return _add_tree(
            archive,
            source,
            "data/chroma_db",
            manifest_files,
            exclude_names={".chroma-write.lock"},
        )


def _add_sqlite_snapshot(
    archive: zipfile.ZipFile,
    source: Path,
    arcname: str,
    stage: Path,
    manifest_files: List[Dict[str, Any]],
) -> bool:
    """Use SQLite's online backup API rather than copying WAL/SHM files."""
    if not source.is_file():
        return False
    destination = stage / "sqlite" / Path(arcname).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source), timeout=30) as src:
        with sqlite3.connect(destination) as dst:
            src.backup(dst)
    archive.write(destination, arcname)
    manifest_files.append(
        {
            "path": arcname,
            "size": destination.stat().st_size,
            "sha256": _sha256(destination),
        }
    )
    return True


def create_backup(*, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """创建一致性 ZIP；返回可下载路径与摘要。"""
    output_dir = (output_dir or BACKUP_ROOT).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    encrypted = _encryption_enabled()
    suffix = ".zip.enc" if encrypted else ".zip"
    output = output_dir / f"zhige-backup-{timestamp}-{uuid.uuid4().hex[:6]}{suffix}"
    plain_partial = output.with_name(
        f".{output.stem}.plain-{uuid.uuid4().hex}.zip.partial"
    )
    partial = output.with_name(output.name + ".partial")
    stage = output_dir / f".backup-stage-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    files: List[Dict[str, Any]] = []
    try:
        database_meta: Dict[str, Any]
        if _database_is_postgres():
            database_file = stage / "database" / "postgres.dump"
            database_file.parent.mkdir(parents=True, exist_ok=True)
            _snapshot_postgres(database_file)
            database_meta = {
                "kind": "postgresql",
                "path": "database/postgres.dump",
                "sha256": _sha256(database_file),
            }
        else:
            database_file = stage / "database" / "users.db"
            database_file.parent.mkdir(parents=True, exist_ok=True)
            _snapshot_sqlite(database_file)
            database_meta = {
                "kind": "sqlite",
                "path": "database/users.db",
                "source_name": _sqlite_path().name,
                "sha256": _sha256(database_file),
            }
        files.append(
            {
                "path": database_meta["path"],
                "size": database_file.stat().st_size,
                "sha256": database_meta["sha256"],
            }
        )

        overrides_file = stage / "config" / "local_overrides.yaml"
        overrides_included = _safe_local_overrides(overrides_file)

        runtime = _runtime_paths()
        included: List[str] = [
            "database",
            "data/ingest_jobs.json",
            "data/eval",
            "data/chroma_db",
            "data/raw",
            "logs/audit.jsonl",
        ]
        manifest = {
            "app": "zhige",
            "schema_version": SCHEMA_VERSION,
            "created_at": _now(),
            "database": database_meta,
            "storage": {key: str(value) for key, value in runtime.items()},
            "included": included,
            "excluded": [".env*", "API Key", "明文密码", "登录 Cookie", "浏览器配置"],
            "files": files,
        }

        with zipfile.ZipFile(plain_partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(database_file, database_meta["path"])
            _add_chroma_tree(archive, runtime["chroma"], files)
            for logical, source in (("data/raw", runtime["raw"]), ("data/eval", runtime["eval"])):
                _add_tree(archive, source, logical, files)

            jobs_file = DATA_ROOT / "ingest_jobs.json"
            if jobs_file.is_file():
                archive.write(jobs_file, "data/ingest_jobs.json")
                files.append({"path": "data/ingest_jobs.json", "size": jobs_file.stat().st_size, "sha256": _sha256(jobs_file)})

            for kg_file in sorted(DATA_ROOT.glob("kg_*.db")):
                _add_sqlite_snapshot(
                    archive,
                    kg_file,
                    f"data/{kg_file.name}",
                    stage,
                    files,
                )

            if overrides_included:
                archive.write(overrides_file, "config/local_overrides.yaml")
                files.append({"path": "config/local_overrides.yaml", "size": overrides_file.stat().st_size, "sha256": _sha256(overrides_file)})

            audit_file = runtime["logs"] / "audit.jsonl"
            if audit_file.is_file():
                archive.write(audit_file, "logs/audit.jsonl")
                files.append({"path": "logs/audit.jsonl", "size": audit_file.stat().st_size, "sha256": _sha256(audit_file)})

            manifest["files"] = files
            manifest["file_count"] = len(files)
            manifest["uncompressed_bytes"] = sum(int(item["size"]) for item in files)
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )
    except Exception:
        partial.unlink(missing_ok=True)
        plain_partial.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    size = plain_partial.stat().st_size
    if size > MAX_ARCHIVE_BYTES:
        partial.unlink(missing_ok=True)
        plain_partial.unlink(missing_ok=True)
        raise BackupError("备份包超过允许大小，请调低数据量或调整 ZHIGE_BACKUP_MAX_MB")
    encryption_meta: Dict[str, Any] = {"encrypted": False}
    try:
        if encrypted:
            encryption_meta = encrypt_file(plain_partial, partial)
            plain_partial.unlink(missing_ok=True)
            os.replace(partial, output)
        else:
            os.replace(plain_partial, output)
    except Exception:
        partial.unlink(missing_ok=True)
        plain_partial.unlink(missing_ok=True)
        raise
    size = output.stat().st_size
    logger.info("备份完成：%s（%.1f MB）", output, size / 1024 / 1024)
    return {
        "path": str(output),
        "filename": output.name,
        "size": size,
        "sha256": _sha256(output),
        "created_at": manifest["created_at"],
        "file_count": int(manifest["file_count"]),
        "encryption": encryption_meta,
    }


def _safe_member(name: str) -> bool:
    if "\x00" in name or "\\" in name:
        return False
    pure = PurePosixPath(name)
    return bool(name) and not pure.is_absolute() and ".." not in pure.parts


def _inspect_archive(archive: zipfile.ZipFile) -> Dict[str, Any]:
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES:
        raise BackupError("恢复包文件数超过安全上限")
    total_size = 0
    actual_files: Dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        if not _safe_member(info.filename):
            raise BackupError(f"恢复包包含非法路径：{info.filename}")
        mode = (info.external_attr >> 16) & 0o777777
        if stat.S_ISLNK(mode):
            raise BackupError(f"恢复包包含符号链接：{info.filename}")
        total_size += int(info.file_size)
        if total_size > MAX_EXTRACTED_BYTES:
            raise BackupError("恢复包解压后超过安全大小上限")
        if not info.is_dir():
            if info.filename in actual_files:
                raise BackupError(f"恢复包包含重复文件：{info.filename}")
            actual_files[info.filename] = info
    try:
        raw = archive.read("manifest.json")
        manifest = json.loads(raw.decode("utf-8"))
    except KeyError as exc:
        raise BackupError("恢复包缺少 manifest.json") from exc
    except Exception as exc:  # noqa: BLE001
        raise BackupError(f"manifest.json 无效：{exc}") from exc
    try:
        manifest_version = int(manifest.get("schema_version", 0))
    except (TypeError, ValueError):
        manifest_version = 0
    if manifest.get("app") != "zhige" or manifest_version != SCHEMA_VERSION:
        raise BackupError("不是受支持的知阁备份版本")

    declared_items = manifest.get("files")
    if not isinstance(declared_items, list):
        raise BackupError("manifest.json 缺少文件清单")
    declared: Dict[str, Dict[str, Any]] = {}
    for item in declared_items:
        if not isinstance(item, dict):
            raise BackupError("manifest.json 文件清单格式无效")
        name = str(item.get("path") or "")
        if not _safe_member(name) or name == "manifest.json":
            raise BackupError(f"manifest.json 包含非法文件路径：{name}")
        if name in declared:
            raise BackupError(f"manifest.json 包含重复文件：{name}")
        digest = str(item.get("sha256") or "").lower()
        if len(digest) != 64:
            raise BackupError(f"文件摘要无效：{name}")
        try:
            expected_size = int(item.get("size"))
        except (TypeError, ValueError):
            raise BackupError(f"文件大小无效：{name}") from None
        if expected_size < 0:
            raise BackupError(f"文件大小无效：{name}")
        declared[name] = {**item, "sha256": digest, "size": expected_size}

    expected_names = set(declared) | {"manifest.json"}
    actual_names = set(actual_files)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing:
        raise BackupError(f"恢复包缺少清单文件：{', '.join(missing[:5])}")
    if unexpected:
        raise BackupError(f"恢复包包含清单外文件：{', '.join(unexpected[:5])}")

    for name, item in declared.items():
        info = actual_files[name]
        if int(info.file_size) != int(item["size"]):
            raise BackupError(f"文件大小校验失败：{name}")
        digest = hashlib.sha256()
        try:
            with archive.open(info) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except Exception as exc:  # noqa: BLE001
            raise BackupError(f"恢复包文件读取失败：{name}") from exc
        if digest.hexdigest() != item["sha256"]:
            raise BackupError(f"文件完整性校验失败：{name}")

    database = manifest.get("database") or {}
    database_path = str(database.get("path") or "")
    database_item = declared.get(database_path)
    if not database_item:
        raise BackupError("恢复包数据库文件不在完整性清单中")
    if str(database.get("sha256") or "").lower() != database_item["sha256"]:
        raise BackupError("恢复包数据库摘要与文件清单不一致")
    return manifest


def pending_restore() -> Optional[Dict[str, Any]]:
    if not PENDING_FILE.exists():
        return None
    return _read_json(PENDING_FILE)


def last_restore_status() -> Optional[Dict[str, Any]]:
    if not STATUS_FILE.exists():
        return None
    try:
        return _read_json(STATUS_FILE)
    except BackupError:
        return None


def stage_restore(upload_path: Path, original_name: str = "backup.zip") -> Dict[str, Any]:
    """验证并暂存恢复包；真正应用发生在下一次服务启动。"""
    if pending_restore() is not None:
        raise BackupError("已经有一个待应用的恢复包，请先重启服务或取消恢复")
    size = upload_path.stat().st_size
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise BackupError("恢复包为空或超过大小上限")
    inspection_path = upload_path
    decrypted_path: Optional[Path] = None
    try:
        with open(upload_path, "rb") as probe:
            is_encrypted = probe.read(len(_ENC_MAGIC)) == _ENC_MAGIC
        if is_encrypted:
            decrypted_path = upload_path.with_name(
                f".{upload_path.name}.decrypted-{uuid.uuid4().hex}.zip"
            )
            decrypt_file(upload_path, decrypted_path)
            inspection_path = decrypted_path
        with zipfile.ZipFile(inspection_path) as archive:
            manifest = _inspect_archive(archive)
    except zipfile.BadZipFile as exc:
        raise BackupError("上传文件不是有效的 ZIP 恢复包") from exc

    token = uuid.uuid4().hex
    staging = RESTORE_ROOT / token
    payload = staging / "payload"
    payload.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(inspection_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    (payload / info.filename).mkdir(parents=True, exist_ok=True)
                    continue
                target = (payload / info.filename).resolve()
                if not target.is_relative_to(payload.resolve()):
                    raise BackupError(f"恢复包路径越界：{info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, open(target, "wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        pending = {
            "token": token,
            "staging": str(staging),
            "original_name": Path(original_name).name,
            "created_at": _now(),
            "manifest": manifest,
            "encrypted": bool(is_encrypted),
        }
        _write_json(PENDING_FILE, pending)
        return pending
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        upload_path.unlink(missing_ok=True)
        if decrypted_path is not None:
            decrypted_path.unlink(missing_ok=True)


def verify_backup(path: Path) -> Dict[str, Any]:
    """Verify ZIP structure, manifest and encrypted authentication."""
    path = Path(path).resolve()
    if not path.is_file():
        raise BackupError("备份文件不存在")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise BackupError("备份文件为空或超过大小上限")
    encrypted = path.name.endswith(".enc")
    inspect_path = path
    temporary: Optional[Path] = None
    try:
        if encrypted:
            temporary = path.with_name(f".{path.name}.verify-{uuid.uuid4().hex}.zip")
            decrypt_file(path, temporary)
            inspect_path = temporary
        with zipfile.ZipFile(inspect_path) as archive:
            return _inspect_archive(archive)
    except zipfile.BadZipFile as exc:
        raise BackupError("备份包不是有效 ZIP") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cancel_pending_restore() -> bool:
    pending = pending_restore()
    if pending is None:
        return False
    staging = Path(str(pending.get("staging") or ""))
    if staging.is_dir() and staging.resolve().is_relative_to(RESTORE_ROOT.resolve()):
        shutil.rmtree(staging, ignore_errors=True)
    PENDING_FILE.unlink(missing_ok=True)
    return True


def _replace_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.restore-{uuid.uuid4().hex}")
    shutil.copy2(source, tmp)
    _replace_with_retry(tmp, target)


def _replace_with_retry(source: Path, target: Path) -> None:
    """Retry brief Windows locks without weakening the final replacement."""
    deadline = time.monotonic() + float(
        os.getenv("ZHIGE_RESTORE_REPLACE_RETRY_SECONDS", "30") or 30
    )
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def _replace_dir(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.restore-new-{uuid.uuid4().hex}")
    previous = target.with_name(f".{target.name}.restore-old-{uuid.uuid4().hex}")
    shutil.copytree(source, tmp)
    if target.exists():
        _replace_with_retry(target, previous)
    try:
        _replace_with_retry(tmp, target)
    except Exception:
        if previous.exists() and not target.exists():
            _replace_with_retry(previous, target)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def _restore_postgres(dump_file: Path) -> None:
    executable = shutil.which("pg_restore")
    if not executable:
        raise BackupError("PostgreSQL 恢复需要安装 pg_restore，并把它加入 PATH")
    args, env = _pg_connection_args()
    command = [
        executable,
        "--clean",
        "--if-exists",
        "--no-owner",
        "--no-privileges",
        *args,
        str(dump_file),
    ]
    completed = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        timeout=int(os.getenv("ZHIGE_RESTORE_TIMEOUT_SECONDS", "1800")),
        check=False,
    )
    if completed.returncode != 0:
        raise BackupError(f"pg_restore 失败：{(completed.stderr or completed.stdout)[-1000:]}")


def _apply_payload(payload: Path, manifest: Dict[str, Any]) -> None:
    """Apply an already authenticated and extracted backup payload."""
    database = manifest.get("database") or {}
    database_path = payload / str(database.get("path") or "")
    expected_hash = str(database.get("sha256") or "")
    if not database_path.is_file() or (
        expected_hash and _sha256(database_path) != expected_hash
    ):
        raise BackupError("恢复包数据库校验失败")

    if database.get("kind") == "sqlite":
        from src.db import database as database_module

        database_module.engine.dispose()
        target = _sqlite_path()
        for suffix in ("-wal", "-shm"):
            Path(str(target) + suffix).unlink(missing_ok=True)
        _replace_file(database_path, target)
    elif database.get("kind") == "postgresql":
        _restore_postgres(database_path)
    else:
        raise BackupError("恢复包数据库类型不受支持")

    runtime = _runtime_paths()
    for logical, target in (
        ("data/chroma_db", runtime["chroma"]),
        ("data/raw", runtime["raw"]),
        ("data/eval", runtime["eval"]),
    ):
        source = payload / logical
        if source.is_dir():
            _replace_dir(source, target)

    jobs_source = payload / "data/ingest_jobs.json"
    if jobs_source.is_file():
        _replace_file(jobs_source, DATA_ROOT / "ingest_jobs.json")
    for source in sorted((payload / "data").glob("kg_*.db*")):
        if source.is_file():
            _replace_file(source, DATA_ROOT / source.name)

    overrides_source = payload / "config/local_overrides.yaml"
    from src.settings_service import overrides_path

    overrides_target = overrides_path()
    if overrides_source.is_file():
        _replace_file(overrides_source, overrides_target)
    audit_source = payload / "logs/audit.jsonl"
    if audit_source.is_file():
        _replace_file(audit_source, runtime["logs"] / "audit.jsonl")


def _rollback_to_safety(safety_path: Path) -> Dict[str, Any]:
    """Restore the pre-restore safety archive after a partial failure."""
    if not safety_path.is_file():
        raise BackupError("恢复前安全备份不存在")
    rollback_upload = DATA_ROOT / ".restore_rollback" / (
        f"safety-{uuid.uuid4().hex}.zip.enc"
    )
    rollback_upload.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(safety_path, rollback_upload)
    PENDING_FILE.unlink(missing_ok=True)
    try:
        stage_restore(rollback_upload, safety_path.name)
        result = _apply_pending_restore(allow_rollback=False)
        if result is None:
            raise BackupError("恢复前安全备份未能进入待应用状态")
        return result
    finally:
        rollback_upload.unlink(missing_ok=True)


def _apply_pending_restore(*, allow_rollback: bool) -> Optional[Dict[str, Any]]:
    pending = pending_restore()
    if pending is None:
        return None
    if pending.get("blocked"):
        raise BackupError("最近一次恢复失败且自动回滚未完成，请检查状态后取消待恢复包")
    staging = Path(str(pending.get("staging") or "")).resolve()
    if not staging.is_relative_to(RESTORE_ROOT.resolve()) or not staging.is_dir():
        raise BackupError("待恢复包暂存路径无效")
    payload = staging / "payload"
    manifest = pending.get("manifest") or {}

    safety: Optional[Dict[str, Any]] = None
    try:
        safety = create_backup(output_dir=DATA_ROOT / ".restore_safety")
        _apply_payload(payload, manifest)

        status = {
            "status": "applied",
            "applied_at": _now(),
            "original_name": pending.get("original_name"),
            "safety_backup": safety["path"],
        }
        _write_json(STATUS_FILE, status)
        PENDING_FILE.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
        logger.info("恢复包已应用：%s，恢复前备份：%s", pending.get("original_name"), safety["path"])
        return status
    except Exception as exc:
        rollback: Dict[str, Any] = {"status": "not_attempted"}
        if allow_rollback and safety is not None:
            try:
                _rollback_to_safety(Path(str(safety["path"])))
                rollback = {"status": "succeeded"}
                shutil.rmtree(staging, ignore_errors=True)
            except Exception as rollback_exc:  # noqa: BLE001
                logger.exception("恢复失败后的自动回滚也失败")
                rollback = {
                    "status": "failed",
                    "error": str(rollback_exc)[:2000],
                }
                blocked = {**pending, "blocked": True, "rollback": rollback}
                _write_json(PENDING_FILE, blocked)
        _write_json(
            STATUS_FILE,
            {
                "status": "failed",
                "failed_at": _now(),
                "error": str(exc)[:2000],
                "safety_backup": safety.get("path") if safety else None,
                "original_name": pending.get("original_name"),
                "rollback": rollback,
            },
        )
        raise


def apply_pending_restore() -> Optional[Dict[str, Any]]:
    """启动时原子化地应用恢复包；后续步骤失败时自动恢复安全备份。"""
    return _apply_pending_restore(allow_rollback=True)


__all__ = [
    "BackupError",
    "apply_pending_restore",
    "backup_key_id",
    "cancel_pending_restore",
    "create_backup",
    "decrypt_file",
    "encrypt_file",
    "last_restore_status",
    "pending_restore",
    "stage_restore",
    "verify_backup",
]
