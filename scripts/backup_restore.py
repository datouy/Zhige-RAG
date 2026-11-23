"""知阁完整备份与恢复工具。

备份范围：
- PostgreSQL 业务库和 pgvector 向量表（pg_dump）；
- 或本地 SQLite 业务库、Chroma 持久化目录；
- data/raw 原始文档；
- SQLite 知识图谱文件或 Neo4j Entity/RELATION 导出；
- config.yaml / prompts.yaml / custom_dict.txt 及脱敏后的 local_overrides.yaml。

安全规则：
- 不备份 .env、模型凭据、Cookie、Vault token；
- 恢复必须显式传 --force；
- 覆盖 SQLite 前自动保留 .pre-restore 文件；
- PostgreSQL 恢复默认不读取环境变量，必须显式传 --database-url。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARCHIVE_VERSION = 1


class BackupError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_sqlite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(destination)
    try:
        src.backup(dst)
        dst.commit()
    finally:
        dst.close()
        src.close()


def _restore_sqlite(source: Path, destination: Path) -> Optional[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if destination.exists():
        backup = destination.with_suffix(destination.suffix + ".pre-restore")
        _copy_sqlite(destination, backup)
    tmp = destination.with_suffix(destination.suffix + ".restore-tmp")
    shutil.copy2(source, tmp)
    tmp.replace(destination)
    return backup


def _ensure_child(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise BackupError(f"拒绝操作项目根目录之外的路径：{resolved}")
    return resolved


def _config_paths(root: Path) -> list[Path]:
    config_dir = root / "config"
    return [
        path
        for path in (
            config_dir / "config.yaml",
            config_dir / "prompts.yaml",
            config_dir / "custom_dict.txt",
            config_dir / "local_overrides.yaml",
        )
        if path.exists()
    ]


def _sanitized_config(path: Path) -> bytes:
    import yaml

    from src.secrets_store import strip_plaintext_secrets

    if path.suffix.lower() not in {".yaml", ".yml"}:
        return path.read_bytes()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cleaned, _found = strip_plaintext_secrets(data)
    return yaml.safe_dump(cleaned, allow_unicode=True, sort_keys=False).encode("utf-8")


def _export_neo4j(path: Path) -> None:
    uri = os.getenv("NEO4J_URI", "")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not uri or not password:
        raise BackupError("Neo4j 备份需要 NEO4J_URI / NEO4J_PASSWORD")
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        raise BackupError("Neo4j 备份需要安装 neo4j Python 驱动") from exc
    driver = GraphDatabase.driver(uri, auth=(user, password))
    rows = []
    try:
        with driver.session(database=database) as session:
            for record in session.run(
                "MATCH (n:Entity) RETURN n.name AS name, n.tenant_id AS tenant_id, "
                "n.type AS type, n.description AS description, n.aliases AS aliases, "
                "n.attributes AS attributes"
            ):
                rows.append({"kind": "entity", **dict(record)})
            for record in session.run(
                "MATCH (a:Entity)-[r:RELATION]->(b:Entity) "
                "WHERE a.tenant_id = b.tenant_id AND r.tenant_id = a.tenant_id "
                "RETURN a.name AS source, b.name AS target, a.tenant_id AS tenant_id, "
                "r.type AS type, r.description AS description, r.weight AS weight, "
                "r.attributes AS attributes"
            ):
                rows.append({"kind": "relation", **dict(record)})
    finally:
        driver.close()
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows) + "\n",
        encoding="utf-8",
    )


def _import_neo4j(path: Path) -> int:
    uri = os.getenv("NEO4J_URI", "")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not uri or not password:
        raise BackupError("Neo4j 恢复需要 NEO4J_URI / NEO4J_PASSWORD")
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        raise BackupError("Neo4j 恢复需要安装 neo4j Python 驱动") from exc
    driver = GraphDatabase.driver(uri, auth=(user, password))
    count = 0
    try:
        with driver.session(database=database) as session:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "entity":
                    session.run(
                        "MERGE (n:Entity {name: $name, tenant_id: $tenant_id}) "
                        "SET n.type = $type, n.description = $description, "
                        "n.aliases = $aliases, n.attributes = $attributes",
                        row,
                    )
                elif row.get("kind") == "relation":
                    session.run(
                        "MATCH (a:Entity {name: $source, tenant_id: $tenant_id}) "
                        "MATCH (b:Entity {name: $target, tenant_id: $tenant_id}) "
                        "MERGE (a)-[r:RELATION {type: $type, tenant_id: $tenant_id}]->(b) "
                        "SET r.description = $description, r.weight = $weight, "
                        "r.attributes = $attributes",
                        row,
                    )
                count += 1
    finally:
        driver.close()
    return count


def _run_pg_dump(database_url: str, output: Path) -> None:
    command = [
        "pg_dump",
        "--format=custom",
        "--no-owner",
        "--no-privileges",
        f"--file={output}",
        f"--dbname={database_url}",
    ]
    subprocess.run(command, check=True)


def _run_pg_restore(database_url: str, source: Path) -> None:
    command = [
        "pg_restore",
        "--clean",
        "--if-exists",
        "--no-owner",
        "--no-privileges",
        f"--dbname={database_url}",
        str(source),
    ]
    subprocess.run(command, check=True)


def create_backup(
    output: Path,
    *,
    root: Path = ROOT,
    backends: Optional[Dict[str, str]] = None,
    allow_live: bool = False,
) -> Dict[str, Any]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    backends = dict(backends or {})
    database_url = backends.get("database_url") or os.getenv("DATABASE_URL", "")
    vector_backend = str(backends.get("vector", "chroma")).lower()
    kg_backend = str(backends.get("kg", "sqlite")).lower()
    warnings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="zhige-backup-") as temp_name:
        temp = Path(temp_name)
        (temp / "config").mkdir()

        if database_url.startswith(("postgresql", "postgres")):
            _run_pg_dump(database_url, temp / "database.dump")
            db_kind = "postgresql"
        else:
            db_path = Path(
                backends.get("sqlite_db")
                or os.getenv("DB_PATH")
                or root / "data" / "users.db"
            )
            if not db_path.is_absolute():
                db_path = root / db_path
            if not db_path.exists():
                raise BackupError(f"SQLite 数据库不存在：{db_path}")
            _copy_sqlite(db_path, temp / "sqlite" / "users.db")
            db_kind = "sqlite"

        raw_dir = Path(backends.get("raw_dir") or root / "data" / "raw")
        if raw_dir.is_absolute():
            raw_source = raw_dir
        else:
            raw_source = root / raw_dir
        if raw_source.exists():
            shutil.copytree(raw_source, temp / "raw")

        if vector_backend == "chroma":
            chroma_dir = Path(backends.get("chroma_dir") or root / "data" / "chroma_db")
            if not chroma_dir.is_absolute():
                chroma_dir = root / chroma_dir
            if chroma_dir.exists():
                if not allow_live:
                    warnings.append(
                        "Chroma 目录按在线文件复制。生产备份应在维护窗口停止 API，"
                        "或使用集中式 pgvector 后端配合 pg_dump。"
                    )
                shutil.copytree(chroma_dir, temp / "vector" / "chroma")
        elif vector_backend == "pgvector":
            if db_kind != "postgresql":
                raise BackupError("pgvector 备份必须同时提供 PostgreSQL DATABASE_URL")
        else:
            raise BackupError(f"未知向量后端：{vector_backend}")

        if kg_backend == "sqlite":
            kg_dir = temp / "kg" / "sqlite"
            kg_dir.mkdir(parents=True, exist_ok=True)
            for db_file in sorted((root / "data").glob("kg*.db")):
                _copy_sqlite(db_file, kg_dir / db_file.name)
        elif kg_backend == "neo4j":
            (temp / "kg").mkdir()
            _export_neo4j(temp / "kg" / "neo4j.jsonl")
        else:
            raise BackupError(f"未知图谱后端：{kg_backend}")

        for path in _config_paths(root):
            (temp / "config" / path.name).write_bytes(_sanitized_config(path))

        files = []
        for path in sorted(temp.rglob("*")):
            if path.is_file() and path.name != "manifest.json" and path.name != "checksums.sha256":
                files.append(
                    {
                        "path": path.relative_to(temp).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
        manifest = {
            "archive_version": ARCHIVE_VERSION,
            "created_at": datetime.now().astimezone().isoformat(),
            "database": db_kind,
            "vector_backend": vector_backend,
            "kg_backend": kg_backend,
            "files": files,
            "warnings": warnings,
        }
        (temp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (temp / "checksums.sha256").write_text(
            "\n".join(f"{item['sha256']}  {item['path']}" for item in files) + "\n",
            encoding="utf-8",
        )
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(temp.iterdir()):
                archive.add(path, arcname=path.name)
    return {
        "archive": str(output),
        "bytes": output.stat().st_size,
        "database": db_kind,
        "warnings": warnings,
    }


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(root):
            raise BackupError(f"归档包含非法路径：{member.name}")
        if member.issym() or member.islnk():
            raise BackupError(f"归档包含链接，已拒绝：{member.name}")
    try:
        archive.extractall(destination, filter="data")
    except TypeError:
        archive.extractall(destination)


def _verify_checksums(directory: Path, manifest: Dict[str, Any]) -> None:
    for item in manifest.get("files", []):
        path = directory / item["path"]
        if not path.is_file():
            raise BackupError(f"备份缺少文件：{item['path']}")
        if _sha256(path) != item["sha256"]:
            raise BackupError(f"备份校验失败：{item['path']}")


def restore_backup(
    archive_path: Path,
    *,
    root: Path = ROOT,
    force: bool = False,
    database_url: Optional[str] = None,
    restore_data: bool = True,
    restore_config: bool = False,
    replace_data: bool = False,
    import_neo4j: bool = False,
    sqlite_db: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    chroma_dir: Optional[Path] = None,
    kg_data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if not force:
        raise BackupError("恢复会覆盖现有数据，必须显式传 --force")
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise BackupError(f"备份文件不存在：{archive_path}")
    result: Dict[str, Any] = {
        "archive": str(archive_path),
        "actions": [],
        "skipped": [],
    }
    with tempfile.TemporaryDirectory(prefix="zhige-restore-") as temp_name:
        temp = Path(temp_name)
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract(archive, temp)
        manifest_path = temp / "manifest.json"
        if not manifest_path.exists():
            raise BackupError("备份缺少 manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("archive_version", 0)) != ARCHIVE_VERSION:
            raise BackupError(f"不支持的备份版本：{manifest.get('archive_version')}")
        _verify_checksums(temp, manifest)

        if manifest.get("database") == "postgresql":
            if not database_url:
                raise BackupError("恢复 PostgreSQL 必须显式传 --database-url")
            _run_pg_restore(database_url, temp / "database.dump")
            result["actions"].append("postgresql")
        elif manifest.get("database") == "sqlite":
            target_db = Path(sqlite_db or os.getenv("DB_PATH") or root / "data" / "users.db")
            if not target_db.is_absolute():
                target_db = root / target_db
            backup = _restore_sqlite(temp / "sqlite" / "users.db", target_db)
            result["actions"].append("sqlite")
            if backup:
                result["pre_restore_sqlite"] = str(backup)

        if restore_data and (temp / "raw").exists():
            target_raw = Path(raw_dir or os.getenv("RAW_DOCS_DIR") or root / "data" / "raw")
            if not target_raw.is_absolute():
                target_raw = root / target_raw
            if replace_data and target_raw.exists():
                shutil.rmtree(_ensure_child(target_raw, root))
            shutil.copytree(temp / "raw", target_raw, dirs_exist_ok=True)
            result["actions"].append("raw_docs")

        if manifest.get("vector_backend") == "chroma" and restore_data:
            target_chroma = Path(
                chroma_dir
                or os.getenv("VECTOR_STORE_PERSIST_DIR")
                or root / "data" / "chroma_db"
            )
            if not target_chroma.is_absolute():
                target_chroma = root / target_chroma
            source_chroma = temp / "vector" / "chroma"
            if source_chroma.exists():
                if replace_data and target_chroma.exists():
                    shutil.rmtree(_ensure_child(target_chroma, root))
                shutil.copytree(source_chroma, target_chroma, dirs_exist_ok=True)
                result["actions"].append("chroma")

        if manifest.get("kg_backend") == "sqlite" and restore_data:
            target_data = Path(kg_data_dir or os.getenv("RAG_DATA_DIR") or root / "data")
            if not target_data.is_absolute():
                target_data = root / target_data
            target_data.mkdir(parents=True, exist_ok=True)
            source_kg = temp / "kg" / "sqlite"
            if source_kg.exists():
                for source in source_kg.glob("*.db"):
                    _restore_sqlite(source, target_data / source.name)
                result["actions"].append("kg_sqlite")

        if import_neo4j and manifest.get("kg_backend") == "neo4j":
            count = _import_neo4j(temp / "kg" / "neo4j.jsonl")
            result["actions"].append("neo4j")
            result["neo4j_rows"] = count
        elif manifest.get("kg_backend") == "neo4j":
            result["skipped"].append("neo4j (需要 --import-neo4j)")

        if restore_config and (temp / "config").exists():
            target_config = root / "config"
            target_config.mkdir(parents=True, exist_ok=True)
            for source in (temp / "config").iterdir():
                shutil.copy2(source, target_config / source.name)
            result["actions"].append("config")
        elif (temp / "config").exists():
            result["skipped"].append("config (需要 --restore-config)")

    result["status"] = "restored"
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="知阁完整备份与恢复")
    sub = parser.add_subparsers(dest="command", required=True)

    backup = sub.add_parser("backup")
    backup.add_argument("--output", default="")
    backup.add_argument("--allow-live", action="store_true")
    backup.add_argument("--sqlite-db", default="")
    backup.add_argument("--raw-dir", default="")
    backup.add_argument("--chroma-dir", default="")
    backup.add_argument("--vector-backend", default="")
    backup.add_argument("--kg-backend", default="")

    restore = sub.add_parser("restore")
    restore.add_argument("archive")
    restore.add_argument("--force", action="store_true")
    restore.add_argument("--database-url", default="")
    restore.add_argument("--restore-config", action="store_true")
    restore.add_argument("--replace-data", action="store_true")
    restore.add_argument("--import-neo4j", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "backup":
            from src.utils import apply_env_overrides, load_config

            cfg = apply_env_overrides(load_config(str(ROOT / "config" / "config.yaml")))
            vector_backend = args.vector_backend or (cfg.get("vector_store") or {}).get("backend", "chroma")
            kg_backend = args.kg_backend or (cfg.get("knowledge_graph") or {}).get("backend", "sqlite")
            output = Path(args.output) if args.output else (
                ROOT
                / "data"
                / "backups"
                / f"zhige-{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar.gz"
            )
            result = create_backup(
                output,
                backends={
                    "database_url": os.getenv("DATABASE_URL", ""),
                    "sqlite_db": args.sqlite_db,
                    "raw_dir": args.raw_dir,
                    "chroma_dir": args.chroma_dir,
                    "vector": vector_backend,
                    "kg": kg_backend,
                },
                allow_live=args.allow_live,
            )
        else:
            result = restore_backup(
                Path(args.archive),
                force=args.force,
                database_url=args.database_url or None,
                restore_config=args.restore_config,
                replace_data=args.replace_data,
                import_neo4j=args.import_neo4j,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (BackupError, subprocess.CalledProcessError, OSError) as exc:
        print(f"备份/恢复失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
