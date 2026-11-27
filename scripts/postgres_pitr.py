#!/usr/bin/env python
"""PostgreSQL base-backup, WAL archive and point-in-time restore helper.

Examples (PowerShell):

    python scripts/postgres_pitr.py base-backup --output .pitr/base
    python scripts/postgres_pitr.py sync-wal --archive .pitr/wal --offsite D:/pitr/wal
    python scripts/postgres_pitr.py restore --base .pitr/base --archive .pitr/wal \
        --output .pitr/restore --target-time 2026-11-27T10:00:00Z --port 55433 --start
    python scripts/postgres_pitr.py stop --data-dir .pitr/restore

The database URL is read from ``DATABASE_URL``. Credentials are passed to
PostgreSQL tools through ``PGPASSWORD`` and are never written to the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.engine import make_url


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _database_parts(url_text: str) -> dict[str, Any]:
    if not url_text.strip():
        raise ValueError("DATABASE_URL is required")
    url = make_url(url_text)
    if not url.drivername.startswith("postgresql"):
        raise ValueError("DATABASE_URL must point to PostgreSQL")
    if not url.host or not url.database:
        raise ValueError("DATABASE_URL must include host and database")
    return {
        "host": url.host,
        "port": int(url.port or 5432),
        "user": url.username or "postgres",
        "database": url.database,
        "password": url.password or "",
    }


def _tool(pg_bin: str, name: str) -> str:
    if pg_bin:
        path = Path(pg_bin) / name
        if os.name == "nt":
            path = path.with_suffix(".exe")
        if path.is_file():
            return str(path)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"{name} not found; pass --pg-bin or add PostgreSQL bin to PATH"
    )


def _env(parts: dict[str, Any]) -> dict[str, str]:
    result = dict(os.environ)
    if parts["password"]:
        result["PGPASSWORD"] = str(parts["password"])
    return result


def _run(command: list[str], *, parts: dict[str, Any], timeout: float = 1800) -> None:
    completed = subprocess.run(
        command,
        env=_env(parts),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()[-3000:]
        raise RuntimeError(f"{' '.join(command[:2])} failed: {message}")


def _common_connection(parts: dict[str, Any]) -> list[str]:
    return [
        "-h",
        str(parts["host"]),
        "-p",
        str(parts["port"]),
        "-U",
        str(parts["user"]),
    ]


def base_backup(args: argparse.Namespace) -> dict[str, Any]:
    parts = _database_parts(args.database_url)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("base-backup output must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    _run(
        [
            _tool(args.pg_bin, "pg_basebackup"),
            *_common_connection(parts),
            "-D",
            str(output),
            "-Fp",
            "-X",
            "stream",
            "--checkpoint=fast",
            "--no-password",
        ],
        parts=parts,
        timeout=args.timeout,
    )
    completed = datetime.now(timezone.utc)
    manifest = {
        "kind": "postgresql-base-backup",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "host": parts["host"],
        "port": parts["port"],
        "database": parts["database"],
        "path": str(output),
        "pg_version_file": (output / "PG_VERSION").read_text().strip(),
    }
    (output.parent / f"{output.name}-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def sync_wal(args: argparse.Namespace) -> dict[str, Any]:
    archive = Path(args.archive).resolve()
    offsite = Path(args.offsite).resolve()
    archive.mkdir(parents=True, exist_ok=True)
    offsite.mkdir(parents=True, exist_ok=True)
    copied = 0
    verified = 0
    files: list[dict[str, Any]] = []
    for source in sorted(archive.glob("*")):
        if not source.is_file() or source.name.endswith(".partial"):
            continue
        digest = _sha256(source)
        destination = offsite / source.name
        if not destination.is_file() or _sha256(destination) != digest:
            partial = destination.with_name(destination.name + ".partial")
            shutil.copy2(source, partial)
            os.replace(partial, destination)
            copied += 1
        if _sha256(destination) != digest:
            raise RuntimeError(f"offsite WAL digest mismatch: {source.name}")
        verified += 1
        files.append({"name": source.name, "size": source.stat().st_size, "sha256": digest})
    manifest = {
        "kind": "postgresql-wal-offsite-sync",
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "offsite": str(offsite),
        "copied": copied,
        "verified": verified,
        "files": files,
    }
    status = offsite / "wal-sync-manifest.json"
    temporary = status.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, status)
    return manifest


def restore(args: argparse.Namespace) -> dict[str, Any]:
    parts = _database_parts(args.database_url)
    base = Path(args.base).resolve()
    archive = Path(args.archive).resolve()
    output = Path(args.output).resolve()
    if not base.is_dir() or not (base / "PG_VERSION").is_file():
        raise RuntimeError("base backup directory is invalid")
    if not archive.is_dir():
        raise RuntimeError("WAL archive directory does not exist")
    if output.exists():
        raise RuntimeError("restore output must not already exist")
    started = time.perf_counter()
    shutil.copytree(base, output)
    for stale in ("postmaster.pid", "postmaster.opts"):
        (output / stale).unlink(missing_ok=True)

    if os.name == "nt":
        # PostgreSQL invokes restore_command through cmd.exe on Windows. A
        # small wrapper avoids nested-quote ambiguity while retaining the
        # documented %f/%p substitution.
        wrapper = output.parent / f".{output.name}-restore-wal.cmd"
        wrapper.write_text(
            "@echo off\r\n"
            f"copy /Y \"{archive}\\%~1\" \"%~2\"\r\n"
            "if errorlevel 1 exit /b 1\r\n",
            encoding="utf-8",
        )
        restore_command = f'"{wrapper}" "%f" "%p"'
    else:
        restore_command = f'cp "{archive / "%f"}" "%p"'
    # postgresql.auto.conf uses SQL-style string escaping on Windows, so a
    # single backslash would turn ``E:\project`` into ``E:project``.
    escaped_command = restore_command.replace("\\", "\\\\").replace("'", "''")
    auto_conf = output / "postgresql.auto.conf"
    with auto_conf.open("a", encoding="utf-8") as stream:
        stream.write("\n# Added by scripts/postgres_pitr.py for isolated recovery\n")
        stream.write(f"restore_command = '{escaped_command}'\n")
        stream.write(f"recovery_target_time = '{args.target_time}'\n")
        stream.write("recovery_target_inclusive = on\n")
        stream.write("recovery_target_action = 'promote'\n")
        stream.write("hot_standby = on\n")
        stream.write(f"port = {args.port}\n")
        stream.write("listen_addresses = '127.0.0.1'\n")
    (output / "recovery.signal").write_text("", encoding="ascii")

    if args.start:
        log_path = Path(args.log).resolve() if args.log else output.parent / "restore-postgres.log"
        _run(
            [
                _tool(args.pg_bin, "pg_ctl"),
                "-D",
                str(output),
                "-l",
                str(log_path),
                "start",
            ],
            parts=parts,
            timeout=120,
        )
        ready = _tool(args.pg_bin, "pg_isready")
        deadline = time.monotonic() + args.wait_seconds
        while time.monotonic() < deadline:
            probe = subprocess.run(
                [ready, "-h", "127.0.0.1", "-p", str(args.port)],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode == 0 and "accepting connections" in probe.stdout:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("restored PostgreSQL did not become ready")
    return {
        "kind": "postgresql-pitr-restore",
        "restored_at": datetime.now(timezone.utc).isoformat(),
        "base": str(base),
        "target_time": args.target_time,
        "output": str(output),
        "port": args.port,
        "started": bool(args.start),
        "copy_and_config_seconds": round(time.perf_counter() - started, 3),
    }


def stop(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise RuntimeError("data directory does not exist")
    _run(
        [_tool(args.pg_bin, "pg_ctl"), "-D", str(data_dir), "stop", "-m", "fast"],
        parts=_database_parts(args.database_url),
        timeout=120,
    )
    return {"stopped": str(data_dir), "at": datetime.now(timezone.utc).isoformat()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--pg-bin", default=os.getenv("PG_BIN", ""))
    sub = parser.add_subparsers(dest="command", required=True)

    backup_parser = sub.add_parser("base-backup")
    backup_parser.add_argument("--output", required=True)
    backup_parser.add_argument("--timeout", type=float, default=1800)
    backup_parser.set_defaults(func=base_backup)

    sync_parser = sub.add_parser("sync-wal")
    sync_parser.add_argument("--archive", required=True)
    sync_parser.add_argument("--offsite", required=True)
    sync_parser.set_defaults(func=sync_wal)

    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--base", required=True)
    restore_parser.add_argument("--archive", required=True)
    restore_parser.add_argument("--output", required=True)
    restore_parser.add_argument("--target-time", required=True)
    restore_parser.add_argument("--port", type=int, default=55433)
    restore_parser.add_argument("--log")
    restore_parser.add_argument("--start", action="store_true")
    restore_parser.add_argument("--wait-seconds", type=float, default=180)
    restore_parser.set_defaults(func=restore)

    stop_parser = sub.add_parser("stop")
    stop_parser.add_argument("--data-dir", required=True)
    stop_parser.set_defaults(func=stop)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
