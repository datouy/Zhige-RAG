from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request

import src.backup_service as backup_service
import src.ingest_jobs as ingest_jobs
import src.tenant_quota as tenant_quota
from api.routes import team


ROOT = Path(__file__).resolve().parents[1]


def test_secure_session_cookie_has_expected_attributes(monkeypatch):
    monkeypatch.setenv("ZHIGE_COOKIE_SECURE", "1")
    response = Response()
    team._set_session_cookie(response, "opaque-test-token")
    header = response.headers["set-cookie"].lower()
    assert "httponly" in header
    assert "; secure" in header
    assert "samesite=lax" in header
    assert response.headers["cache-control"] == "no-store"


def test_token_response_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ZHIGE_TOKEN_RESPONSE_ENABLED", raising=False)
    assert team._token_response_enabled() is False


def test_bootstrap_requires_configuration_token(monkeypatch):
    monkeypatch.setattr(team, "team_mode", lambda: True)
    monkeypatch.delenv("ZHIGE_BOOTSTRAP_TOKEN", raising=False)
    body = team.BootstrapBody(
        username="first-admin",
        email="first@example.invalid",
        password="not-a-production-password",
    )
    with pytest.raises(HTTPException) as caught:
        team.bootstrap_admin(body, Request({"type": "http"}), Response(), object())
    assert caught.value.status_code == 503
    assert "初始化令牌" in str(caught.value.detail)


def _request(client: str, headers: dict[str, str] | None = None) -> Request:
    raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "root_path": "",
            "headers": raw_headers,
            "client": (client, 12345),
            "server": ("zhige.local", 80),
        }
    )


def test_request_logging_reuses_trusted_proxy_resolution(monkeypatch):
    from src.middleware import logging as logging_middleware

    monkeypatch.setattr("src.rate_limit.get_real_client_ip", lambda request: "198.51.100.77")
    assert logging_middleware._get_client_ip(_request("10.0.0.2")) == "198.51.100.77"


def test_local_tenant_quota_is_atomic_and_tenant_scoped(monkeypatch):
    monkeypatch.setenv("ZHIGE_TENANT_QUOTA_ENABLED", "true")
    monkeypatch.setenv("ZHIGE_QUOTA_CHAT_MINUTE", "1")
    monkeypatch.setenv("ZHIGE_QUOTA_CHAT_DAY", "100")
    monkeypatch.setattr(tenant_quota, "_redis", lambda: None)
    monkeypatch.setattr(tenant_quota, "_LOCAL", {})

    first = tenant_quota.check_tenant_quota("tenant-a", "chat")
    second = tenant_quota.check_tenant_quota("tenant-a", "chat")
    other = tenant_quota.check_tenant_quota("tenant-b", "chat")

    assert first.allowed is True
    assert second.allowed is False
    assert second.window == "minute"
    assert second.retry_after == 60
    assert other.allowed is True


def test_ingest_worker_lease_reports_missing_and_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INGEST_EXECUTION_MODE", "worker")
    assert ingest_jobs.worker_health()["status"] == "missing"
    (tmp_path / "ingest_worker_heartbeat.json").write_text(
        json.dumps({"worker_id": "test", "updated_at": time.time() - 61}),
        encoding="utf-8",
    )
    assert ingest_jobs.worker_health(max_age_seconds=30)["status"] == "stale"


def test_two_worker_processes_claim_a_job_once(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ingest_jobs, "_STATE_LOCK", None)
    with ingest_jobs._JOBS_LOCK:
        ingest_jobs._JOBS.clear()
    job_id = ingest_jobs.create_job("file", "atomic.txt", {"tenant": "t1"})

    script = """
import json
import sys
import src.ingest_jobs as jobs
claimed = jobs.claim_next_job(sys.argv[1])
print(json.dumps(claimed.get('job_id') if claimed else None))
"""
    env = {**os.environ, "RAG_DATA_DIR": str(tmp_path)}
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", script, f"worker-{index}"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for index in range(2)
    ]
    outputs = []
    for worker in workers:
        stdout, stderr = worker.communicate(timeout=30)
        assert worker.returncode == 0, stderr
        outputs.append(json.loads(stdout.strip().splitlines()[-1]))
    assert sorted(value is not None for value in outputs) == [False, True]
    assert job_id in outputs


def test_encrypted_backup_verifies_detects_tamper_and_stages_restore(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    raw = runtime / "raw"
    chroma = runtime / "chroma"
    eval_dir = runtime / "eval"
    logs = runtime / "logs"
    for path in (raw, chroma, eval_dir, logs):
        path.mkdir(parents=True, exist_ok=True)
    (raw / "policy.md").write_text("public test policy", encoding="utf-8")
    (eval_dir / "questions.jsonl").write_text('{"question":"test"}\n', encoding="utf-8")
    (logs / "audit.jsonl").write_text('{"action":"test"}\n', encoding="utf-8")
    (chroma / "index.bin").write_bytes(os.urandom(1024 * 1100))
    (chroma / ".chroma-write.lock").write_text("", encoding="utf-8")
    (runtime / "ingest_jobs.json").write_text("{}", encoding="utf-8")
    with sqlite3.connect(runtime / "kg_test.db") as db:
        db.execute("CREATE TABLE nodes(id TEXT)")
    with sqlite3.connect(runtime / "users.db") as db:
        db.execute("CREATE TABLE users(id TEXT)")
        db.execute("INSERT INTO users VALUES ('test')")

    monkeypatch.setenv("ZHIGE_BACKUP_ENCRYPTION", "required")
    monkeypatch.setenv("ZHIGE_BACKUP_KEY_FILE", str(tmp_path / "backup.key"))
    monkeypatch.setattr(backup_service, "DATA_ROOT", runtime)
    monkeypatch.setattr(backup_service, "BACKUP_ROOT", runtime / "backups")
    monkeypatch.setattr(backup_service, "RESTORE_ROOT", runtime / ".restore_staging")
    monkeypatch.setattr(backup_service, "PENDING_FILE", runtime / ".restore_pending.json")
    monkeypatch.setattr(backup_service, "STATUS_FILE", runtime / ".restore_status.json")
    monkeypatch.setattr(backup_service, "_sqlite_path", lambda: runtime / "users.db")
    monkeypatch.setattr(
        backup_service,
        "_runtime_paths",
        lambda: {"raw": raw, "chroma": chroma, "eval": eval_dir, "logs": logs},
    )

    result = backup_service.create_backup()
    path = Path(result["path"])
    manifest = backup_service.verify_backup(path)
    assert path.name.endswith(".zip.enc")
    assert result["encryption"]["algorithm"] == "AES-256-GCM"
    assert manifest["database"]["kind"] == "sqlite"
    assert any(item["path"] == "data/chroma_db/index.bin" for item in manifest["files"])
    assert all(item["path"] != "data/chroma_db/.chroma-write.lock" for item in manifest["files"])

    tampered = tmp_path / "tampered.zip.enc"
    shutil.copy2(path, tampered)
    with tampered.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        value = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(backup_service.BackupError):
        backup_service.verify_backup(tampered)

    staged_input = tmp_path / "restore.zip.enc"
    shutil.copy2(path, staged_input)
    pending = backup_service.stage_restore(staged_input, path.name)
    assert pending["encrypted"] is True
    assert Path(pending["staging"]).is_dir()
    backup_service.cancel_pending_restore()


def test_restore_replace_retries_transient_windows_permission_error(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("new", encoding="utf-8")
    target.write_text("old", encoding="utf-8")
    real_replace = os.replace
    attempts = {"count": 0}

    def flaky_replace(src, dst):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("temporarily locked")
        return real_replace(src, dst)

    monkeypatch.setattr(backup_service.os, "replace", flaky_replace)
    monkeypatch.setenv("ZHIGE_RESTORE_REPLACE_RETRY_SECONDS", "2")
    backup_service._replace_with_retry(source, target)
    assert attempts["count"] == 3
    assert target.read_text(encoding="utf-8") == "new"


def test_failed_multi_store_restore_rolls_back_database_and_files(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    raw = runtime / "raw"
    chroma = runtime / "chroma"
    eval_dir = runtime / "eval"
    logs = runtime / "logs"
    for path in (raw, chroma, eval_dir, logs):
        path.mkdir(parents=True, exist_ok=True)
    user_db = runtime / "users.db"

    def write_user_db(value: str) -> None:
        with closing(sqlite3.connect(user_db)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS settings(value TEXT)")
            db.execute("DELETE FROM settings")
            db.execute("INSERT INTO settings VALUES (?)", (value,))

    def read_user_db() -> str:
        with closing(sqlite3.connect(user_db)) as db:
            return str(db.execute("SELECT value FROM settings").fetchone()[0])

    write_user_db("backup")
    (raw / "policy.md").write_text("backup policy", encoding="utf-8")
    monkeypatch.setenv("ZHIGE_BACKUP_ENCRYPTION", "required")
    monkeypatch.setenv("ZHIGE_BACKUP_KEY_FILE", str(tmp_path / "backup.key"))
    monkeypatch.setattr(backup_service, "DATA_ROOT", runtime)
    monkeypatch.setattr(backup_service, "BACKUP_ROOT", runtime / "backups")
    monkeypatch.setattr(backup_service, "RESTORE_ROOT", runtime / ".restore_staging")
    monkeypatch.setattr(backup_service, "PENDING_FILE", runtime / ".restore_pending.json")
    monkeypatch.setattr(backup_service, "STATUS_FILE", runtime / ".restore_status.json")
    monkeypatch.setattr(backup_service, "_sqlite_path", lambda: user_db)
    monkeypatch.setattr(
        backup_service,
        "_runtime_paths",
        lambda: {"raw": raw, "chroma": chroma, "eval": eval_dir, "logs": logs},
    )

    backup_path = Path(backup_service.create_backup()["path"])
    write_user_db("current")
    (raw / "policy.md").write_text("current policy", encoding="utf-8")
    upload = tmp_path / "restore.zip.enc"
    shutil.copy2(backup_path, upload)
    backup_service.stage_restore(upload, upload.name)

    real_replace_dir = backup_service._replace_dir
    attempts = {"count": 0}

    def fail_after_database(restore_source, target):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("injected multi-store failure")
        return real_replace_dir(restore_source, target)

    monkeypatch.setattr(backup_service, "_replace_dir", fail_after_database)
    with pytest.raises(OSError, match="injected multi-store failure"):
        backup_service.apply_pending_restore()

    assert read_user_db() == "current"
    assert (raw / "policy.md").read_text(encoding="utf-8") == "current policy"
    status = backup_service.last_restore_status()
    assert status["status"] == "failed"
    assert status["rollback"]["status"] == "succeeded"
    assert backup_service.pending_restore() is None


def test_prometheus_merges_counters_from_two_worker_processes(tmp_path):
    metrics_dir = tmp_path / "prometheus"
    metrics_dir.mkdir()
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(metrics_dir)}
    worker_script = """
from src.monitoring.metrics import http_requests_total, llm_tokens_total
http_requests_total.labels(method='GET', endpoint='/health', status_code='200').inc()
llm_tokens_total.labels(kind='prompt').inc(7)
"""
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", worker_script],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
    scrape_script = """
from src.monitoring.metrics import metrics_collector
payload, _ = metrics_collector.generate_metrics_output()
print(payload.decode('utf-8'))
"""
    completed = subprocess.run(
        [sys.executable, "-c", scrape_script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert 'http_requests_total{endpoint="/health",method="GET",status_code="200"} 2.0' in completed.stdout
    assert 'llm_tokens_total{kind="prompt"} 14.0' in completed.stdout


def test_load_test_cookie_file_adapter_does_not_log_secret(tmp_path):
    from scripts.load_test import _session_headers

    first = tmp_path / "one.cookie"
    second = tmp_path / "two.cookie"
    first.write_text("zhige_session=one", encoding="utf-8")
    second.write_text("zhige_session=two", encoding="utf-8")
    headers = _session_headers([first, second])
    assert headers == [
        {"Cookie": "zhige_session=one"},
        {"Cookie": "zhige_session=two"},
    ]
    second.write_text("zhige_session=two\nZhigE_SESSION=leak", encoding="utf-8")
    with pytest.raises(ValueError):
        _session_headers([second])
