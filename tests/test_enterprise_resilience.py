from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from ipaddress import IPv4Network
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

import src.ingest_jobs as ingest_jobs
import src.rate_limit as rate_limit
from api.routes import chat as chat_routes
from src.circuit_breaker import CircuitBreakerOpenError
from src.db.models import Base, ChatMessage, ChatSession
from src.http_client import RemoteServiceError
from src.kb_service import KBError, append_message
from src.llm_concurrency import LLMConcurrencyLimitError


def test_public_health_llm_snapshot_omits_internal_endpoint_and_model():
    from api.routes.system import _llm_health_snapshot

    status, details = _llm_health_snapshot(
        {
            "backend": "openai",
            "openai": {
                "model": "private-model",
                "base_url": "http://internal-llm.example/v1",
                "api_key": "secret-key",
            },
        }
    )

    serialized = json.dumps(details)
    assert status == "available"
    assert details["configured"] is True
    assert details["api_key_configured"] is True
    assert "private-model" not in serialized
    assert "internal-llm.example" not in serialized
    assert "secret-key" not in serialized


def test_public_health_helpers_omit_paths_and_collection_names():
    from api.routes.system import _disk_health_details, _storage_health_details

    disk = SimpleNamespace(
        total=300 * 1024**3,
        used=60 * 1024**3,
        free=240 * 1024**3,
        percent=20.0,
    )
    disk_details = _disk_health_details(disk)
    assert disk_details == {
        "total_gb": 300.0,
        "used_gb": 60.0,
        "free_gb": 240.0,
        "percent": 20.0,
    }

    storage_details = _storage_health_details(
        ["active_collection", "tenant_private_orphan"],
        {"active_collection"},
        2,
    )
    serialized = json.dumps(storage_details)
    assert "active_collection" not in serialized
    assert "tenant_private_orphan" not in serialized
    assert storage_details["chroma_collections_orphaned"] == 1
    assert storage_details["cleanup_required"] is True


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


def test_untrusted_direct_client_cannot_spoof_forwarded_for():
    request = _request(
        "203.0.113.9",
        {"X-Forwarded-For": "198.51.100.1, 198.51.100.2"},
    )
    assert rate_limit.get_real_client_ip(request) == "203.0.113.9"


def test_trusted_proxy_uses_rightmost_untrusted_hop(monkeypatch):
    monkeypatch.setattr(
        rate_limit,
        "_TRUSTED_NETWORKS",
        [IPv4Network("127.0.0.1/32"), IPv4Network("10.0.0.0/8")],
    )
    request = _request(
        "10.0.0.2",
        {
            # 最左侧是客户端自带的伪造值；右侧才是代理实际观察到的地址。
            "X-Forwarded-For": "203.0.113.99, 198.51.100.20, 10.0.0.3",
        },
    )
    assert rate_limit.get_real_client_ip(request) == "198.51.100.20"


def test_team_session_token_is_hashed_rate_limit_key(monkeypatch):
    import src.team_auth as team_auth

    monkeypatch.setattr(team_auth, "team_mode", lambda: True)
    monkeypatch.setattr(team_auth, "request_token", lambda request: "opaque-secret")
    request = _request("198.51.100.7")
    expected = hashlib.sha256(b"opaque-secret").hexdigest()[:24]
    assert rate_limit.get_rate_limit_key(request) == f"session:{expected}"


def test_rate_limiter_has_bounded_redis_timeout_and_memory_fallback():
    assert rate_limit.limiter._in_memory_fallback_enabled is True
    assert rate_limit.limiter._in_memory_fallback
    assert rate_limit.limiter._storage_options["socket_connect_timeout"] == 0.5
    assert rate_limit.limiter._storage_options["socket_timeout"] == 0.5


def test_rate_limit_health_does_not_expose_storage_url(monkeypatch):
    monkeypatch.setattr(rate_limit.limiter._storage, "check", lambda: False)
    monkeypatch.setattr(
        rate_limit,
        "_RATE_LIMIT_STORAGE_URL",
        "redis://user:secret@example.invalid/0",
    )
    assert rate_limit.rate_limit_health() == {
        "status": "degraded",
        "backend": "redis",
        "fallback": True,
    }


def test_generation_failure_mapping_is_safe_and_retryable():
    sensitive = (
        "RemoteServiceError: POST http://internal-llm.example/v1 "
        "model=secret-model Authorization=Bearer top-secret"
    )
    error = chat_routes._generation_failure_http(RemoteServiceError(sensitive))

    assert error is not None
    assert error.status_code == 503
    assert error.headers["Retry-After"] == "10"
    assert error.headers["X-Zhige-Error-Code"] == "generation_backend_unavailable"
    assert "internal-llm" not in str(error.detail)
    assert "secret-model" not in str(error.detail)
    assert "top-secret" not in str(error.detail)

    circuit = chat_routes._generation_failure_http(
        CircuitBreakerOpenError("llm", 12.2)
    )
    assert circuit is not None
    assert circuit.status_code == 503
    assert circuit.headers["Retry-After"] == "13"

    saturated = chat_routes._generation_failure_http(
        LLMConcurrencyLimitError("active=14, waiting=64")
    )
    assert saturated is not None
    assert saturated.status_code == 503
    assert saturated.headers["X-Zhige-Error-Code"] == "generation_capacity_exhausted"


def test_chat_endpoint_returns_safe_503_when_generation_dependency_fails(monkeypatch):
    class FailingPipeline:
        long_term_store = None

        def answer(self, *_args, **_kwargs):
            raise RemoteServiceError(
                "POST http://internal-llm.example/v1 model=secret-model failed"
            )

    class AuditSink:
        def log(self, **_kwargs):
            return None

    monkeypatch.setattr(chat_routes, "get_runtime_pipeline", lambda **_kwargs: FailingPipeline())
    monkeypatch.setattr(chat_routes, "get_audit_logger", lambda: AuditSink())

    app = FastAPI()
    app.include_router(chat_routes.router)
    app.dependency_overrides[chat_routes.get_current_user] = lambda: type(
        "CurrentUser", (), {"id": "user-1", "department": "finance"}
    )()
    app.dependency_overrides[chat_routes.get_db] = lambda: object()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"query": "private question", "direct": True},
        )

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "10"
    assert response.headers["X-Zhige-Error-Code"] == "generation_backend_unavailable"
    assert "internal-llm" not in response.text
    assert "secret-model" not in response.text
    assert "private question" not in response.text


def test_pipeline_stream_error_does_not_expose_dependency_details():
    from src.rag_pipeline import _safe_stream_error

    event = _safe_stream_error(
        RemoteServiceError(
            "POST http://internal-llm.example/v1 model=private-model failed"
        )
    )

    assert event["event"] == "error"
    assert event["code"] == "generation_backend_unavailable"
    serialized = json.dumps(event, ensure_ascii=False)
    assert "internal-llm" not in serialized
    assert "private-model" not in serialized


def test_database_session_history_is_available_without_process_memory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'sessions.db'}")
    Base.metadata.create_all(
        engine,
        tables=[ChatSession.__table__, ChatMessage.__table__],
    )
    db = sessionmaker(bind=engine)()
    session_id = str(uuid.uuid4())
    db.add(ChatSession(id=session_id, user_id="user-1"))
    db.commit()
    append_message(db, "user-1", session_id, "user", "公司的报销上限是多少？")
    append_message(db, "user-1", session_id, "assistant", "单次上限为 800 元 [1]")

    from api.routes.chat import _load_session_history
    from src.memory import SESSION_MEMORY

    SESSION_MEMORY.clear("user-1", session_id)
    history = _load_session_history(db, "user-1", session_id)
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert history[0]["content"] == "公司的报销上限是多少？"
    db.close()


def test_append_message_rejects_orphan_or_cross_tenant_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'orphan.db'}")
    Base.metadata.create_all(
        engine,
        tables=[ChatSession.__table__, ChatMessage.__table__],
    )
    db = sessionmaker(bind=engine)()
    session_id = str(uuid.uuid4())
    db.add(ChatSession(id=session_id, user_id="owner"))
    db.commit()

    with pytest.raises(KBError):
        append_message(db, "other-user", session_id, "user", "越权写入")
    assert db.query(ChatMessage).count() == 0
    db.close()


def test_ingest_job_state_is_visible_across_real_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    with ingest_jobs._JOBS_LOCK:
        ingest_jobs._JOBS.clear()
        ingest_jobs._FUTURES.clear()
    monkeypatch.setattr(ingest_jobs, "_STATE_LOCK", None)

    create_script = """
import json
import src.ingest_jobs as jobs
job_id = jobs.create_job('file', 'cross-process.txt', {'tenant': 't1'})
jobs._update(job_id, status='running', stage='ocr', progress=0.4)
print(json.dumps({'job_id': job_id}))
"""
    env = {**os.environ, "RAG_DATA_DIR": str(tmp_path)}
    created = subprocess.run(
        [sys.executable, "-c", create_script],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    job_id = json.loads(created.stdout.strip().splitlines()[-1])["job_id"]

    job = ingest_jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "running"
    assert job["stage"] == "ocr"
    assert job["meta"] == {"tenant": "t1"}

    ingest_jobs._update(job_id, status="done", stage="done", progress=1.0)
    read_script = f"""
import json
import src.ingest_jobs as jobs
print(json.dumps(jobs.get_job({job_id!r})))
"""
    observed = subprocess.run(
        [sys.executable, "-c", read_script],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    remote_job = json.loads(observed.stdout.strip().splitlines()[-1])
    assert remote_job["status"] == "done"
    assert remote_job["owner"] != ingest_jobs._OWNER_ID
