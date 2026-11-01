import asyncio

import api.routes.system as system_routes
from src.db.database import init_db


class _FailingStore:
    def health_check(self):
        raise RuntimeError("metadata segment mismatch")


class _HealthyStore:
    def health_check(self):
        return {"collections": 1, "chunks": 2, "query_probe": True}


def test_readiness_fails_when_collection_probe_fails(monkeypatch):
    init_db()
    system_routes._clear_readiness_cache()
    monkeypatch.setattr(system_routes, "get_vector_store", lambda: _FailingStore())
    response = asyncio.run(system_routes.readiness_check())
    assert response.status_code == 503
    payload = __import__("json").loads(response.body)
    assert payload["checks"]["chroma"] == "error"
    assert "metadata" not in payload["checks"]["chroma_detail"].get("error", "")


def test_readiness_succeeds_after_real_collection_probe(monkeypatch):
    init_db()
    system_routes._clear_readiness_cache()
    monkeypatch.setattr(system_routes, "get_vector_store", lambda: _HealthyStore())
    response = asyncio.run(system_routes.readiness_check())
    assert response.status_code == 200
    payload = __import__("json").loads(response.body)
    assert payload["checks"]["chroma_detail"]["query_probe"] is True
