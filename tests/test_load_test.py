"""真实压测脚本的结果归类测试。"""
from __future__ import annotations

from scripts import load_test


def test_run_plan_reports_rate_limit_separately_from_errors(monkeypatch):
    def fake_request(_base_url, _method, path, _body, _timeout):
        if path == "/api/v1/search":
            return 12.0, 429, "too many requests"
        return 8.0, 200, ""

    monkeypatch.setattr(load_test, "_request", fake_request)
    plan = [
        ("GET", "/api/health", None),
        ("GET", "/api/health", None),
        ("POST", "/api/v1/search", {"query": "RAG"}),
        ("POST", "/api/v1/search", {"query": "RAG"}),
    ]

    summary = load_test._run_plan("http://test", plan, concurrency=2, timeout=1.0)

    assert summary["status_codes"] == {"200": 2, "429": 2}
    assert summary["rate_limited"] == 2
    assert summary["rate_limited_rate"] == 0.5
    assert summary["error_rate"] == 0.0
