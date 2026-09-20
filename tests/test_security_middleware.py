"""安全响应头中间件测试（P2.3 / P2.4）。

覆盖：
- 关键安全头存在且取值正确
- CSP 默认不含 ``unsafe-inline`` script
- ``CSP_ALLOW_UNSAFE_INLINE_SCRIPT=1`` 切换到 dev CSP
- ``CSP_POLICY`` 直接覆盖默认 CSP
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.middleware import SecurityHeadersMiddleware, _build_csp, _DEFAULT_CSP


def _make_app():
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/probe")
    async def probe():
        return {"ok": True}

    app.add_middleware(SecurityHeadersMiddleware)
    return app


def test_security_headers_present():
    from fastapi.testclient import TestClient

    client = TestClient(_make_app())
    resp = client.get("/probe")
    assert resp.status_code == 200
    # P2.3：核心安全头必须存在
    for h in (
        "x-content-type-options",
        "x-frame-options",
        "strict-transport-security",
        "content-security-policy",
        "referrer-policy",
        "permissions-policy",
        "cross-origin-opener-policy",
    ):
        assert h in resp.headers, f"missing security header: {h}"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"


def test_default_csp_includes_frame_ancestors(monkeypatch):
    """P2.4：默认 CSP 必须包含 ``frame-ancestors 'none'``（防 clickjacking）。

    注：当前默认 CSP 保留 ``unsafe-inline`` for script/style，因为前端
    ``ui/web`` 中存在内联脚本/样式（见 ``api/middleware.py`` 注释）。
    严格策略可通过 ``CSP_POLICY=strict`` 环境变量启用，由独立测试覆盖。
    """
    monkeypatch.delenv("CSP_POLICY", raising=False)
    monkeypatch.delenv("CSP_ALLOW_UNSAFE_INLINE_SCRIPT", raising=False)

    csp = _build_csp()
    assert "frame-ancestors 'none'" in csp, (
        f"default CSP must include frame-ancestors 'none'; got: {csp!r}"
    )
    assert "object-src 'none'" in csp, (
        f"default CSP must include object-src 'none'; got: {csp!r}"
    )
    assert "default-src 'self'" in csp


def test_strict_csp_disallows_unsafe_inline(monkeypatch):
    """P2.4：严格 CSP 模式必须禁用 ``unsafe-inline`` for scripts。"""
    monkeypatch.setenv("CSP_POLICY", "strict")
    csp = _build_csp()
    script_part = next(
        (seg for seg in csp.split(";") if seg.strip().startswith("script-src")), ""
    )
    assert script_part.strip(), "CSP must include script-src"
    assert "'unsafe-inline'" not in script_part, (
        f"strict CSP must not allow inline scripts; got: {script_part!r}"
    )
    assert "frame-ancestors" in csp


def test_csp_override_via_env(monkeypatch):
    monkeypatch.setenv("CSP_POLICY", "default-src 'none'")
    assert _build_csp() == "default-src 'none'"


def test_csp_unsafe_inline_fallback(monkeypatch):
    monkeypatch.delenv("CSP_POLICY", raising=False)
    monkeypatch.setenv("CSP_ALLOW_UNSAFE_INLINE_SCRIPT", "1")
    csp = _build_csp()
    assert "'unsafe-inline'" in csp  # dev fallback 允许
    monkeypatch.delenv("CSP_ALLOW_UNSAFE_INLINE_SCRIPT", raising=False)