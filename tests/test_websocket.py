"""WebSocket 路由测试。

服务不需要登录：WS 连接不校验 token，空 query 返回 error 事件而不是崩溃。

不实际起 LLM —— 只验证连接、参数容错与 Agent 事件流。
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    from api.main import app

    return TestClient(app)


# ----------------------------------------------------------------------
#  /ws/chat
# ----------------------------------------------------------------------


def test_websocket_chat_connects_without_token(client):
    """无需 token 即可建立 /ws/chat 连接。"""
    with client.websocket_connect("/ws/chat") as ws:
        assert ws is not None


def test_websocket_chat_empty_query_returns_error(client):
    """空 query 应返回 error 事件，而不是抛异常。"""
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"query": ""})
        msg = ws.receive_json()
        assert msg.get("event") == "error"
        assert "required" in msg.get("data", "")


def test_websocket_chat_ignores_legacy_token_param(client):
    """老前端可能仍带 ?token=…：多余参数应被忽略，连接照常建立。"""
    with client.websocket_connect("/ws/chat?token=garbage") as ws:
        ws.send_json({"query": ""})
        msg = ws.receive_json()
        assert msg.get("event") == "error"


# ----------------------------------------------------------------------
#  /api/v1/ws/agent
# ----------------------------------------------------------------------


class _FakeAgent:
    """假 Agent：不加载 LLM，直接产出流式事件。"""

    max_steps = 5
    llm = None
    tools: list = []
    decision_provider = None

    def run(self, query, stream=True):
        yield {"event": "token", "data": "ok"}
        yield {"event": "done", "data": []}


def test_websocket_agent_allows_anonymous(client):
    """Agent WS 同样不要求 token，且空 query 返回 error 事件。"""
    with client.websocket_connect("/api/v1/ws/agent") as ws:
        ws.send_json({"query": ""})
        msg = ws.receive_json()
        assert msg.get("event") == "error"
        assert "required" in msg.get("data", "")


def test_websocket_agent_streams(client, monkeypatch):
    """有效消息应正常产出事件（并发闸代码路径可走通）。"""
    import api.routes.agent as agent_module

    monkeypatch.setattr(agent_module, "_get_agent", lambda user_id=None: _FakeAgent())
    with client.websocket_connect("/api/v1/ws/agent") as ws:
        ws.send_json({"query": "你好"})
        events = []
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev.get("event") in ("done", "error"):
                break
        types = [e.get("event") for e in events]
        assert "error" not in types, f"不应出现 error 事件：{events}"
        assert "done" in types


# ----------------------------------------------------------------------
#  登录 / 注册相关模块应已彻底移除
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "src.auth",
        "src.auth.jwt_handler",
        "src.auth.password",
        "src.password_validator",
        "api.routes.auth",
    ],
)
def test_auth_modules_are_gone(module):
    """这些模块必须导入失败，避免登录注册代码被悄悄带回来。"""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize(
    "relpath",
    [
        "src/auth",
        "src/password_validator.py",
        "api/routes/auth.py",
        "check_auth_mode.py",
    ],
)
def test_auth_files_deleted(relpath):
    """文件层面也必须确实不存在（空的 __init__.py 残留会让包又能被导入）。"""
    assert not (ROOT / relpath).exists(), f"{relpath} 应已被删除"
