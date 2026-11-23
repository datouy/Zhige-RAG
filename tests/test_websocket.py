"""WebSocket 路由测试。

测试运行在 local 认证模式；生产 enterprise 模式必须携带有效会话。
不实际起 LLM —— 只验证连接、参数容错与 Agent 事件流。
"""

from __future__ import annotations

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


def test_websocket_chat_connects_in_loopback_test_mode(client):
    """测试专用回环模式下可建立 /ws/chat 连接。"""
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


def test_websocket_agent_allows_loopback_test_mode(client):
    """local 测试模式下 Agent WS 可连接，空 query 返回 error 事件。"""
    with client.websocket_connect("/api/v1/ws/agent") as ws:
        ws.send_json({"query": ""})
        msg = ws.receive_json()
        assert msg.get("event") == "error"
        assert "required" in msg.get("data", "")


def test_websocket_agent_streams(client, monkeypatch):
    """有效消息应正常产出事件（并发闸代码路径可走通）。"""
    import api.routes.agent as agent_module

    monkeypatch.setattr(
        agent_module,
        "_get_agent",
        lambda user_id=None, allowed_groups=None: _FakeAgent(),
    )
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
