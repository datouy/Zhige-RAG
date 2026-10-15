"""WebSocket 路由的 token 解码路径测试（P3.2）。

WS 测试本身依赖实际 LLM/向量库比较重（需要 accept + 全双工客户端）。本测试覆盖：
- ``src.auth.jwt_handler.decode_token`` 路径能正确从 query param ``token`` 解出 user_id。
- 错误 / 过期 token 解码后 user_id 为 None，调用方仍能继续往下走（WS 路由设计如此）。
- WS handler 在 connect 后立刻收到 ``query=""`` 时返回 ``{"event":"error", "data":"query is required"}`` 而不是崩溃。

不实际起 LLM —— 我们只验证：1) token 校验；2) query 缺失容错。
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient

from src.auth import jwt_handler


# ----------------------------------------------------------------------
#  Token decode round-trip（独立、不依赖 app）
# ----------------------------------------------------------------------


def _make_token(user_id: str, type_: str, expires_in_minutes: int = 30) -> str:
    return jwt.encode(
        {
            "sub": user_id,
            "exp": datetime.utcnow() + timedelta(minutes=expires_in_minutes),
            "iat": datetime.utcnow(),
            "type": type_,
        },
        jwt_handler.SECRET_KEY,
        algorithm=jwt_handler.ALGORITHM,
    )


def test_decode_valid_token_yields_user_id():
    token = _make_token("user-1", "access")
    payload = jwt_handler.decode_token(token)
    assert payload is not None
    assert payload["sub"] == "user-1"
    assert payload["type"] == "access"


def test_decode_expired_token_returns_none():
    token = _make_token("user-1", "access", expires_in_minutes=-1)
    assert jwt_handler.decode_token(token) is None


def test_decode_garbage_returns_none():
    assert jwt_handler.decode_token("not-a-jwt") is None
    assert jwt_handler.decode_token("") is None


def test_decode_wrong_signature_returns_none():
    bad = jwt.encode(
        {"sub": "u", "exp": datetime.utcnow() + timedelta(hours=1)},
        "wrong-secret",
        algorithm="HS256",
    )
    assert jwt_handler.decode_token(bad) is None


# ----------------------------------------------------------------------
#  WebSocket handler integration
# ----------------------------------------------------------------------


@pytest.fixture
def client():
    from api.main import app

    return TestClient(app)


def test_websocket_chat_rejects_empty_query(client):
    """打开 ``/ws/chat``，立刻发 ``{"query": ""}`` 应收到 ``error`` 事件，而不是抛异常。

    P5 收紧后：WS 与 REST 认证语义一致——有效签名但用户不存在的 token
    会被拒绝（4403），因此这里必须先创建真实用户。
    """
    from src.db.database import SessionLocal
    from src.db.models import User

    db = SessionLocal()
    try:
        uid = "ws-empty-query-user"
        if db.query(User).filter(User.id == uid).first() is None:
            db.add(User(id=uid, username="ws_empty_query_user", email="ws_empty_query_user@example.com",
                        password_hash="x", is_active=True))
            db.commit()
    finally:
        db.close()

    token = _make_token(uid, "access")
    with client.websocket_connect(f"/ws/chat?token={token}") as ws:
        ws.send_json({"query": ""})
        msg = ws.receive_json()
        assert msg.get("event") == "error"
        assert "required" in msg.get("data", "")


def test_websocket_chat_rejects_invalid_token(client, monkeypatch):
    """启用认证时，token 无效的 WS 连接应被拒绝（4401）。

    默认是本地单用户模式（不校验 token，匿名落到本地用户），
    因此这里显式开启认证来测拒绝路径。
    """
    from starlette.testclient import WebSocketDisconnect

    monkeypatch.setenv("AUTH_ENABLED", "1")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/chat?token=garbage"):
            pass


def test_websocket_agent_rejects_invalid_token(client, monkeypatch):
    """Agent WS 同样拒绝无效 token（认证策略与 /ws/chat 保持一致）。"""
    from starlette.testclient import WebSocketDisconnect

    monkeypatch.setenv("AUTH_ENABLED", "1")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/v1/ws/agent?token=garbage"):
            pass


def test_websocket_chat_no_token_param_rejected(client, monkeypatch):
    """启用认证时，不带 token 参数也应被拒绝。"""
    from starlette.testclient import WebSocketDisconnect

    monkeypatch.setenv("AUTH_ENABLED", "1")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/chat"):
            pass


def test_websocket_chat_valid_token_then_empty_query(client):
    """有效 token 连接后，空 query 返回 error 事件。"""
    from src.db.database import SessionLocal
    from src.db.models import User

    # 直接建一个测试用户拿真实 token（不经过限流的注册端点）
    db = SessionLocal()
    try:
        uid = "ws-valid-user"
        if db.query(User).filter(User.id == uid).first() is None:
            db.add(User(id=uid, username="ws_valid_user", email="ws_valid_user@example.com",
                        password_hash="x", is_active=True))
            db.commit()
    finally:
        db.close()

    token = _make_token(uid, "access")
    with client.websocket_connect(f"/ws/chat?token={token}") as ws:
        ws.send_json({"query": ""})
        msg = ws.receive_json()
        assert msg.get("event") == "error"
        assert "required" in msg.get("data", "")


# ----------------------------------------------------------------------
#  Agent WS：与 /ws/chat 的认证策略保持一致
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


def test_websocket_agent_allows_local_mode(client):
    """本地单用户模式（默认）下，Agent WS 不要求 token —— 与 /ws/chat 一致。

    回归防护：此前 agent WS 无条件要求 token，本地模式下前端拿不到 token，
    连接会被 4401 直接拒绝，Agent 功能形同不可用。
    """
    with client.websocket_connect("/api/v1/ws/agent") as ws:
        ws.send_json({"query": ""})
        msg = ws.receive_json()
        assert msg.get("event") == "error"
        assert "required" in msg.get("data", "")


def test_websocket_agent_streams_without_nameerror(client, monkeypatch):
    """有效消息应正常产出事件（并发闸代码路径可走通）。

    回归防护：P0-1 删除了 AsyncLLMExecutor，但路由里残留 ``executor`` 引用，
    导致首条消息即 NameError。此处用假 Agent 走通那段代码。
    """
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
