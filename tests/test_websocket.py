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


def test_websocket_chat_rejects_invalid_token(client):
    """token 无效时 WS 应拒绝连接（4401），不再匿名落到全局知识库。"""
    from starlette.testclient import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/chat?token=garbage"):
            pass


def test_websocket_agent_rejects_invalid_token(client):
    """Agent WS 同样拒绝无效 token。"""
    from starlette.testclient import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/v1/ws/agent?token=garbage"):
            pass


def test_websocket_chat_no_token_param_rejected(client):
    """不带 token 参数也应被拒绝（不再允许匿名会话）。"""
    from starlette.testclient import WebSocketDisconnect

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
