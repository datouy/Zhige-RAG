"""``api.routes.auth`` 与 ``src.auth.jwt_handler`` 的最小烟雾测试（P3.2）。

目标：
- JWT 编码/解码（access vs refresh，type 校验）。
- 过期/无效 token 的解码返回 None。
- 用 ``TestClient`` + ``dependency_overrides`` 注入 SQLite in-memory，跑
  ``register → login → me → refresh → logout`` 完整链路，并验证错误路径
  （无效/过期 token）返回 401。

注意：不触发 rate-limiter / audit（仅覆盖 happy path 与 token 错误）。
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.auth import jwt_handler
from src.db.database import get_db
from src.db.models import Base, User


# ----------------------------------------------------------------------
#  Pure-JWT tests（无 DB 依赖）
# ----------------------------------------------------------------------


def test_decode_token_returns_none_for_invalid():
    """空字符串、垃圾字符、签名错误都应返回 None。"""
    assert jwt_handler.decode_token("") is None
    assert jwt_handler.decode_token("garbage-token") is None
    # 篡改签名
    bad = jwt.encode({"sub": "u", "exp": datetime.utcnow() + timedelta(hours=1)}, "wrong-secret", algorithm="HS256")
    assert jwt_handler.decode_token(bad) is None


def test_decode_token_returns_none_for_expired():
    """过期 token 应被 jwt_handler 捕获并返回 None。"""
    expired = jwt.encode(
        {
            "sub": "u1",
            "exp": datetime.utcnow() - timedelta(seconds=1),
            "iat": datetime.utcnow() - timedelta(hours=1),
            "type": "access",
        },
        jwt_handler.SECRET_KEY,
        algorithm=jwt_handler.ALGORITHM,
    )
    assert jwt_handler.decode_token(expired) is None


def test_create_access_token_decodes_with_type_access():
    token = jwt_handler.create_access_token("user-123")
    payload = jwt_handler.decode_token(token)
    assert payload is not None
    assert payload["sub"] == "user-123"
    assert payload["type"] == "access"


def test_create_refresh_token_decodes_with_type_refresh():
    token = jwt_handler.create_refresh_token("user-456")
    payload = jwt_handler.decode_token(token)
    assert payload["sub"] == "user-456"
    assert payload["type"] == "refresh"


# ----------------------------------------------------------------------
#  Integration：register → login → me → refresh → logout
# ----------------------------------------------------------------------


@pytest.fixture
def client_with_db(monkeypatch, tmp_path):
    """构造一个带 SQLite in-memory DB 的 FastAPI TestClient，并把 ``get_db``
    替换为本次测试的 session。"""
    # 强制 audit logger 走 stub（不写文件）
    monkeypatch.setattr("src.middleware.audit.AuditLogger", MagicMock())

    # 用临时 SQLite 文件，避免多连接问题
    db_path = tmp_path / "users.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # 关掉 rate limiter（slowapi）以免被限速阻塞测试
    from api.routes import auth as auth_module

    # slowapi: 直接清空 limiter
    if hasattr(auth_module, "limiter"):
        auth_module.limiter.enabled = False

    from api.main import app
    from api.routes import auth as auth_routes_module

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        yield client, TestingSessionLocal

    app.dependency_overrides.clear()


def _register_payload(suffix="0"):
    return {
        "username": f"tester{suffix}",
        "email": f"tester{suffix}@example.com",
        "password": "S3cret-Pass!",
    }


def test_register_login_me_refresh_logout_flow(client_with_db):
    client, SessionLocal = client_with_db

    # 1. register
    r = client.post("/api/v1/auth/register", json=_register_payload("a"))
    assert r.status_code == 201, r.text
    user = r.json()
    assert user["username"] == "testera"
    assert user["email"] == "testera@example.com"

    # 2. login
    r = client.post(
        "/api/v1/auth/login",
        json={"login": "testera", "password": "S3cret-Pass!"},
    )
    assert r.status_code == 200, r.text
    tokens = r.json()
    assert tokens["token_type"] == "bearer"
    assert tokens["access_token"]
    assert tokens["refresh_token"]
    access = tokens["access_token"]
    refresh = tokens["refresh_token"]

    # 3. me
    r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access}"})
    assert r.status_code == 200
    me = r.json()
    assert me["username"] == "testera"

    # 4. refresh
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 200
    refreshed = r.json()
    assert refreshed["access_token"]
    assert refreshed["refresh_token"]

    # 5. logout（用刚刷新的 refresh token + 任意 access token）
    r = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": refreshed["refresh_token"]},
        headers={"Authorization": f"Bearer {access}"},
    )
    assert r.status_code == 200


def test_register_duplicate_username_returns_400(client_with_db):
    client, _ = client_with_db
    r1 = client.post("/api/v1/auth/register", json=_register_payload("dup"))
    assert r1.status_code == 201
    r2 = client.post("/api/v1/auth/register", json=_register_payload("dup"))
    assert r2.status_code == 400


def test_login_wrong_password_returns_401(client_with_db):
    client, _ = client_with_db
    client.post("/api/v1/auth/register", json=_register_payload("wp"))
    r = client.post(
        "/api/v1/auth/login",
        json={"login": "testerwp", "password": "wrong-password"},
    )
    assert r.status_code == 401


def test_me_without_token_returns_401_or_403(client_with_db):
    client, _ = client_with_db
    r = client.get("/api/v1/auth/me")
    # HTTPBearer 在缺失 header 时返回 403
    assert r.status_code in (401, 403)


def test_me_with_invalid_token_returns_401(client_with_db):
    client, _ = client_with_db
    r = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid.token.value"},
    )
    assert r.status_code == 401


def test_refresh_with_invalid_token_returns_401(client_with_db):
    client, _ = client_with_db
    r = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": "garbage"},
    )
    assert r.status_code == 401


def test_refresh_with_access_token_type_rejected(client_with_db):
    """把 access token 当 refresh token 用，type 校验失败应 401。"""
    client, _ = client_with_db
    client.post("/api/v1/auth/register", json=_register_payload("mix"))
    r = client.post(
        "/api/v1/auth/login",
        json={"login": "testermix", "password": "S3cret-Pass!"},
    )
    access_token = r.json()["access_token"]
    r = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": access_token},
    )
    assert r.status_code == 401
