"""企业认证、停用账号和角色职责分离测试。"""
from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def _password() -> str:
    return "Strong-Pass-2026!"


def _create_user(role: str = "member"):
    from src.db.database import SessionLocal, init_db
    from src.security import create_user

    init_db()
    suffix = uuid.uuid4().hex[:8]
    db = SessionLocal()
    try:
        user = create_user(
            db,
            username=f"user-{suffix}",
            email=f"user-{suffix}@example.test",
            password=_password(),
            role=role,
            department="研发部",
        )
        return user.id, user.username
    finally:
        db.close()


def test_password_hash_is_salted_and_verifiable():
    from src.security import hash_password, verify_password

    first = hash_password(_password())
    second = hash_password(_password())
    assert first != second
    assert verify_password(_password(), first) is True
    assert verify_password("wrong-password", first) is False
    assert _password() not in first


def test_enterprise_login_me_logout_and_disable(monkeypatch):
    from api.main import app
    from src.db.database import SessionLocal
    from src.db.models import User
    from src.security import revoke_user_sessions

    user_id, username = _create_user("member")

    monkeypatch.setenv("AUTH_MODE", "enterprise")
    client = TestClient(app)

    assert client.get("/api/v1/auth/me").status_code == 401
    login = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": _password()},
    )
    assert login.status_code == 200, login.text
    payload = login.json()
    assert payload["user"]["role"] == "member"
    assert payload["access_token"].startswith("zhg_")

    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["username"] == username

    denied = client.get("/api/v1/admin/users")
    assert denied.status_code == 403

    db = SessionLocal()
    try:
        revoke_user_sessions(db, user_id)
    finally:
        db.close()
    assert client.get("/api/v1/auth/me").status_code == 401

    relogin = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": _password()},
    )
    assert relogin.status_code == 200
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        user.is_active = False
        db.commit()
    finally:
        db.close()
    assert client.get("/api/v1/auth/me").status_code == 401


def test_admin_can_create_and_reset_users(monkeypatch):
    from api.main import app

    _admin_id, admin_username = _create_user("admin")
    suffix = uuid.uuid4().hex[:8]
    monkeypatch.setenv("AUTH_MODE", "enterprise")
    client = TestClient(app)

    login = client.post(
        "/api/v1/auth/login",
        json={"username": admin_username, "password": _password()},
    )
    assert login.status_code == 200, login.text
    created = client.post(
        "/api/v1/admin/users",
        json={
            "username": f"member-{suffix}",
            "email": f"member-{suffix}@example.test",
            "password": "Another-Strong-2026!",
            "role": "member",
            "department": "财务部",
        },
    )
    assert created.status_code == 201, created.text
    member_id = created.json()["user"]["id"]
    users = client.get("/api/v1/admin/users")
    assert users.status_code == 200
    assert any(item["id"] == member_id for item in users.json()["users"])
    reset = client.post(
        f"/api/v1/admin/users/{member_id}/reset-password",
        json={"new_password": "Reset-Strong-2026!"},
    )
    assert reset.status_code == 200, reset.text


def test_bootstrap_admin_has_no_default_password(monkeypatch):
    import pytest

    from src.db.database import SessionLocal, init_db
    from src.security import AuthError, ensure_bootstrap_admin

    monkeypatch.setenv("AUTH_MODE", "enterprise")
    monkeypatch.delenv("BOOTSTRAP_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("BOOTSTRAP_ADMIN_PASSWORD_FILE", raising=False)
    init_db()
    db = SessionLocal()
    try:
        # Other tests may already have admin credentials; force the dedicated path.
        from src.db.models import AuthCredential, User

        db.query(AuthCredential).delete()
        db.query(User).filter(User.username != "local").update(
            {User.role: "member", User.is_admin: False}, synchronize_session=False
        )
        db.commit()
        with pytest.raises(AuthError, match="BOOTSTRAP_ADMIN_PASSWORD"):
            ensure_bootstrap_admin(db)
    finally:
        db.close()


def test_local_mode_rejects_non_loopback_client(monkeypatch):
    from api.main import app

    monkeypatch.setenv("AUTH_MODE", "local")
    monkeypatch.setenv("AUTH_ALLOW_TESTCLIENT", "0")
    response = TestClient(app).get("/api/v1/kb")
    assert response.status_code == 403
    assert "回环" in response.json()["message"]
