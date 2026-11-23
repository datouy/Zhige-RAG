"""知识库对象级授权：所有者、部门共享、显式成员和越权 ID。"""
from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


PASSWORD = "Strong-Pass-2026!"


def _user(db, role: str = "member", department: str = "研发部"):
    from src.security import create_user

    suffix = uuid.uuid4().hex[:8]
    return create_user(
        db,
        username=f"acl-{suffix}",
        email=f"acl-{suffix}@example.test",
        password=PASSWORD,
        role=role,
        department=department,
    )


def test_kb_department_and_member_object_acl(monkeypatch):
    from api.main import app
    from api.deps import get_runtime_config
    from src.db.database import SessionLocal, init_db
    from src.kb_service import (
        can_access_kb,
        create_kb,
        create_session,
        set_kb_sharing,
    )
    from src.security import create_session as issue_session

    init_db()
    monkeypatch.setenv("AUTH_MODE", "enterprise")
    db = SessionLocal()
    try:
        owner = _user(db, department="研发部")
        colleague = _user(db, department="研发部")
        outsider = _user(db, department="财务部")
        admin = _user(db, role="admin", department="管理部")
        kb = create_kb(db, owner, f"产品手册-{uuid.uuid4().hex[:6]}", "", get_runtime_config())
        chat_session = create_session(db, owner.id, "越权测试", kb.id)

        assert can_access_kb(db, colleague, kb, "read") is False
        assert can_access_kb(db, outsider, kb, "read") is False
        assert can_access_kb(db, admin, kb, "write") is True

        set_kb_sharing(db, owner, kb.id, visibility="department")
        assert can_access_kb(db, colleague, kb, "read") is True
        assert can_access_kb(db, colleague, kb, "write") is False
        assert can_access_kb(db, outsider, kb, "read") is False

        set_kb_sharing(
            db,
            owner,
            kb.id,
            members=[{"user_id": outsider.id, "permission": "write"}],
        )
        assert can_access_kb(db, outsider, kb, "write") is True

        colleague_token, _ = issue_session(db, colleague)
        outsider_token, _ = issue_session(db, outsider)
        kb_id = kb.id
        chat_session_id = chat_session.id
        colleague_headers = {"Authorization": f"Bearer {colleague_token}"}
        outsider_headers = {"Authorization": f"Bearer {outsider_token}"}
    finally:
        db.close()

    client = TestClient(app)
    private_id, private_owner_token = create_one_private_kb_for_route_test()
    owner_private = client.get(
        f"/api/v1/kb/{private_id}",
        headers={"Authorization": f"Bearer {private_owner_token}"},
    )
    assert owner_private.status_code == 200
    assert client.get(f"/api/v1/kb/{private_id}", headers=colleague_headers).status_code in {400, 404}

    assert client.get(f"/api/v1/kb/{kb_id}", headers=colleague_headers).status_code == 200
    assert client.get(f"/api/v1/kb/{kb_id}", headers=outsider_headers).status_code == 200
    assert client.patch(
        f"/api/v1/kb/{kb_id}",
        headers=outsider_headers,
        json={"description": "不应成功"},
    ).status_code == 400
    assert client.get(
        f"/api/v1/sessions/{chat_session_id}", headers=outsider_headers
    ).status_code == 404


def create_one_private_kb_for_route_test():
    from api.deps import get_runtime_config
    from src.db.database import SessionLocal
    from src.kb_service import create_kb
    from src.security import create_session

    db = SessionLocal()
    try:
        owner = _user(db, department="研发部")
        kb = create_kb(db, owner, f"私有库-{uuid.uuid4().hex[:6]}", "", get_runtime_config())
        token, _ = create_session(db, owner)
        return kb.id, token
    finally:
        db.close()
