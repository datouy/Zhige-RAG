"""feedback 路由测试：反馈提交 / 统计 / 差评导出评估集 / 长期记忆管理。

不加载 LLM：用最小 FastAPI 应用只挂 feedback 路由，并覆盖认证与 DB 依赖。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.database import get_db as _get_db
from src.db.models import Base, User
from src.middleware.auth import get_current_user as _get_current_user


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False)

    user = User(id="u-1", username="alice", email="a@b.c", is_active=True, is_admin=False)
    admin = User(id="u-admin", username="root", email="r@b.c", is_active=True, is_admin=True)

    app = FastAPI()
    from api.routes.feedback import router

    app.include_router(router)

    def _override_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[_get_db] = _override_db
    app.dependency_overrides[_get_current_user] = lambda: user

    # 导出路径指向临时目录，避免污染项目 data/
    import api.routes.feedback as fb

    monkeypatch.setattr(fb, "resolve_path", lambda p: tmp_path / p)

    with TestClient(app) as c:
        c.extra_user = user
        c.extra_admin = admin
        c.extra_override = app.dependency_overrides
        yield c


class TestSubmitFeedback:
    def test_submit_ok(self, client):
        resp = client.post(
            "/api/v1/feedback",
            json={"query": "年假有几天", "answer": "5 天起 [1]", "rating": "helpful"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_invalid_rating_rejected(self, client):
        resp = client.post("/api/v1/feedback", json={"query": "q", "rating": "meh"})
        assert resp.status_code == 422

    def test_requires_auth(self, client):
        # 仅移除认证覆盖（保留 DB 覆盖），真实 get_current_user 会因无 token 拒绝
        client.extra_override.pop(_get_current_user)
        resp = client.post("/api/v1/feedback", json={"query": "q", "rating": "helpful"})
        assert resp.status_code in (401, 403)


class TestStats:
    def test_stats_and_isolation(self, client):
        client.post("/api/v1/feedback", json={"query": "q1", "rating": "helpful"})
        client.post(
            "/api/v1/feedback",
            json={"query": "q2", "rating": "not_helpful", "correction": "应该是 10 天", "question_type": "假期"},
        )
        stats = client.get("/api/v1/feedback/stats").json()
        assert stats["total"] == 2
        assert stats["helpful"] == 1
        assert stats["not_helpful"] == 1
        assert stats["bad_rate"] == 0.5
        assert stats["unresolved_bad"] == 1
        assert stats["bad_by_type"] == {"假期": 1}


class TestExport:
    def test_export_bad_with_correction(self, client, tmp_path):
        client.post("/api/v1/feedback", json={"query": "q-good", "rating": "helpful"})
        client.post("/api/v1/feedback", json={"query": "q-bad", "rating": "not_helpful"})
        client.post(
            "/api/v1/feedback",
            json={"query": "q-bad2", "rating": "not_helpful", "correction": "正确答案"},
        )
        resp = client.post("/api/v1/eval/export-feedback")
        assert resp.status_code == 200
        data = resp.json()
        # 默认 only_with_correction=True → 只导出带纠错的差评
        assert data["exported"] == 1
        out = Path(data["path"])
        assert out.exists()
        line = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert line["question"] == "q-bad2"
        assert line["reference_answer"] == "正确答案"

        # 已导出的差评被标记 resolved，重复导出为 0
        again = client.post("/api/v1/eval/export-feedback").json()
        assert again["exported"] == 0

    def test_export_without_correction(self, client):
        client.post("/api/v1/feedback", json={"query": "q-bad", "rating": "not_helpful"})
        data = client.post("/api/v1/eval/export-feedback?only_with_correction=false").json()
        assert data["exported"] == 1


class TestMemoryAPI:
    def test_upsert_list_delete(self, client):
        r = client.post("/api/v1/memory", json={"key": "部门", "value": "研发部"})
        assert r.status_code == 200
        # upsert 覆盖
        client.post("/api/v1/memory", json={"key": "部门", "value": "市场部"})
        items = client.get("/api/v1/memory").json()["items"]
        assert len(items) == 1
        assert items[0]["value"] == "市场部"
        # 删除
        r = client.delete("/api/v1/memory/部门")
        assert r.json()["deleted"] == 1
        assert client.get("/api/v1/memory").json()["total"] == 0
