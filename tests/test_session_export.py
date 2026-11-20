"""会话 Markdown 导出：身份边界、引用与空会话行为。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def _new_session(client: TestClient, title: str = "导出测试") -> dict:
    response = client.post("/api/v1/sessions", json={"title": title})
    assert response.status_code == 201
    return response.json()


def test_export_markdown_contains_messages_and_sources(client):
    session = _new_session(client)
    client.post(
        f"/api/v1/sessions/{session['id']}/messages",
        json={"role": "user", "content": "如何配置备份？"},
    )
    client.post(
        f"/api/v1/sessions/{session['id']}/messages",
        json={
            "role": "assistant",
            "content": "每天自动备份。",
            "sources": [
                {
                    "title": "运维手册",
                    "source": "ops/manual.md",
                    "snippet": "备份每日执行。",
                }
            ],
        },
    )

    response = client.get(f"/api/v1/sessions/{session['id']}/export.md")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "attachment;" in response.headers["content-disposition"]
    assert "filename*=UTF-8''" in response.headers["content-disposition"]
    assert "# 导出测试" in response.text
    assert "## 用户" in response.text
    assert "如何配置备份？" in response.text
    assert "## 助手" in response.text
    assert "每天自动备份。" in response.text
    assert "### 引用" in response.text
    assert "运维手册" in response.text
    assert "ops/manual.md" in response.text
    assert "> 备份每日执行。" in response.text


def test_export_empty_session_is_stable(client):
    session = _new_session(client, "空对话")

    response = client.get(f"/api/v1/sessions/{session['id']}/export.md")

    assert response.status_code == 200
    assert "# 空对话" in response.text
    assert "暂无消息。" in response.text


def test_export_unknown_session_returns_404(client):
    response = client.get("/api/v1/sessions/not-found/export.md")
    assert response.status_code == 404
