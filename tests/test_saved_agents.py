"""保存 Agent：CRUD、知识范围与提示词行为。"""
from __future__ import annotations

from fastapi.testclient import TestClient
import pytest


@pytest.fixture()
def client():
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def _default_kb(client: TestClient) -> dict:
    response = client.get("/api/v1/kb")
    assert response.status_code == 200
    return response.json()["knowledge_bases"][0]


def test_saved_agent_crud_keeps_scope_and_tool_whitelist(client):
    kb = _default_kb(client)
    response = client.post(
        "/api/v1/agents",
        json={
            "name": "运维",
            "kb_id": kb["id"],
            "instructions": "先查运行手册，再回答。",
            "allowed_tools": ["search_documents", "search_documents", "bad name"],
            "max_steps": 3,
        },
    )
    assert response.status_code == 201
    agent = response.json()
    assert agent["kb_id"] == kb["id"]
    assert agent["allowed_tools"] == ["search_documents"]
    assert agent["max_steps"] == 3

    listed = client.get("/api/v1/agents").json()
    assert listed["total"] == 1
    assert listed["agents"][0]["id"] == agent["id"]

    updated = client.patch(
        f"/api/v1/agents/{agent['id']}",
        json={"instructions": "只回答手册里的内容。", "allowed_tools": []},
    )
    assert updated.status_code == 200
    assert updated.json()["allowed_tools"] == []

    deleted = client.delete(f"/api/v1/agents/{agent['id']}")
    assert deleted.status_code == 200
    assert client.get(f"/api/v1/agents/{agent['id']}").status_code == 404


def test_saved_agent_rejects_unknown_kb_and_duplicate_name(client):
    kb = _default_kb(client)
    payload = {"name": "研究", "kb_id": kb["id"], "allowed_tools": []}
    assert client.post("/api/v1/agents", json=payload).status_code == 201
    assert client.post("/api/v1/agents", json=payload).status_code == 409

    payload.update({"name": "另一个", "kb_id": "missing"})
    assert client.post("/api/v1/agents", json=payload).status_code == 400


def test_agent_prompt_includes_saved_instructions():
    from src.agent import BuiltinTools, ReActAgent

    class _LLM:
        pass

    agent = ReActAgent(
        llm=_LLM(),
        tools=BuiltinTools.create(include=[]),
        system_prompt="回答前列出检索到的文档。",
    )
    assert "回答前列出检索到的文档。" in agent._build_system_prompt()
    assert "回答前列出检索到的文档。" in agent._build_fc_system_prompt()
