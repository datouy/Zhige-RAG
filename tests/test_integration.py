"""集成测试套件 - 测试核心业务流程。

测试覆盖：
1. 认证流程：注册 -> 登录 -> 访问受保护端点
2. 知识库 CRUD：创建 KG -> 添加实体 -> 查询 -> 删除
3. RAG 查询：文档入库 -> 查询 -> 验证响应

运行测试：
    pytest tests/test_integration.py -v -s
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["STREAMLIT_RUNTIME"] = "enable"


class _FakePipeline:
    """轻量替身：覆盖 answer / astream_answer，避免测试真实加载 LLM。"""

    # chat 路由会探测长期记忆存储；None = 不可用，路由须容忍
    long_term_store = None

    def answer(self, question, top_k=None, where=None, stream=False, history=None, user_id=None, allowed_groups=None):
        from src.rag_pipeline import RAGResult

        return RAGResult(answer="集成测试回答", sources=[], raw_hits=[], timings={})

    async def astream_answer(self, question, top_k=None, history=None, user_id=None, allowed_groups=None):
        yield {"event": "hits", "data": []}
        yield {"event": "token", "data": "集成"}
        yield {"event": "token", "data": "测试回答"}
        yield {"event": "done", "data": []}


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """创建 FastAPI 测试客户端。

    - 关闭限流（注册 3/min、登录 5/min 会干扰连续注册测试用户）。
    - 打桩 ``get_runtime_pipeline``，问答/WS 测试不加载真实 LLM。
    """
    from api.main import app
    from api.routes.auth import limiter
    import api.routes.chat as chat_module

    limiter.enabled = False
    original = chat_module.get_runtime_pipeline
    chat_module.get_runtime_pipeline = lambda user_id=None: _FakePipeline()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        chat_module.get_runtime_pipeline = original


@pytest.fixture(scope="function")
def test_user() -> dict:
    """生成测试用户数据。"""
    unique_id = uuid.uuid4().hex[:8]
    return {
        "username": f"testuser_{unique_id}",
        "email": f"test_{unique_id}@example.com",
        "password": "TestPassword123!",
    }


@pytest.fixture(scope="function")
def authenticated_client(
    client: TestClient,
    test_user: dict,
) -> Generator[tuple[TestClient, str, dict], None, None]:
    """创建已认证的客户端（注册并登录）。"""
    register_response = client.post("/api/v1/auth/register", json=test_user)
    assert register_response.status_code == 201, f"注册失败: {register_response.json()}"

    login_response = client.post(
        "/api/v1/auth/login",
        json={"login": test_user["username"], "password": test_user["password"]},
    )
    assert login_response.status_code == 200, f"登录失败: {login_response.json()}"

    tokens = login_response.json()
    access_token = tokens["access_token"]

    headers = {"Authorization": f"Bearer {access_token}"}

    yield client, access_token, {**test_user, "headers": headers}


class TestAuthFlow:
    """认证流程测试。"""

    def test_register(self, client: TestClient, test_user: dict):
        """测试用户注册。"""
        response = client.post("/api/v1/auth/register", json=test_user)
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == test_user["username"]
        assert data["email"] == test_user["email"]
        assert "id" in data
        assert data["subscription_tier"] == "free"
        assert data["is_active"] is True

    def test_register_duplicate_username(self, client: TestClient, test_user: dict):
        """测试重复用户名注册应失败。"""
        client.post("/api/v1/auth/register", json=test_user)
        response = client.post("/api/v1/auth/register", json=test_user)
        assert response.status_code == 400
        assert "用户名已存在" in response.json().get("detail", "")

    def test_register_duplicate_email(self, client: TestClient):
        """测试重复邮箱注册应失败。"""
        unique_id = uuid.uuid4().hex[:8]
        email = f"dup_{unique_id}@example.com"
        user1 = {
            "username": f"user1_{unique_id}",
            "email": email,
            "password": "TestPassword123!",
        }
        user2 = {
            "username": f"user2_{unique_id}",
            "email": email,
            "password": "TestPassword123!",
        }
        client.post("/api/v1/auth/register", json=user1)
        response = client.post("/api/v1/auth/register", json=user2)
        assert response.status_code == 400
        assert "邮箱已被注册" in response.json().get("detail", "")

    def test_login(self, client: TestClient, test_user: dict):
        """测试用户登录。"""
        client.post("/api/v1/auth/register", json=test_user)

        response = client.post(
            "/api/v1/auth/login",
            json={"login": test_user["username"], "password": test_user["password"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_with_email(self, client: TestClient, test_user: dict):
        """测试使用邮箱登录。"""
        client.post("/api/v1/auth/register", json=test_user)

        response = client.post(
            "/api/v1/auth/login",
            json={"login": test_user["email"], "password": test_user["password"]},
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_login_invalid_credentials(self, client: TestClient, test_user: dict):
        """测试无效凭据登录。"""
        client.post("/api/v1/auth/register", json=test_user)

        response = client.post(
            "/api/v1/auth/login",
            json={"login": test_user["username"], "password": "wrongpassword"},
        )
        assert response.status_code == 401
        assert "用户名或密码错误" in response.json().get("detail", "")

    def test_get_me(self, authenticated_client: tuple):
        """测试获取当前用户信息。"""
        client, _, user_data = authenticated_client

        response = client.get("/api/v1/auth/me", headers=user_data["headers"])
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == user_data["username"]
        assert data["email"] == user_data["email"]

    def test_access_protected_without_token(self, client: TestClient):
        """测试无 Token 访问受保护端点应失败。"""
        response = client.get("/api/v1/documents")
        assert response.status_code == 401

    def test_access_protected_with_invalid_token(self, client: TestClient):
        """测试无效 Token 访问受保护端点应失败。"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/documents", headers=headers)
        assert response.status_code == 401

    def test_refresh_token(self, authenticated_client: tuple):
        """测试刷新 Token。"""
        client, _, user_data = authenticated_client

        login_response = client.post(
            "/api/v1/auth/login",
            json={"login": user_data["username"], "password": user_data["password"]},
        )
        refresh_token = login_response.json()["refresh_token"]

        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data


class TestKGCrudFlow:
    """知识图谱 CRUD 流程测试。"""

    def test_kg_stats_empty(self, authenticated_client: tuple):
        """测试空知识图谱统计。"""
        client, _, user_data = authenticated_client

        response = client.get("/api/v1/kg/stats", headers=user_data["headers"])
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "relations" in data
        assert data["entities"] == 0
        assert data["relations"] == 0

    def test_kg_extract_entities(self, authenticated_client: tuple):
        """测试实体抽取。

        实体抽取依赖 LLM；测试环境未加载模型时端点返回 503（合理降级），
        此时只验证降级行为而非抽取结果。
        """
        client, _, user_data = authenticated_client

        test_text = "苹果公司成立于1976年，由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩创立。"

        response = client.post(
            "/api/v1/kg/extract",
            json={"text": test_text, "source_doc": "test_doc", "persist": True},
            headers=user_data["headers"],
        )
        if response.status_code == 503:
            # LLM 不可用（未加载/加载失败）时的降级路径
            return
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "relations" in data
        assert "persisted" in data

    def test_kg_list_entities(self, authenticated_client: tuple):
        """测试列出实体。"""
        client, _, user_data = authenticated_client

        test_text = "OpenAI 是一家专注于人工智能研究的公司。"
        client.post(
            "/api/v1/kg/extract",
            json={"text": test_text, "source_doc": "test"},
            headers=user_data["headers"],
        )

        response = client.get(
            "/api/v1/kg/entities",
            headers=user_data["headers"],
        )
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "total" in data

    def test_kg_search(self, authenticated_client: tuple):
        """测试图谱检索。"""
        client, _, user_data = authenticated_client

        test_text = "人工智能是计算机科学的一个分支。"
        client.post(
            "/api/v1/kg/extract",
            json={"text": test_text, "source_doc": "ai_doc"},
            headers=user_data["headers"],
        )

        response = client.post(
            "/api/v1/kg/search",
            json={"query": "人工智能", "top_k": 5, "hops": 2},
            headers=user_data["headers"],
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "entities" in data
        assert "relations" in data

    def test_kg_list_relations(self, authenticated_client: tuple):
        """测试列出关系。"""
        client, _, user_data = authenticated_client

        response = client.get(
            "/api/v1/kg/relations",
            headers=user_data["headers"],
        )
        assert response.status_code == 200
        data = response.json()
        assert "relations" in data
        assert "total" in data


class TestRAGQueryFlow:
    """RAG 查询流程测试。"""

    def test_list_documents_empty(self, authenticated_client: tuple):
        """测试空文档列表。"""
        client, _, user_data = authenticated_client

        response = client.get("/api/v1/documents", headers=user_data["headers"])
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert "total_chunks" in data
        assert data["total"] == 0

    def test_search_without_results(self, authenticated_client: tuple):
        """测试无结果的检索。"""
        client, _, user_data = authenticated_client

        response = client.post(
            "/api/v1/search",
            json={"query": "这是一个完全不存在的查询关键词xyz123456", "top_k": 4},
            headers=user_data["headers"],
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "hits" in data
        assert "total" in data

    def test_chat_sync(self, authenticated_client: tuple):
        """测试同步问答。"""
        client, _, user_data = authenticated_client

        response = client.post(
            "/api/v1/chat",
            json={"query": "你好，请介绍一下自己", "top_k": 2},
            headers=user_data["headers"],
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["answer"], str)

    def test_chat_empty_query_fails(self, authenticated_client: tuple):
        """测试空查询应返回验证错误。"""
        client, _, user_data = authenticated_client

        response = client.post(
            "/api/v1/chat",
            json={"query": "", "top_k": 4},
            headers=user_data["headers"],
        )
        assert response.status_code == 422


class TestAgentFlow:
    """Agent 流程测试。"""

    def test_list_agent_tools(self, authenticated_client: tuple):
        """测试列出 Agent 工具。"""
        client, _, user_data = authenticated_client

        response = client.get(
            "/api/v1/agent/tools",
            headers=user_data["headers"],
        )
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "total" in data
        assert isinstance(data["tools"], list)

    def test_agent_chat_sync(self, authenticated_client: tuple):
        """测试同步 Agent 调用。"""
        client, _, user_data = authenticated_client

        response = client.post(
            "/api/v1/agent/chat",
            json={"query": "你好", "max_steps": 2},
            headers=user_data["headers"],
        )
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "steps" in data
        else:
            assert response.status_code in (500, 503)


class TestHealthEndpoint:
    """健康检查端点测试。"""

    def test_health_check(self, client: TestClient):
        """测试健康检查端点。"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_readiness_check(self, client: TestClient):
        """测试就绪探针。"""
        response = client.get("/api/ready")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data
        assert "checks" in data

    def test_metrics_endpoint(self, client: TestClient):
        """测试指标端点。

        P5 收紧后 /api/v1/metrics 需要认证（docstring 一直如此声称，
        之前实现漏了 Depends）：匿名访问必须 401。
        """
        response = client.get("/api/v1/metrics")
        assert response.status_code == 401


class TestWebSocketFlow:
    """WebSocket 流程测试。"""

    def test_websocket_chat_connection(self, authenticated_client: tuple):
        """测试 WebSocket 连接。"""
        client, access_token, user_data = authenticated_client

        with client.websocket_connect(f"/ws/chat?token={access_token}") as ws:
            assert ws is not None

    def test_websocket_chat_message(self, authenticated_client: tuple):
        """测试 WebSocket 消息发送和接收。"""
        client, access_token, user_data = authenticated_client

        with client.websocket_connect(f"/ws/chat?token={access_token}") as ws:
            ws.send_json({"query": "你好", "top_k": 2})

            events = []
            try:
                while True:
                    event = ws.receive_json()
                    events.append(event)
                    if event.get("event") == "done":
                        break
            except Exception:
                pass

            assert len(events) > 0
            assert any(e.get("event") == "done" for e in events)
