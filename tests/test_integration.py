"""集成测试套件 - 测试核心业务流程。

测试覆盖：
1. 无需登录：受保护端点直接可访问，登录注册端点已移除
2. 知识库 CRUD：创建 KG -> 添加实体 -> 查询 -> 删除
3. RAG 查询：文档入库 -> 查询 -> 验证响应
4. WebSocket 流式问答

运行测试：
    pytest tests/test_integration.py -v -s
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

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

    - 关闭限流。
    - 打桩 ``get_runtime_pipeline``，问答/WS 测试不加载真实 LLM。
    """
    from api.main import app
    from src.rate_limit import limiter
    import api.routes.chat as chat_module

    limiter.enabled = False
    original = chat_module.get_runtime_pipeline
    # 路由以 get_runtime_pipeline(user_id=..., kb=...) 调用（多知识库支持），
    # 替身必须同时接受 kb，否则 TypeError → 500 / WS 卡住等 done。
    chat_module.get_runtime_pipeline = lambda user_id=None, kb=None: _FakePipeline()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        chat_module.get_runtime_pipeline = original


class TestNoLoginSurface:
    """登录 / 注册面已移除，业务端点直接可用。"""

    def test_login_endpoint_gone(self, client: TestClient):
        response = client.post("/api/v1/auth/login", json={"login": "x", "password": "y"})
        assert response.status_code in (404, 405)

    def test_register_endpoint_gone(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/register",
            json={"username": "x", "email": "x@y.z", "password": "p"},
        )
        assert response.status_code in (404, 405)

    def test_documents_without_token(self, client: TestClient):
        """无凭证访问受保护端点应直接成功。"""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200

    def test_documents_with_bogus_token(self, client: TestClient):
        """伪造的 Authorization 头不再被解析，也不会被拒绝。"""
        response = client.get("/api/v1/documents", headers={"Authorization": "Bearer invalid_token"})
        assert response.status_code == 200


class TestKGCrudFlow:
    """知识图谱 CRUD 流程测试。"""

    def test_kg_stats(self, client: TestClient):
        """测试知识图谱统计结构。

        服务只有一个本地用户，其图谱里可能已有历史数据，因此这里只校验结构。
        """
        response = client.get("/api/v1/kg/stats")
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "relations" in data

    def test_kg_extract_entities(self, client: TestClient):
        """测试实体抽取。

        实体抽取依赖 LLM；测试环境未加载模型时端点返回 503（合理降级），
        此时只验证降级行为而非抽取结果。
        """
        test_text = "苹果公司成立于1976年，由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩创立。"

        response = client.post(
            "/api/v1/kg/extract",
            json={"text": test_text, "source_doc": "test_doc", "persist": True},
        )
        if response.status_code == 503:
            # LLM 不可用（未加载/加载失败）时的降级路径
            return
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "relations" in data
        assert "persisted" in data

    def test_kg_list_entities(self, client: TestClient):
        """测试列出实体。"""
        client.post(
            "/api/v1/kg/extract",
            json={"text": "OpenAI 是一家专注于人工智能研究的公司。", "source_doc": "test"},
        )

        response = client.get("/api/v1/kg/entities")
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "total" in data

    def test_kg_search(self, client: TestClient):
        """测试图谱检索。"""
        client.post(
            "/api/v1/kg/extract",
            json={"text": "人工智能是计算机科学的一个分支。", "source_doc": "ai_doc"},
        )

        response = client.post("/api/v1/kg/search", json={"query": "人工智能", "top_k": 5, "hops": 2})
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "entities" in data
        assert "relations" in data

    def test_kg_list_relations(self, client: TestClient):
        """测试列出关系。"""
        response = client.get("/api/v1/kg/relations")
        assert response.status_code == 200
        data = response.json()
        assert "relations" in data
        assert "total" in data


class TestRAGQueryFlow:
    """RAG 查询流程测试。"""

    def test_list_documents_structure(self, client: TestClient):
        """测试文档列表结构。

        服务只有一个本地用户，库里可能已有历史文档，故只校验结构。
        """
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert "total_chunks" in data

    def test_search_without_results(self, client: TestClient):
        """测试无结果的检索。"""
        response = client.post(
            "/api/v1/search",
            json={"query": "这是一个完全不存在的查询关键词xyz123456", "top_k": 4},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "hits" in data
        assert "total" in data

    def test_chat_sync(self, client: TestClient):
        """测试同步问答（使用 mock pipeline，固定返回测试数据）。"""
        response = client.post("/api/v1/chat", json={"query": "你好，请介绍一下自己", "top_k": 2})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        # mock pipeline 返回 "集成测试回答"
        assert data["answer"] == "集成测试回答"

    def test_chat_empty_query_fails(self, client: TestClient):
        """测试空查询应返回验证错误。"""
        response = client.post("/api/v1/chat", json={"query": "", "top_k": 4})
        assert response.status_code == 422


class TestAgentFlow:
    """Agent 流程测试。"""

    def test_list_agent_tools(self, client: TestClient):
        """测试列出 Agent 工具。"""
        response = client.get("/api/v1/agent/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "total" in data
        assert isinstance(data["tools"], list)

    def test_agent_chat_sync(self, client: TestClient):
        """测试同步 Agent 调用（依赖 LLM，可能失败）。"""
        response = client.post("/api/v1/agent/chat", json={"query": "你好", "max_steps": 2})
        # Agent 需要真实 LLM，测试环境可能不可用
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "steps" in data
        else:
            # 允许 500/503（LLM 不可用）
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
        """测试指标端点（无需任何凭证）。"""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200


class TestWebSocketFlow:
    """WebSocket 流程测试。"""

    def test_websocket_chat_connection(self, client: TestClient):
        """测试 WebSocket 连接（无需 token）。"""
        with client.websocket_connect("/ws/chat") as ws:
            assert ws is not None

    def test_websocket_chat_message(self, client: TestClient):
        """测试 WebSocket 消息发送和接收。"""
        with client.websocket_connect("/ws/chat") as ws:
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
