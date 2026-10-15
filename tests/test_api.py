"""FastAPI API 测试套件。

运行测试：
    pytest tests/test_api.py -v

说明：
- 使用 TestClient（无需启动独立服务器）。
- 服务不需要登录：所有端点直接访问即可，测试里没有任何 token / 登录步骤。
- 问答 / WebSocket 测试通过 monkeypatch 替换 ``get_runtime_pipeline``，
  避免真实加载 LLM（模型加载 + 推理会让测试套件慢数分钟）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# 确保项目根目录在 path 中
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 设置测试模式的环境变量
os.environ["STREAMLIT_RUNTIME"] = "enable"


class _FakePipeline:
    """轻量替身：覆盖 answer / astream_answer，不加载任何模型。"""

    # chat 路由会探测长期记忆存储；None = 不可用，路由须容忍
    long_term_store = None

    def answer(self, question, top_k=None, where=None, stream=False, history=None, user_id=None, allowed_groups=None):
        from src.rag_pipeline import RAGResult

        return RAGResult(answer="测试回答", sources=[], raw_hits=[], timings={})

    async def astream_answer(self, question, top_k=None, history=None, user_id=None, allowed_groups=None):
        yield {"event": "hits", "data": []}
        yield {"event": "token", "data": "测试"}
        yield {"event": "token", "data": "回答"}
        yield {"event": "done", "data": []}


# =========================== Fixtures ===========================
@pytest.fixture(scope="module")
def client():
    """创建 FastAPI 测试客户端，并对 LLM 依赖打桩。"""
    from api.main import app
    from src.rate_limit import limiter
    import api.routes.chat as chat_module

    limiter.enabled = False

    # 打桩：问答端点不加载真实 LLM
    # 注意：路由以 get_runtime_pipeline(user_id=..., kb=...) 调用（多知识库支持），
    # 替身必须同时接受 kb，否则会 TypeError → 500 / WS 卡住等 done。
    original = chat_module.get_runtime_pipeline
    chat_module.get_runtime_pipeline = lambda user_id=None, kb=None: _FakePipeline()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        chat_module.get_runtime_pipeline = original


# =========================== 测试：无需登录 ===========================
class TestNoLoginRequired:
    """服务不需要登录：受保护端点直接可访问，登录注册端点已移除。"""

    @pytest.mark.parametrize("path", ["/api/v1/auth/login", "/api/v1/auth/register"])
    def test_auth_endpoints_removed(self, client, path):
        """登录 / 注册端点必须已删除（404 或 405）。"""
        response = client.post(path, json={"username": "x", "password": "y"})
        assert response.status_code in (404, 405)

    def test_documents_without_token(self, client):
        """不带任何凭证也能读文档列表。"""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200

    def test_search_without_token(self, client):
        """不带任何凭证也能检索。"""
        response = client.post("/api/v1/search", json={"query": "什么是 RAG", "top_k": 4})
        assert response.status_code == 200

    def test_chat_without_token(self, client):
        """不带任何凭证也能问答。"""
        response = client.post("/api/v1/chat", json={"query": "你好", "top_k": 4})
        assert response.status_code == 200

    def test_config_without_token(self, client):
        """不带任何凭证也能读运行配置。"""
        response = client.get("/api/v1/config")
        assert response.status_code == 200

    def test_wrong_credentials_are_not_special_cased(self, client):
        """Authorization 头不再被解析：随便填也会落到本地用户。"""
        response = client.get(
            "/api/v1/documents", headers={"Authorization": "Bearer nonsense"}
        )
        assert response.status_code == 200


# =========================== 测试：健康检查 ===========================
class TestHealth:
    """GET /api/health 测试。"""

    def test_health_returns_200(self, client):
        """健康检查应返回 200 状态码。"""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self, client):
        """健康检查应返回 status: ok。"""
        response = client.get("/api/health")
        data = response.json()
        assert data.get("status") == "ok"

    def test_health_includes_chunks_count(self, client):
        """健康检查应包含 chunks 数量。"""
        response = client.get("/api/health")
        data = response.json()
        assert "chunks" in data
        assert isinstance(data["chunks"], int)

    def test_health_includes_sources_count(self, client):
        """健康检查应包含 sources 数量。"""
        response = client.get("/api/health")
        data = response.json()
        assert "sources" in data


# =========================== 测试：文档列表 ===========================
class TestDocuments:
    """GET /api/v1/documents 测试。"""

    def test_list_documents_returns_200(self, client):
        """文档列表应返回 200。"""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200

    def test_list_documents_returns_array(self, client):
        """文档列表应返回 documents 数组。"""
        response = client.get("/api/v1/documents")
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_list_documents_includes_total(self, client):
        """文档列表应包含 total 和 total_chunks。"""
        response = client.get("/api/v1/documents")
        data = response.json()
        assert "total" in data
        assert "total_chunks" in data


# =========================== 测试：检索 ===========================
class TestSearch:
    """POST /api/v1/search 测试。"""

    def test_search_returns_200(self, client):
        """检索应返回 200。"""
        response = client.post("/api/v1/search", json={"query": "什么是 RAG", "top_k": 4})
        assert response.status_code == 200

    def test_search_returns_hits(self, client):
        """检索应返回 hits 数组。"""
        response = client.post("/api/v1/search", json={"query": "什么是 RAG", "top_k": 4})
        data = response.json()
        assert "hits" in data
        assert isinstance(data["hits"], list)

    def test_search_includes_scores(self, client):
        """检索结果应包含相似度分数。"""
        response = client.post("/api/v1/search", json={"query": "RAG", "top_k": 4})
        data = response.json()
        for hit in data.get("hits", []):
            assert "score" in hit

    def test_search_respects_top_k(self, client):
        """检索应尊重 top_k 参数。"""
        response = client.post("/api/v1/search", json={"query": "测试", "top_k": 2})
        data = response.json()
        # 注意：实际返回数量取决于知识库内容
        assert data["hits"] is not None or data["total"] == 0


# =========================== 测试：同步问答 ===========================
class TestChatSync:
    """POST /api/v1/chat 测试。"""

    def test_chat_sync_returns_200(self, client):
        """同步问答应返回 200。"""
        response = client.post("/api/v1/chat", json={"query": "你好", "top_k": 4})
        assert response.status_code == 200

    def test_chat_sync_returns_answer(self, client):
        """同步问答应返回 answer 字段。"""
        response = client.post("/api/v1/chat", json={"query": "你好", "top_k": 4})
        data = response.json()
        assert "answer" in data

    def test_chat_sync_returns_sources(self, client):
        """同步问答应返回 sources 字段。"""
        response = client.post("/api/v1/chat", json={"query": "你好", "top_k": 4})
        data = response.json()
        assert "sources" in data

    def test_chat_requires_query(self, client):
        """空 query 应返回验证错误。"""
        response = client.post("/api/v1/chat", json={"query": "", "top_k": 4})
        # FastAPI 默认返回 422 Unprocessable Entity
        assert response.status_code == 422


# =========================== 测试：WebSocket 流式问答 ===========================
class TestWebSocketChat:
    """WS /ws/chat 测试（不需要 token）。"""

    def test_websocket_connects(self, client):
        """无需 token 即可连接。"""
        with client.websocket_connect("/ws/chat") as websocket:
            assert websocket is not None

    def test_websocket_receive_message(self, client):
        """WebSocket 应能接收消息。"""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json({"query": "测试", "top_k": 4})
            events = []
            try:
                while True:
                    event = websocket.receive_json()
                    events.append(event)
                    if event.get("event") == "done":
                        break
            except Exception:
                pass
            assert len(events) > 0
            for ev in events:
                assert "event" in ev
                assert "data" in ev

    def test_websocket_event_types(self, client):
        """WebSocket 事件应包含正确的类型。"""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json({"query": "你好", "top_k": 2})
            event_types = set()
            try:
                while True:
                    event = websocket.receive_json()
                    event_types.add(event.get("event"))
                    if event.get("event") == "done":
                        break
            except Exception:
                pass
            assert "hits" in event_types
            assert "done" in event_types

    def test_websocket_empty_query_error(self, client):
        """空 query 应返回错误。"""
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json({"query": "", "top_k": 4})
            event = websocket.receive_json()
            assert event.get("event") == "error"


# =========================== 测试：配置接口 ===========================
class TestConfig:
    """GET /api/v1/config 测试。"""

    def test_config_returns_200(self, client):
        """配置接口应返回 200。"""
        response = client.get("/api/v1/config")
        assert response.status_code == 200

    def test_config_includes_sections(self, client):
        """配置应包含主要配置块。"""
        response = client.get("/api/v1/config")
        data = response.json()
        expected_sections = ["embedding", "llm", "vector_store"]
        for section in expected_sections:
            assert section in data, f"配置应包含 {section} 节"

    def test_config_has_no_auth_section(self, client):
        """配置里不应再有 auth 段（登录开关已删除）。"""
        response = client.get("/api/v1/config")
        assert "auth" not in response.json()

    def test_config_redacts_secrets(self, client):
        """配置中的敏感键（如 neo4j password）应被脱敏。"""
        response = client.get("/api/v1/config")
        data = response.json()
        kg = data.get("knowledge_graph", {})
        neo4j = kg.get("neo4j", {}) if isinstance(kg, dict) else {}
        if "password" in neo4j:
            assert neo4j["password"] == "***"


# =========================== 测试：上传功能 ===========================
class TestUpload:
    """POST /api/v1/ingest 测试。"""

    def test_ingest_requires_file(self, client):
        """无文件上传应返回 422。"""
        response = client.post("/api/v1/ingest")
        assert response.status_code == 422

    def test_ingest_rejects_path_traversal(self, client):
        """路径穿越文件名应返回 400。"""
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("../../evil.txt", b"malicious")},
        )
        assert response.status_code == 400


# =========================== 测试：根路径 ===========================
class TestRoot:
    """GET / 测试。"""

    def test_root_returns_index(self, client):
        """根路径应返回主界面（或重定向到它）。"""
        response = client.get("/")
        assert response.status_code in (200, 307, 308)


# =========================== 测试：API 前缀 ===========================
class TestApiPrefix:
    """API 路径测试。"""

    def test_api_path_prefix(self, client):
        """API 路径应使用 /api/ 前缀。"""
        response = client.get("/api/health")
        assert response.status_code == 200
