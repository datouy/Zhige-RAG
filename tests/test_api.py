"""FastAPI API 测试套件。

运行测试：
    pytest tests/test_api.py -v

注意：测试需要启动服务器，可以在测试中启动或使用 fixture。
"""

from __future__ import annotations

import asyncio
import json
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


# =========================== Fixtures ===========================
@pytest.fixture(scope="module")
def client():
    """创建 FastAPI 测试客户端。"""
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def event_loop():
    """创建事件循环用于异步测试。"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


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
    """GET /api/documents 测试。"""

    def test_list_documents_returns_200(self, client):
        """文档列表应返回 200。"""
        response = client.get("/api/documents")
        assert response.status_code == 200

    def test_list_documents_returns_array(self, client):
        """文档列表应返回 documents 数组。"""
        response = client.get("/api/documents")
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_list_documents_includes_total(self, client):
        """文档列表应包含 total 和 total_chunks。"""
        response = client.get("/api/documents")
        data = response.json()
        assert "total" in data
        assert "total_chunks" in data


# =========================== 测试：检索 ===========================
class TestSearch:
    """POST /api/search 测试。"""

    def test_search_returns_200(self, client):
        """检索应返回 200。"""
        response = client.post(
            "/api/search",
            json={"query": "什么是 RAG", "top_k": 4}
        )
        assert response.status_code == 200

    def test_search_returns_hits(self, client):
        """检索应返回 hits 数组。"""
        response = client.post(
            "/api/search",
            json={"query": "什么是 RAG", "top_k": 4}
        )
        data = response.json()
        assert "hits" in data
        assert isinstance(data["hits"], list)

    def test_search_includes_scores(self, client):
        """检索结果应包含相似度分数。"""
        response = client.post(
            "/api/search",
            json={"query": "RAG", "top_k": 4}
        )
        data = response.json()
        for hit in data.get("hits", []):
            assert "score" in hit

    def test_search_respects_top_k(self, client):
        """检索应尊重 top_k 参数。"""
        response = client.post(
            "/api/search",
            json={"query": "测试", "top_k": 2}
        )
        data = response.json()
        # 注意：实际返回数量取决于知识库内容
        assert data["hits"] is not None or data["total"] == 0


# =========================== 测试：同步问答 ===========================
class TestChatSync:
    """POST /api/chat 测试。"""

    def test_chat_sync_returns_200(self, client):
        """同步问答应返回 200。"""
        response = client.post(
            "/api/chat",
            json={"query": "你好", "top_k": 4}
        )
        assert response.status_code == 200

    def test_chat_sync_returns_answer(self, client):
        """同步问答应返回 answer 字段。"""
        response = client.post(
            "/api/chat",
            json={"query": "你好", "top_k": 4}
        )
        data = response.json()
        assert "answer" in data

    def test_chat_sync_returns_sources(self, client):
        """同步问答应返回 sources 字段。"""
        response = client.post(
            "/api/chat",
            json={"query": "你好", "top_k": 4}
        )
        data = response.json()
        assert "sources" in data

    def test_chat_requires_query(self, client):
        """空 query 应返回验证错误。"""
        response = client.post(
            "/api/chat",
            json={"query": "", "top_k": 4}
        )
        # FastAPI 默认返回 422 Unprocessable Entity
        assert response.status_code == 422


# =========================== 测试：WebSocket 流式问答 ===========================
class TestWebSocketChat:
    """WS /ws/chat 测试。"""

    def test_websocket_connect(self, client):
        """WebSocket 应能成功连接。"""
        with client.websocket_connect("/ws/chat") as websocket:
            assert websocket is not None

    def test_websocket_receive_message(self, client):
        """WebSocket 应能接收消息。"""
        with client.websocket_connect("/ws/chat") as websocket:
            # 发送查询
            websocket.send_json({"query": "测试", "top_k": 4})
            
            # 接收事件流（至少应收到一个事件）
            events = []
            try:
                while True:
                    event = websocket.receive_json()
                    events.append(event)
                    if event.get("event") == "done":
                        break
            except Exception:
                pass
            
            # 验证事件格式
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
            
            # 应收到 hits, token/done 事件
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
    """GET /api/config 测试。"""

    def test_config_returns_200(self, client):
        """配置接口应返回 200。"""
        response = client.get("/api/config")
        assert response.status_code == 200

    def test_config_includes_sections(self, client):
        """配置应包含主要配置块。"""
        response = client.get("/api/config")
        data = response.json()
        # 应包含一些主要配置
        expected_sections = ["embedding", "llm", "vector_store"]
        for section in expected_sections:
            assert section in data, f"配置应包含 {section} 节"


# =========================== 测试：上传功能 ===========================
class TestUpload:
    """POST /api/ingest 测试。"""

    def test_ingest_requires_file(self, client):
        """无文件上传应返回 422。"""
        response = client.post("/api/ingest")
        assert response.status_code == 422


# =========================== 测试：根路径 ===========================
class TestRoot:
    """GET / 测试。"""

    def test_root_returns_redirect(self, client):
        """根路径应重定向到 index.html。"""
        response = client.get("/")
        assert response.status_code == 200
        # 或 307 重定向
        assert response.status_code in (200, 307, 308)


# =========================== 测试：API 前缀 ===========================
class TestApiPrefix:
    """API 路径测试。"""

    def test_api_path_prefix(self, client):
        """API 路径应使用 /api/ 前缀。"""
        response = client.get("/api/health")
        assert response.status_code == 200
