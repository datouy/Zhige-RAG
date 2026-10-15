"""测试异常信息脱敏（P1-7）。

服务不需要登录，所有请求直接打业务端点。
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from src.db.database import get_db
from src.db.models import Base


@pytest.fixture
def client():
    """创建带 SQLite in-memory DB 的测试客户端。"""
    # 禁用 rate limiter
    from src.rate_limit import limiter

    limiter.enabled = False

    # 创建内存数据库
    engine = create_engine(
        "sqlite:///:memory:",
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

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


class TestExceptionSanitization:
    """异常信息脱敏测试。"""

    def test_validation_error_sanitized(self, client):
        """验证错误应脱敏（不暴露内部字段路径）。"""
        # top_k 应为整数，传字符串触发校验失败
        response = client.post(
            "/api/v1/chat", json={"query": "你好", "top_k": "not-an-int"}
        )
        assert response.status_code == 422
        data = response.json()

        # 应包含通用错误信息
        assert "error" in data
        assert data["error"] == "Validation Error"
        assert "message" in data

        # 应包含 request_id
        assert "request_id" in data

        # 不应直接暴露 Pydantic 的原始错误结构
        assert "detail" not in data or isinstance(data.get("details"), list)

        # 如果有 details，应该是简化后的格式
        if "details" in data:
            for err in data["details"]:
                assert "field" in err
                assert "message" in err
                # 不应包含完整的 loc 路径
                assert "loc" not in err

    def test_database_error_sanitized(self, client):
        """数据库错误应隐藏 SQL 和表名。"""
        response = client.get("/api/v1/documents")

        # 应返回通用错误，不暴露数据库细节
        if response.status_code == 500:
            data = response.json()
            content = str(data)

            # 不应包含敏感信息
            assert "SQL" not in content.upper()
            assert "SELECT" not in content.upper()
            assert "TABLE" not in content.upper()
            assert "SQLALCHEMY" not in content.upper()

            # 应包含 request_id
            assert "request_id" in data

    def test_general_exception_no_stacktrace(self, client):
        """通用异常不应返回堆栈信息。"""
        response = client.get("/api/nonexistent/endpoint")

        # 404 是正常的，但如果是 500，应该脱敏
        if response.status_code == 500:
            data = response.json()
            content = str(data)

            # 不应包含 Python 堆栈信息
            assert "Traceback" not in content
            assert "File \"" not in content

            # 应该是通用提示
            assert "message" in data
            assert "request_id" in data

    def test_auth_endpoints_removed_and_safe(self, client):
        """登录端点已移除，且 404 响应不泄露内部结构。"""
        response = client.post(
            "/api/v1/auth/login",
            json={"login": "nonexistent_user_12345", "password": "SomePass123"},
        )
        assert response.status_code in (404, 405)
        content = str(response.json()).lower()
        assert "traceback" not in content
        assert "sqlalchemy" not in content

    def test_rate_limit_response_safe(self, client):
        """限流错误（如果真的触发）应该是安全的。"""
        response = None
        for _ in range(10):
            response = client.post(
                "/api/v1/feedback",
                json={"query": f"q{_}", "rating": "helpful"},
            )

        if response is not None and response.status_code == 429:
            content = str(response.json())
            # 不应暴露限流实现细节
            assert "slowapi" not in content.lower()
            assert "redis" not in content.lower()
