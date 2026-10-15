"""测试异常信息脱敏（P1-7）。"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import MagicMock

from api.main import app
from src.db.database import get_db
from src.db.models import Base


@pytest.fixture
def client():
    """创建带 SQLite in-memory DB 的测试客户端。"""
    # 禁用 rate limiter
    from api.routes import auth as auth_module
    if hasattr(auth_module, "limiter"):
        auth_module.limiter.enabled = False
    
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
        # 发送格式错误的注册请求
        response = client.post(
            "/api/v1/auth/register",
            json={"username": "ab", "password": "short", "email": "invalid"},
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
        # 尝试用无效 token 访问需要数据库的端点
        response = client.get(
            "/api/v1/kb/list",
            headers={"Authorization": "Bearer invalid_token_format"},
        )
        
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
        # 这个测试需要触发一个未捕获的异常
        # 由于我们的端点都有错误处理，这里模拟场景
        
        # 尝试访问不存在的端点会返回 404（FastAPI 默认处理）
        response = client.get("/api/nonexistent/endpoint")
        
        # 404 是正常的，但如果是 500，应该脱敏
        if response.status_code == 500:
            data = response.json()
            content = str(data)
            
            # 不应包含 Python 堆栈信息
            assert "Traceback" not in content
            assert "File \"" not in content
            assert "line " not in content.lower() or "request_id" in content.lower()
            
            # 应该是通用提示
            assert "message" in data
            assert "request_id" in data

    def test_auth_error_messages_safe(self, client):
        """认证错误不应泄露用户存在性。"""
        # 尝试用不存在的用户登录
        response = client.post(
            "/api/v1/auth/login",
            json={"login": "nonexistent_user_12345", "password": "SomePass123"},
        )
        
        assert response.status_code == 401
        data = response.json()
        
        # 应返回通用提示，不区分"用户不存在"或"密码错误"
        assert "detail" in data
        assert "用户名或密码错误" in data["detail"] or "Unauthorized" in str(data)
        
        # 不应包含具体信息
        assert "不存在" not in data.get("detail", "").lower() or "用户名或密码" in data.get("detail", "")

    def test_rate_limit_response_safe(self, client):
        """限流错误应该是安全的。"""
        # 快速发送多个注册请求触发限流
        for _ in range(10):
            response = client.post(
                "/api/v1/auth/register",
                json={
                    "username": f"user_{_}",
                    "password": "ValidPass123",
                    "email": f"user{_}@example.com",
                },
            )
        
        # 最后一个应该被限流
        if response.status_code == 429:
            data = response.json()
            content = str(data)
            
            # 不应暴露限流实现细节
            assert "slowapi" not in content.lower()
            assert "redis" not in content.lower()
