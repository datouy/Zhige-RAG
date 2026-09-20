"""FastAPI Web 服务 - 中文知识库 RAG 系统.

提供 REST API + WebSocket 流式问答，逐步替代 Streamlit 交互界面。
支持多租户认证和用户隔离。

启动命令：
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

本文件仅负责：
- 加载 FastAPI 应用
- 注册全局中间件（CORS、RequestID、PerformanceMonitor、Security Headers）
- ``include_router`` 各子路由
- 提供应用生命周期 (lifespan)
- 挂载静态前端

所有业务路由实现位于 ``api/routes/*.py``，单例与共享依赖位于 ``api/deps.py``，
安全中间件位于 ``api/middleware.py``。

## ChineseRAGKB 企业知识库系统

### Authentication
所有认证相关端点无需认证，其他受保护端点需要 `Authorization: Bearer <token>` header。

### Rate Limiting
- 认证端点: 5 requests/minute
- 注册端点: 3 requests/minute
- 查询端点: 60 requests/minute
- 其他端点: 120 requests/minute

### Error Codes
| Code | Description |
|------|-------------|
| 400 | Bad Request - 请求参数错误 |
| 401 | Unauthorized - 未认证或 Token 无效 |
| 403 | Forbidden - 无权限访问 |
| 422 | Validation Error - 请求验证失败 |
| 429 | Rate Limit Exceeded - 请求频率超限 |
| 500 | Internal Server Error - 服务器内部错误 |
| 503 | Service Unavailable - 服务不可用 |

### Monitoring Endpoints
- GET /api/health - 健康检查（无需认证，5s TTL 缓存）
- GET /api/ready - Kubernetes 就绪探针（无需认证）
- GET /api/metrics - Prometheus 格式指标（无需认证）
- GET /api/v1/metrics - JSON 格式指标（需要认证）
- GET /api/health/invalidate - 失效健康缓存（无需认证）
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# 项目根目录加入 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.database import init_db
from src.db.models import User
from src.middleware.auth import get_current_user
from src.middleware.logging import (
    PerformanceMonitorMiddleware,
    RequestLoggingMiddleware,
)
from src.utils import get_logger

logger = get_logger("api")


# =========================== Request ID 中间件 ===========================
class RequestIDMiddleware(BaseHTTPMiddleware):
    """为每个请求添加唯一请求 ID，便于链路追踪。"""

    async def dispatch(self, request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response


# =========================== 应用生命周期 ===========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动和优雅关闭。

    Startup:
        - 初始化数据库
        - 验证配置
        - 记录启动日志

    Shutdown:
        - 关闭数据库连接池
        - 关闭 LLM 客户端
        - 刷新日志缓冲区
        - 重置熔断器状态
    """
    logger.info("=" * 50)
    logger.info("ChineseRAGKB API 正在启动...")
    logger.info("=" * 50)

    try:
        init_db()
        logger.info("数据库初始化完成")

        # P1.2: 启动期预热配置缓存，避免首次请求再走 yaml.safe_load
        try:
            from api.deps import get_runtime_config

            cfg = get_runtime_config()
            logger.info("配置预热完成（%d 个顶级字段）", len(cfg))
        except Exception as cfg_err:
            logger.warning("配置预热失败（继续启动）: %s", cfg_err)

        try:
            from src.config.validator import load_and_validate_config

            load_and_validate_config("config/config.yaml")
            logger.info("配置验证通过")
        except Exception as config_err:
            logger.warning("配置验证失败（继续启动）: %s", config_err)

        logger.info("ChineseRAGKB API 启动完成")
        logger.info("=" * 50)
    except Exception as startup_err:
        logger.error("启动失败: %s", startup_err)
        raise

    yield

    logger.info("=" * 50)
    logger.info("ChineseRAGKB API 正在关闭...")
    logger.info("=" * 50)
    try:
        from src.db.database import close_db

        close_db()
        logger.info("数据库连接池已关闭")

        from api.deps import get_llm_executor

        executor = get_llm_executor()
        # P5 审计修复：AsyncLLMExecutor 只有 stop()，之前 hasattr 检查的
        # close/shutdown 都不存在，后台循环任务从不停止。
        stop = getattr(executor, "stop", None)
        if stop is not None:
            try:
                await asyncio.wait_for(executor.stop(), timeout=5.0)
                logger.info("LLM 执行器已关闭")
            except asyncio.TimeoutError:
                logger.warning("LLM 执行器关闭超时")
            except Exception as llm_err:
                logger.warning("LLM 执行器关闭时出错: %s", llm_err)

        for handler in logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        try:
            from src.utils.circuit_breaker import reset_all_circuit_breakers

            reset_all_circuit_breakers()
            logger.info("熔断器状态已重置")
        except ImportError:
            pass

        logger.info("ChineseRAGKB API 关闭完成")
        logger.info("=" * 50)
    except Exception as shutdown_err:
        logger.error("关闭时出错: %s", shutdown_err)


# =========================== 应用实例 ===========================
app = FastAPI(
    title="ChineseRAGKB API",
    description="中文知识库 RAG 系统的 FastAPI 接口，支持多租户",
    version="0.3.0",
    lifespan=lifespan,
)


# =========================== 全局中间件 ===========================
app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    PerformanceMonitorMiddleware,
    slow_request_threshold_ms=float(os.getenv("SLOW_REQUEST_THRESHOLD_MS", "1000.0")),
)

app.add_middleware(
    RequestLoggingMiddleware,
    sample_rate=float(os.getenv("LOG_SAMPLE_RATE", "1.0")),
    log_request_body=os.getenv("LOG_REQUEST_BODY", "false").lower() == "true",
    log_response_body=False,
)

app.add_middleware(SlowAPIMiddleware)

# 安全响应头中间件（P2.3 / P2.4）
from api.middleware import SecurityHeadersMiddleware

app.add_middleware(SecurityHeadersMiddleware)

# 速率限制器（auth 路由共享）
from api.routes.auth import limiter

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
if _allowed_origins == [""] or _allowed_origins == ["*"]:
    _allowed_origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


# =========================== 路由挂载 ===========================
from api.routes import (
    agent_router,
    auth_router,
    chat_router,
    eval_router,
    feedback_router,
    kg_router,
    subscription_router,
    system_router,
)

# 直接使用子路由里写明的 ``Depends(get_current_user)``：每个受保护端点
# 在路由函数声明处显式依赖认证，P1 阶段无需 ``dependency_overrides`` hack。
app.include_router(auth_router)
app.include_router(subscription_router)
app.include_router(system_router)
app.include_router(chat_router)
app.include_router(kg_router)
app.include_router(agent_router)
app.include_router(eval_router)
app.include_router(feedback_router)


# =========================== 异常处理器 ===========================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "detail": exc.errors()},
    )


@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.error("数据库错误: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Database Error", "message": "数据库操作失败，请稍后重试"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.error("未处理异常 [request_id=%s]: %s", request_id, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "发生未知错误，请联系管理员",
        },
    )


# =========================== 根路径 ===========================
@app.get("/")
async def root():
    return RedirectResponse(url="/index.html")


# =========================== 静态文件 ===========================
web_static_dir = ROOT / "ui" / "web"
if web_static_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)