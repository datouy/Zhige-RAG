"""FastAPI Web 服务 - 中文知识库 RAG 系统.

提供 REST API + WebSocket 流式问答，逐步替代 Streamlit 交互界面。

本服务**不需要登录**：所有请求自动归属唯一的本地用户（见
``src/middleware/identity.py``），因此没有任何注册 / 登录端点。

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

### Rate Limiting
- 查询端点: 60 requests/minute
- 其他端点: 120 requests/minute

### Error Codes
| Code | Description |
|------|-------------|
| 400 | Bad Request - 请求参数错误 |
| 401 | Unauthorized - 请求被拒绝 |
| 403 | Forbidden - 无权限访问 |
| 422 | Validation Error - 请求验证失败 |
| 429 | Rate Limit Exceeded - 请求频率超限 |
| 500 | Internal Server Error - 服务器内部错误 |
| 503 | Service Unavailable - 服务不可用 |

### Monitoring Endpoints
- GET /api/health - 健康检查（5s TTL 缓存）
- GET /api/ready - Kubernetes 就绪探针
- GET /api/metrics - Prometheus 格式指标
- GET /api/v1/metrics - JSON 格式指标
- GET /api/health/invalidate - 失效健康缓存
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# 项目根目录加入 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.database import init_db
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


def _warmup_enabled() -> bool:
    """是否启用启动期预热（``RAG_WARMUP=0`` 可关闭）。"""
    return os.getenv("RAG_WARMUP", "1").strip().lower() not in ("0", "false", "no", "off")


def _warm_up_models() -> None:
    """后台预热：把"第一次请求才付"的固定成本挪到启动期。

    冷启动最贵的其实不是推理，而是**惰性 import + 模型首次加载**：
    ``torch`` / ``transformers`` / ``sentence-transformers`` 三个导入就要
    6 秒上下，再加上 Chroma 客户端、prompt 模板与分块器的初始化。这些开销
    与用户问了什么无关，却全部压在那个"恰好第一个提问"的请求上（实测首个
    ``/api/v1/search`` 要 17 秒、首个 ``/api/v1/chat`` 要 37 秒）。

    这里把它们放到启动后台线程里做一遍，用户打开页面时已经是热的。

    刻意用后台线程而不是 await：不拖慢服务就绪，k8s 探针与 ``/api/health``
    立刻可应答。LLM 是否一并预热由 ``RAG_PRELOAD_LLM`` 决定（默认关，避免
    启动即占显存）；置 1 则首个问题连权重加载都省掉。
    """
    try:
        from api.deps import get_embedding, get_pipeline, get_vector_store

        get_embedding()
        get_vector_store()
        pipeline = get_pipeline()
        logger.info("预热完成：Embedding / 向量库 / Pipeline 已就绪")

        if (
            not pipeline._is_extractive_mode()
            and os.getenv("RAG_PRELOAD_LLM", "0").strip().lower()
            in ("1", "true", "yes", "on")
        ):
            started = time.perf_counter()
            pipeline.ensure_llm()
            logger.info("预热完成：LLM 已加载（%.1fs）", time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001
        logger.warning("后台预热失败（不影响服务，首次请求时会自行加载）: %s", exc)


def _log_orphan_summary() -> None:
    """启动时记录孤儿资源数量，提示可清理（不阻塞启动）。"""
    try:
        data_dir = ROOT / "data"
        kg_count = len(list(data_dir.glob("kg_*.db"))) if data_dir.exists() else 0
        if kg_count > 0:
            logger.warning(
                "检测到 %d 个图谱文件（data/kg_*.db）。如需清理孤儿资源，"
                "调用 POST /api/v1/system/maintenance/cleanup（dry_run=true 预览，false 清理）。",
                kg_count,
            )
    except Exception:  # noqa: BLE001
        pass


# =========================== 应用生命周期 ===========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动和优雅关闭。

    Startup:
        - 初始化数据库
        - 验证配置
        - 后台预热模型（不阻塞就绪）
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
        _log_orphan_summary()

        # P1.2: 启动期预热配置缓存，避免首次请求再走 yaml.safe_load
        try:
            from api.deps import get_runtime_config

            cfg = get_runtime_config()
            logger.info("配置预热完成（%d 个顶级字段）", len(cfg))
        except Exception as cfg_err:
            logger.warning("配置预热失败（继续启动）: %s", cfg_err)

        try:
            from src.config.validator import validate_config

            # 校验最终运行时配置（包含 local_overrides 与环境变量），避免界面
            # 写入的非法值或字段拼写错误拖到首个真实请求才暴露。
            validate_config(get_runtime_config())
            logger.info("配置验证通过")
        except Exception as config_err:
            logger.error("配置验证失败，拒绝启动: %s", config_err)
            raise

        # 冷启动预热：放后台线程，服务立刻可应答（见 _warm_up_models）
        if _warmup_enabled():
            threading.Thread(
                target=_warm_up_models, name="rag-warmup", daemon=True
            ).start()
            logger.info("已在后台开始预热（embedding / 向量库 / pipeline）")

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

        try:
            from src.ingest_jobs import shutdown as _ingest_shutdown

            _ingest_shutdown()
            logger.info("入库任务线程池已关闭")
        except Exception as _ingest_shutdown_err:
            logger.warning("入库任务线程池关闭失败: %s", _ingest_shutdown_err)

        # P0-1 修复：原 AsyncLLMExecutor 是死代码（已删除），
        # 现在的 LLMConcurrencyGate 是无状态的 semaphore 包装，无需清理

        for handler in logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        try:
            # 注意：模块已从 src/utils/circuit_breaker.py 迁移到
            # src/circuit_breaker.py。原路径不可达——src/utils.py 是普通模块，
            # src/utils/ 目录（无 __init__.py）会被它遮蔽，导致子模块无法导入，
            # 这里的重置逻辑此前被 except ImportError 静默吞掉、从未执行。
            from src.circuit_breaker import reset_all_circuit_breakers

            reset_all_circuit_breakers()
            logger.info("熔断器状态已重置")
        except ImportError as cb_err:
            logger.warning("熔断器模块导入失败，跳过重置: %s", cb_err)

        logger.info("ChineseRAGKB API 关闭完成")
        logger.info("=" * 50)
    except Exception as shutdown_err:
        logger.error("关闭时出错: %s", shutdown_err)


# =========================== 应用实例 ===========================
app = FastAPI(
    title="知阁 Zhige API",
    description="知阁（Zhige）—— 本地中文知识库 RAG 平台，无需登录，支持多模型后端",
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

# 速率限制器（全局限流中间件与各路由共用同一个实例）
from src.rate_limit import limiter

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if "*" in _origins:
    # "*" 与 allow_credentials=True 组合时，浏览器会允许任意站点携带凭证，
    # 属高危配置。此前只处理"整个值就等于 *"的情况，写成
    # "https://a.com,*" 会漏网。这里统一剔除通配符。
    logger.warning(
        "ALLOWED_ORIGINS 含通配符 '*'，与 allow_credentials=True 冲突，已剔除；"
        "请显式列出允许的源。"
    )
    _origins = [o for o in _origins if o != "*"]
if not _origins:
    _origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]
_allowed_origins = _origins

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
    chat_router,
    eval_router,
    feedback_router,
    kb_router,
    kg_router,
    settings_router,
    system_router,
)

# 直接使用子路由里写明的 ``Depends(get_current_user)``：每个端点都要拿
# 当前用户身份（本地单用户），P1 阶段无需 ``dependency_overrides`` hack。
app.include_router(system_router)
app.include_router(settings_router)
app.include_router(kb_router)
app.include_router(chat_router)
app.include_router(kg_router)
app.include_router(agent_router)
app.include_router(eval_router)
app.include_router(feedback_router)


# =========================== 异常处理器 ===========================
# P1-7 修复：异常信息脱敏，防止内部实现细节外泄

def _sanitize_validation_errors(errors: list) -> list:
    """脱敏验证错误（隐藏内部字段路径细节）。"""
    sanitized = []
    for err in errors:
        # 保留 type 和 msg，但简化 loc（位置）信息
        loc = err.get("loc", [])
        # 只保留最后一级字段名，隐藏嵌套路径
        simplified_loc = [loc[-1]] if loc else []
        sanitized.append({
            "field": simplified_loc[0] if simplified_loc else "unknown",
            "message": err.get("msg", "验证失败"),
        })
    return sanitized


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """P1-7：脱敏验证错误，只返回字段名和友好提示。"""
    request_id = getattr(request.state, "request_id", "N/A")
    logger.warning("请求验证失败 [request_id=%s]: %s", request_id, exc.errors())
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "请求参数验证失败",
            "details": _sanitize_validation_errors(exc.errors()),
            "request_id": request_id,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """统一 HTTP 错误响应，服务端 5xx 不再把内部异常文本返回给客户端。"""
    request_id = getattr(request.state, "request_id", "N/A")
    if exc.status_code >= 500:
        logger.error(
            "HTTP 服务错误 [request_id=%s, status=%s]: %s",
            request_id,
            exc.status_code,
            exc.detail,
        )
        content = {
            "error": "Internal Server Error",
            "message": "服务暂时不可用，请稍后重试",
            "request_id": request_id,
        }
    else:
        content = {
            "error": "HTTP Error",
            "message": exc.detail,
            "request_id": request_id,
        }
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(request: Request, exc: SQLAlchemyError):
    """P1-7：隐藏数据库错误细节（表名、SQL等）。"""
    request_id = getattr(request.state, "request_id", "N/A")
    # 记录完整错误到日志（含堆栈）
    logger.error("数据库错误 [request_id=%s]: %s", request_id, exc, exc_info=True)
    
    # 返回通用提示，不暴露数据库结构
    return JSONResponse(
        status_code=500,
        content={
            "error": "Database Error",
            "message": "数据库操作失败，请稍后重试",
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """P1-7：通用异常处理，完全隐藏内部错误。"""
    request_id = getattr(request.state, "request_id", "N/A")
    # 记录完整堆栈到日志
    logger.error("未处理异常 [request_id=%s]: %s", request_id, exc, exc_info=True)
    
    # 返回通用错误，不泄露任何实现细节
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "服务暂时不可用，请稍后重试",
            "request_id": request_id,
        },
    )


# =========================== 根路径 ===========================
@app.get("/")
async def root():
    """直接返回聊天界面，不做 307 跳转。

    原先返回 ``RedirectResponse("/index.html")``，会让部分内置预览面板 /
    无跳转的客户端拿不到页面（307 不是 200）。直接吐文件最稳。
    """
    index = ROOT / "ui" / "web" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return RedirectResponse(url="/index.html")


# =========================== 静态文件 ===========================
web_static_dir = ROOT / "ui" / "web"
if web_static_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
