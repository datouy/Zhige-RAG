"""请求/响应日志中间件，记录所有 API 请求和响应。

P3.1
----
- 修复原格式字符串 bug：混用 ``%(name)s`` 与 ``%s`` 会抛 ``TypeError``。
- 引入 ``request_id_var``（:class:`contextvars.ContextVar`）：在请求开始时
  通过 :class:`RequestIDContextFilter` 注入到 ``LogRecord``，业务代码无需
  手动加 ``extra={"request_id": ...}`` 也能自动带上。
- 业务侧如需获取当前 request_id，可调用 :func:`get_current_request_id`。
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils import get_logger

logger = get_logger("request_log")


# ----------------------------------------------------------------------
#  Request ID ContextVar（P3.1）
# ----------------------------------------------------------------------
# 在中间件入口处 set，路由代码可以零侵入地从 LogRecord 中读取 request_id。
# ContextVar 是 asyncio 安全的，避免不同请求之间串号。
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


def get_current_request_id() -> Optional[str]:
    """获取当前异步上下文里的 request_id（在请求处理中调用有意义）。"""
    return request_id_var.get()


class RequestIDContextFilter(logging.Filter):
    """Logging filter：把当前 :data:`request_id_var` 注入到 LogRecord。

    用法（在 ``setup_logger`` 里挂上）::

        for h in logger.handlers:
            h.addFilter(RequestIDContextFilter())

    挂上之后，``JSONFormatter`` / 普通 ``Formatter`` 都可以通过
    ``record.request_id`` 拿到值。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.request_id = request_id_var.get()
        except LookupError:
            record.request_id = None
        return True


# ----------------------------------------------------------------------
#  敏感字段脱敏
# ----------------------------------------------------------------------
# 需要脱敏的字段
SENSITIVE_FIELDS: Set[str] = {
    "password",
    "password_hash",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "authorization",
    "credential",
    "private_key",
}

# URL 中的敏感路径模式
SENSITIVE_PATHS: List[re.Pattern] = [
    re.compile(r"/auth/(login|register|password)"),
    re.compile(r"/admin/"),
]


def _mask_sensitive_data(data: Any, depth: int = 0) -> Any:
    """递归脱敏敏感数据。

    Args:
        data: 待脱敏数据
        depth: 递归深度，防止无限递归

    Returns:
        脱敏后的数据
    """
    if depth > 10:
        return "[MAX_DEPTH]"

    if isinstance(data, dict):
        return {
            k: "[REDACTED]" if k.lower() in SENSITIVE_FIELDS else _mask_sensitive_data(v, depth + 1)
            for k, v in data.items()
        }
    elif isinstance(data, (list, tuple)):
        return [_mask_sensitive_data(item, depth + 1) for item in data]
    elif isinstance(data, str) and len(data) > 100:
        return data[:50] + "...[TRUNCATED]"
    return data


def _is_sensitive_path(path: str) -> bool:
    """检查路径是否包含敏感操作。"""
    return any(pattern.search(path) for pattern in SENSITIVE_PATHS)


def _get_client_ip(request: Request) -> str:
    """获取客户端真实 IP。"""
    # 优先从代理头获取
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # 回退到直接获取
    if request.client:
        return request.client.host

    return "unknown"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求/响应日志中间件。

    功能：
    - 记录所有入站请求
    - 记录响应状态和耗时
    - 支持采样率配置（高流量场景）
    - 敏感信息自动脱敏
    - 请求 ID 追踪（P3.1：通过 ContextVar 自动注入到所有 LogRecord）
    """

    def __init__(
        self,
        app: Any,
        sample_rate: float = 1.0,
        log_request_body: bool = True,
        log_response_body: bool = False,
        exclude_paths: Optional[List[str]] = None,
    ):
        """初始化中间件。

        Args:
            app: FastAPI 应用
            sample_rate: 日志采样率（0.0-1.0）
            log_request_body: 是否记录请求体
            log_response_body: 是否记录响应体（默认关闭，避免性能问题）
            exclude_paths: 排除的路径列表（如健康检查）
        """
        super().__init__(app)
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or ["/api/health", "/api/ready", "/metrics"]

    def _should_log(self, path: str) -> bool:
        """判断是否应该记录此请求。"""
        # 排除特定路径
        for exclude in self.exclude_paths:
            if path.startswith(exclude):
                return False

        # 采样
        if self.sample_rate < 1.0:
            import random
            return random.random() < self.sample_rate

        return True

    def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """提取请求信息。"""
        info = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": _get_client_ip(request),
            "user_agent": request.headers.get("User-Agent", ""),
        }

        # 获取路径参数（如果有）
        if hasattr(request, "path_params"):
            info["path_params"] = dict(request.path_params)

        return info

    async def _extract_request_body(self, request: Request) -> Optional[Any]:
        """尝试提取请求体。"""
        if not self.log_request_body:
            return None

        try:
            # 尝试获取 body（可能为空或不适用）
            body = await request.body()
            if body:
                # 尝试解析 JSON
                try:
                    data = json.loads(body)
                    # 脱敏
                    return _mask_sensitive_data(data)
                except json.JSONDecodeError:
                    return "[BINARY or TEXT DATA]"
        except Exception:
            pass

        return None

    def _extract_response_info(self, response: Response) -> Dict[str, Any]:
        """提取响应信息。"""
        return {
            "status_code": response.status_code,
        }

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """处理请求。"""
        # 生成请求 ID（优先沿用 RequestIDMiddleware 已设置的，否则新建）
        request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
        request.state.request_id = request_id
        # P3.1：写入 ContextVar，让下游 logger 自动带上 request_id
        token = request_id_var.set(request_id)
        try:
            return await self._do_dispatch(request, call_next, request_id)
        finally:
            request_id_var.reset(token)

    async def _do_dispatch(
        self,
        request: Request,
        call_next: Callable,
        request_id: str,
    ) -> Response:
        """真正的请求处理逻辑（外层用 try/finally 包裹 ContextVar 重置）。"""
        # 检查是否应该记录
        should_log = self._should_log(request.url.path)

        # 记录开始时间
        start_time = time.perf_counter()

        # 请求信息
        request_info = self._extract_request_info(request)
        request_info["request_id"] = request_id
        request_info["timestamp"] = time.time()

        # 敏感路径标记
        is_sensitive = _is_sensitive_path(request.url.path)

        if should_log:
            # 仅对非敏感路径记录请求体
            if not is_sensitive:
                body = await self._extract_request_body(request)
                if body:
                    request_info["body"] = body
            else:
                request_info["body"] = "[SENSITIVE PATH]"

            logger.info(
                "Request started: %(method)s %(path)s from %(client_ip)s [%(request_id)s]",
                request_info,
            )

        # 处理请求
        try:
            response = await call_next(request)

            # 计算耗时
            duration_ms = (time.perf_counter() - start_time) * 1000

            # 响应信息
            response_info = self._extract_response_info(response)
            response_info["duration_ms"] = round(duration_ms, 2)
            response_info["request_id"] = request_id

            # 添加到响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            if should_log:
                # P3.1: 所有命名占位符都从 dict 里取（避免 %(name)s 与 %s 混用）
                merged = {**request_info, **response_info}
                if response.status_code >= 500:
                    logger.error(
                        "Request completed with error: %(method)s %(path)s -> %(status_code)d (%(duration_ms).2fms) [%(request_id)s]",
                        merged,
                    )
                elif response.status_code >= 400:
                    logger.warning(
                        "Request completed with client error: %(method)s %(path)s -> %(status_code)d (%(duration_ms).2fms) [%(request_id)s]",
                        merged,
                    )
                else:
                    logger.info(
                        "Request completed: %(method)s %(path)s -> %(status_code)d (%(duration_ms).2fms) [%(request_id)s]",
                        merged,
                    )

            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000

            if should_log:
                # P3.1: 修复原混用 %s + %(name)s 的 bug
                logger.error(
                    "Request failed: %(method)s %(path)s -> EXCEPTION (%(exception)s) (%(duration_ms).2fms) [%(request_id)s]",
                    {
                        **request_info,
                        "exception": type(exc).__name__,
                        "duration_ms": round(duration_ms, 2),
                    },
                )

            raise


class PerformanceMonitorMiddleware(BaseHTTPMiddleware):
    """性能监控中间件，记录慢请求。"""

    def __init__(
        self,
        app: Any,
        slow_request_threshold_ms: float = 1000.0,
    ):
        """初始化中间件。

        Args:
            app: FastAPI 应用
            slow_request_threshold_ms: 慢请求阈值（毫秒）
        """
        super().__init__(app)
        self.slow_request_threshold_ms = slow_request_threshold_ms

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """处理请求。"""
        start_time = time.perf_counter()
        request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000

        if duration_ms > self.slow_request_threshold_ms:
            # P3.1: 修复 %(duration).2fms 占位符（duration_ms 是数字，正确使用 %(duration_ms).2fms）
            logger.warning(
                "Slow request detected: %(method)s %(path)s took %(duration_ms).2fms [%(request_id)s]",
                {
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "request_id": request_id,
                },
            )

        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
        return response
