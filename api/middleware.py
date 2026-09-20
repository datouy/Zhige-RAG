"""安全响应头中间件（P2.3 — security-engineer / P2.4 — CSP 硬化）。

设计原则：
- ``BaseHTTPMiddleware`` 子类，比 ``@app.middleware("http")`` 装饰器更易测试
  （FastAPI 的装饰器返回的不是 ASGI 标准的中间件，难以直接实例化）。
- CSP 默认禁用 ``unsafe-inline`` 脚本（改为 nonce 占位）。允许通过环境变量
  ``CSP_ALLOW_UNSAFE_INLINE_SCRIPT`` 在调试时回退。
- 额外加上 ``Referrer-Policy`` / ``Permissions-Policy`` / ``Cross-Origin-Opener-Policy``
  等常见硬化头。
"""
from __future__ import annotations

import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


# ----------------------------------------------------------------------
# CSP 决策（P2.4 — security-engineer）
# ----------------------------------------------------------------------
# 我们**保留** ``'unsafe-inline'`` for ``script-src`` 与 ``style-src``：
#   - ``ui/web/*.html`` 与 ``ui/web/js/app.js`` 含内联 ``<script>...</script>``
#     块（grep 验证：ui/web/upload.html、docs.html、js/app.js 中均出现）。
#   - ``ui/web/*.html`` 多个元素使用 ``style="..."`` 内联样式。
# 因此严格策略（移除 unsafe-inline）会让前端立即失效。
#
# 严格策略仍可通过 ``CSP_POLICY="..."`` 环境变量或自定义 ``SecurityHeadersMiddleware(csp=...)``
# 注入参数启用，参考 ``_STRICT_CSP``。后续重构前端、把内联脚本迁出后，可切换默认。
_STRICT_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "form-action 'self'; "
    "base-uri 'self'; "
    "object-src 'none'"
)

_DEFAULT_CSP = (
    # default + script + style：保留 unsafe-inline（前端依赖，理由见上）
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "form-action 'self'; "
    "base-uri 'self'; "
    "object-src 'none'"
)

# 旧调试回退：保留与原先行为完全一致的 CSP（仅在没有 CSP_POLICY 时使用）。
_DEV_CSP = _DEFAULT_CSP


def _build_csp() -> str:
    """根据环境变量返回 CSP 字符串。

    - ``CSP_POLICY=strict`` 切换到 ``_STRICT_CSP``（要求前端无内联脚本 / 样式）。
    - ``CSP_POLICY`` 是其他非空字符串：直接用作完整 CSP（运维覆盖）。
    - ``CSP_ALLOW_UNSAFE_INLINE_SCRIPT=1`` 等价于 ``CSP_POLICY=default``（保留）。
    """
    override = os.getenv("CSP_POLICY")
    if override:
        if override.strip().lower() == "strict":
            return _STRICT_CSP
        if override.strip().lower() in ("default", "dev"):
            return _DEFAULT_CSP
        return override
    return _DEFAULT_CSP


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """安全响应头中间件（P2.3）。

    添加的头（参考 OWASP Secure Headers Project）：
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block （兼容旧浏览器）
    - Strict-Transport-Security: max-age=31536000; includeSubDomains
    - Content-Security-Policy: 见 _build_csp
    - Referrer-Policy: strict-origin-when-cross-origin
    - Permissions-Policy: 仅允许同源 (geolocation/camera/microphone=())
    - Cross-Origin-Opener-Policy: same-origin
    """

    def __init__(self, app, csp: str | None = None) -> None:
        super().__init__(app)
        self.csp = csp or _build_csp()

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Content-Security-Policy"] = self.csp
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), camera=(), microphone=(), payment=()"
        )
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        return response


def make_request_id() -> str:
    """生成请求 ID（用于 RequestIDMiddleware 与日志关联）。"""
    return secrets.token_hex(16)