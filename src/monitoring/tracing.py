"""OpenTelemetry 分布式追踪模块。

提供链路追踪能力，支持将追踪数据导出到：
- Jaeger (默认)
- Zipkin
- OTLP Collector
- Console (开发调试用)

使用方式：
1. 在应用启动时调用 setup_tracing()
2. 使用 @trace_span 装饰器标记函数
3. 配置 OTEL_EXPORTER 环境变量选择导出器

环境变量：
- OTEL_SERVICE_NAME: 服务名称
- OTEL_EXPORTER_TYPE: 导出器类型 (jaeger|zipkin|otlp|console)
- OTEL_EXPORTER_ENDPOINT: 导出器端点 URL
- OTEL_EXPORTER_JAEGER_AGENT_HOST: Jaeger Agent 主机
- OTEL_EXPORTER_JAEGER_AGENT_PORT: Jaeger Agent 端口
"""
from __future__ import annotations

import os
import time
import functools
from typing import Callable, Optional, Any
from contextlib import contextmanager

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

if OTEL_AVAILABLE:
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        JAEGER_AVAILABLE = True
    except ImportError:
        JAEGER_AVAILABLE = False

    try:
        from opentelemetry.exporter.zipkin.thrift import ZipkinExporter
        ZIPKIN_AVAILABLE = True
    except ImportError:
        ZIPKIN_AVAILABLE = False

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        OTLP_AVAILABLE = True
    except ImportError:
        OTLP_AVAILABLE = False


class TracingManager:
    """追踪管理器单例。"""

    _instance: Optional["TracingManager"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if TracingManager._initialized:
            return
        TracingManager._initialized = True
        self._provider: Optional[TracerProvider] = None
        self._tracer: Optional[trace.Tracer] = None
        self._propagator = TraceContextTextMapPropagator()

    def setup_tracing(
        self,
        service_name: str = "ChineseRAGKB",
        exporter_type: str = None,
        endpoint: str = None,
    ) -> "TracingManager":
        """初始化追踪系统。

        Args:
            service_name: 服务名称
            exporter_type: 导出器类型 (jaeger|zipkin|otlp|console)
            endpoint: 导出器端点 URL
        """
        if not OTEL_AVAILABLE:
            print("Warning: OpenTelemetry not available, tracing disabled")
            return self

        exporter_type = exporter_type or os.getenv("OTEL_EXPORTER_TYPE", "console")
        endpoint = endpoint or os.getenv("OTEL_EXPORTER_ENDPOINT", "")

        resource = Resource.create({
            "service.name": service_name,
            "service.version": os.getenv("APP_VERSION", "0.3.0"),
            "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development"),
        })

        self._provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self._provider)

        exporter = self._create_exporter(exporter_type, endpoint)
        if exporter:
            self._provider.add_span_processor(BatchSpanProcessor(exporter))

        self._tracer = trace.get_tracer(service_name)

        return self

    def _create_exporter(self, exporter_type: str, endpoint: str):
        """根据类型创建导出器。"""
        if not OTEL_AVAILABLE:
            return None

        if exporter_type == "console":
            return ConsoleSpanExporter()

        if exporter_type == "jaeger":
            if not JAEGER_AVAILABLE:
                print("Warning: Jaeger exporter not available")
                return ConsoleSpanExporter()
            return JaegerExporter(
                agent_host_name=os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST", "localhost"),
                agent_port=int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831")),
            )

        if exporter_type == "zipkin":
            if not ZIPKIN_AVAILABLE:
                print("Warning: Zipkin exporter not available")
                return ConsoleSpanExporter()
            return ZipkinExporter(
                endpoint=endpoint or "http://localhost:9411/api/v2/spans",
            )

        if exporter_type == "otlp":
            if not OTLP_AVAILABLE:
                print("Warning: OTLP exporter not available")
                return ConsoleSpanExporter()
            return OTLPSpanExporter(
                endpoint=endpoint or "http://localhost:4317",
                insecure=True,
            )

        return ConsoleSpanExporter()

    def get_tracer(self) -> trace.Tracer:
        """获取追踪器实例。"""
        if self._tracer is None:
            self._tracer = trace.get_tracer("ChineseRAGKB")
        return self._tracer

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    ):
        """创建追踪跨度上下文管理器。"""
        tracer = self.get_tracer()
        with tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes or {},
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(
                    Status(StatusCode.ERROR, str(e)),
                )
                span.record_exception(e)
                raise

    def trace_function(
        self,
        name: str = None,
        attributes: dict = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    ) -> Callable:
        """装饰器：追踪函数执行。"""
        def decorator(func: Callable) -> Callable:
            span_name = name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.span(span_name, attributes, kind):
                    return func(*args, **kwargs)

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with self.span(span_name, attributes, kind):
                    return await func(*args, **kwargs)

            try:
                import asyncio
                if asyncio.iscoroutinefunction(func):
                    return async_wrapper
            except Exception:
                pass
            return sync_wrapper

        return decorator

    def add_span_attributes(self, **attributes):
        """为当前跨度添加属性。"""
        span = trace.get_current_span()
        if span and span.is_recording():
            for key, value in attributes.items():
                span.set_attribute(key, value)

    def record_exception(self, exception: Exception):
        """记录异常到当前跨度。"""
        span = trace.get_current_span()
        if span and span.is_recording():
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))

    def inject_context(self, carrier: dict):
        """注入追踪上下文到载体（如 HTTP headers）。"""
        self._propagator.inject(carrier)

    def extract_context(self, carrier: dict):
        """从载体中提取追踪上下文。"""
        return self._propagator.extract(carrier)

    def shutdown(self):
        """关闭追踪系统。"""
        if self._provider:
            self._provider.shutdown()
            self._provider = None
            self._tracer = None


tracing_manager = TracingManager()


def setup_tracing(
    service_name: str = "ChineseRAGKB",
    exporter_type: str = None,
    endpoint: str = None,
) -> TracingManager:
    """便捷函数：设置追踪系统。"""
    return tracing_manager.setup_tracing(service_name, exporter_type, endpoint)


def trace_span(
    name: str = None,
    attributes: dict = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
):
    """装饰器：创建追踪跨度。"""
    return tracing_manager.trace_function(name, attributes, kind)


@contextmanager
def span(
    name: str,
    attributes: dict = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
):
    """上下文管理器：创建追踪跨度。"""
    with tracing_manager.span(name, attributes, kind) as s:
        yield s


def get_current_span():
    """获取当前追踪跨度。"""
    if not OTEL_AVAILABLE:
        return None
    return trace.get_current_span()
