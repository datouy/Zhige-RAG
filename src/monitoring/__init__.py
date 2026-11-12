"""可观测性监控模块。

提供指标收集和分布式追踪功能：
- Prometheus 指标（``metrics.py``，依赖 prometheus_client）
- 零依赖的自定义指标（``custom_metrics.py``，prometheus 不可用时的降级）
- OpenTelemetry 链路追踪
"""
from __future__ import annotations

from .metrics import PrometheusMetrics, metrics_collector, track_request_metrics
from .custom_metrics import MetricsCollector
from .tracing import TracingManager, tracing_manager, setup_tracing, trace_span, span, get_current_span

__all__ = [
    "PrometheusMetrics",
    "metrics_collector",
    "track_request_metrics",
    "MetricsCollector",
    "TracingManager",
    "tracing_manager",
    "setup_tracing",
    "trace_span",
    "span",
    "get_current_span",
]
