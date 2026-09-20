"""Prometheus 指标收集模块。

提供符合 Prometheus 格式的指标，用于监控系统性能和健康状态。

指标类型：
- Counter: 只增不减的计数器
- Histogram: 直方图，统计分布
- Gauge: 可增可减的仪表

运行方式：
1. 启动应用后访问 /api/metrics 获取 Prometheus 格式指标
2. 或配置 Prometheus server 定期拉取 /api/metrics 端点
"""
from __future__ import annotations

import time
import psutil
import os
from typing import Callable, Optional
from functools import wraps

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


if PROMETHEUS_AVAILABLE:
    http_requests_total = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )

    http_request_duration_seconds = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
    )

    active_users = Gauge(
        "active_users",
        "Number of currently active users",
    )

    llm_requests_total = Counter(
        "llm_requests_total",
        "Total LLM API requests",
        ["status"],
    )

    llm_request_duration_seconds = Histogram(
        "llm_request_duration_seconds",
        "LLM request latency in seconds",
        ["model"],
        buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
    )

    vector_search_duration_seconds = Histogram(
        "vector_search_duration_seconds",
        "Vector search latency in seconds",
        ["collection"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    )

    documents_ingested_total = Counter(
        "documents_ingested_total",
        "Total documents ingested",
        ["status"],
    )

    chunks_processed_total = Counter(
        "chunks_processed_total",
        "Total chunks processed",
        ["operation"],
    )

    kg_entities_total = Gauge(
        "kg_entities_total",
        "Total number of knowledge graph entities",
    )

    kg_relations_total = Gauge(
        "kg_relations_total",
        "Total number of knowledge graph relations",
    )

    active_connections = Gauge(
        "active_connections",
        "Number of active database connections",
    )

    cache_hits_total = Counter(
        "cache_hits_total",
        "Total cache hits",
    )

    cache_misses_total = Counter(
        "cache_misses_total",
        "Total cache misses",
    )

    error_total = Counter(
        "errors_total",
        "Total errors",
        ["error_type", "component"],
    )


class PrometheusMetrics:
    """Prometheus 指标管理器。"""

    _instance: Optional["PrometheusMetrics"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._process = psutil.Process(os.getpid())
        self._start_time = time.time()

    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float,
    ):
        """记录 HTTP 请求。"""
        if not PROMETHEUS_AVAILABLE:
            return
        endpoint = self._normalize_endpoint(endpoint)
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()
        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration_seconds)

    def record_llm_request(
        self,
        status: str,
        model: str,
        duration_seconds: float,
    ):
        """记录 LLM 请求。"""
        if not PROMETHEUS_AVAILABLE:
            return
        llm_requests_total.labels(status=status).inc()
        llm_request_duration_seconds.labels(model=model).observe(duration_seconds)

    def record_vector_search(
        self,
        collection: str,
        duration_seconds: float,
    ):
        """记录向量搜索。"""
        if not PROMETHEUS_AVAILABLE:
            return
        vector_search_duration_seconds.labels(collection=collection).observe(duration_seconds)

    def record_document_ingest(self, status: str, count: int = 1):
        """记录文档入库。"""
        if not PROMETHEUS_AVAILABLE:
            return
        documents_ingested_total.labels(status=status).inc(count)

    def record_chunk_processed(self, operation: str, count: int = 1):
        """记录处理的分块。"""
        if not PROMETHEUS_AVAILABLE:
            return
        chunks_processed_total.labels(operation=operation).inc(count)

    def update_kg_stats(self, entities: int, relations: int):
        """更新知识图谱统计。"""
        if not PROMETHEUS_AVAILABLE:
            return
        kg_entities_total.set(entities)
        kg_relations_total.set(relations)

    def update_active_users(self, count: int):
        """更新活跃用户数。"""
        if not PROMETHEUS_AVAILABLE:
            return
        active_users.set(count)

    def update_active_connections(self, count: int):
        """更新活跃连接数。"""
        if not PROMETHEUS_AVAILABLE:
            return
        active_connections.set(count)

    def record_cache_hit(self):
        """记录缓存命中。"""
        if not PROMETHEUS_AVAILABLE:
            return
        cache_hits_total.inc()

    def record_cache_miss(self):
        """记录缓存未命中。"""
        if not PROMETHEUS_AVAILABLE:
            return
        cache_misses_total.inc()

    def record_error(self, error_type: str, component: str):
        """记录错误。"""
        if not PROMETHEUS_AVAILABLE:
            return
        error_total.labels(error_type=error_type, component=component).inc()

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """规范化端点路径（移除动态参数）。"""
        parts = endpoint.strip("/").split("/")
        normalized = []
        for i, part in enumerate(parts):
            if part.isdigit() or len(part) == 36 or "-" in part:
                normalized.append("{id}")
            else:
                normalized.append(part)
        return "/" + "/".join(normalized)

    def get_system_metrics(self) -> dict:
        """获取系统级指标。"""
        try:
            memory = self._process.memory_info()
            cpu_percent = self._process.cpu_percent(interval=0.1)

            return {
                "process_cpu_percent": cpu_percent,
                "process_memory_mb": memory.rss / 1024 / 1024,
                "process_memory_percent": self._process.memory_percent(),
                "process_threads": self._process.num_threads(),
                "process_uptime_seconds": time.time() - self._start_time,
                "system_memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "system_memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
                "system_memory_percent": psutil.virtual_memory().percent,
                "system_cpu_percent": psutil.cpu_percent(interval=0.1),
                "system_disk_usage_percent": psutil.disk_usage("/").percent if os.name != "nt" else psutil.disk_usage("C:").percent,
            }
        except Exception:
            return {}

    def generate_metrics_output(self) -> tuple[bytes, str]:
        """生成 Prometheus 格式的指标输出。"""
        if not PROMETHEUS_AVAILABLE:
            return b"", "text/plain"

        metrics_output = generate_latest()
        return metrics_output, CONTENT_TYPE_LATEST


def track_request_metrics(metric_name: str = "request"):
    """装饰器：自动跟踪函数执行指标。"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if PROMETHEUS_AVAILABLE:
                    http_request_duration_seconds.labels(
                        method="INTERNAL",
                        endpoint=metric_name,
                    ).observe(duration)
                return result
            except Exception as e:
                if PROMETHEUS_AVAILABLE:
                    error_total.labels(
                        error_type=type(e).__name__,
                        component=metric_name,
                    ).inc()
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                if PROMETHEUS_AVAILABLE:
                    http_request_duration_seconds.labels(
                        method="INTERNAL",
                        endpoint=metric_name,
                    ).observe(duration)
                return result
            except Exception as e:
                if PROMETHEUS_AVAILABLE:
                    error_total.labels(
                        error_type=type(e).__name__,
                        component=metric_name,
                    ).inc()
                raise

        if hasattr(func, "__wrapped__"):
            return async_wrapper if hasattr(func, "__await__") else sync_wrapper
        try:
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
        except Exception:
            pass
        return sync_wrapper

    return decorator


metrics_collector = PrometheusMetrics()
