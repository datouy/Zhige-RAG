"""可观测性指标收集（无外部依赖）。"""
from __future__ import annotations
import time
from typing import Dict
from collections import defaultdict
from threading import Lock


class MetricsCollector:
    """轻量级指标收集器。

    提供 counters / histograms / gauges 三种指标类型。
    线程安全，使用双检查锁实现单例。
    """

    _lock = Lock()
    _instance = None

    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.histograms: Dict[str, list] = defaultdict(list)
        self.gauges: Dict[str, float] = {}
        self._start = time.time()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = MetricsCollector()
        return cls._instance

    def inc(self, name: str, value: int = 1) -> None:
        """递增计数器。"""
        with self._lock:
            self.counters[name] += value

    def observe(self, name: str, value: float) -> None:
        """记录直方图数据点。"""
        with self._lock:
            self.histograms[name].append(value)

    def gauge(self, name: str, value: float) -> None:
        """设置仪表值（最新值）。"""
        with self._lock:
            self.gauges[name] = value

    def get_summary(self) -> dict:
        """获取所有指标摘要。"""
        with self._lock:
            uptime = time.time() - self._start
            hist_summary = {}
            for name, values in self.histograms.items():
                if values:
                    hist_summary[name] = {
                        "count": len(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "p50": sorted(values)[len(values) // 2],
                        "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 20 else max(values),
                    }
            return {
                "uptime_seconds": round(uptime, 2),
                "counters": dict(self.counters),
                "histograms": hist_summary,
                "gauges": dict(self.gauges),
            }

    def reset(self) -> None:
        """重置所有指标（测试用）。"""
        with self._lock:
            self.counters.clear()
            self.histograms.clear()
            self.gauges.clear()
            self._start = time.time()
