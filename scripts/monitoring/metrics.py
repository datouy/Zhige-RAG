"""可观测性指标收集（无外部依赖）。"""
from __future__ import annotations
import time
from typing import Dict
from collections import defaultdict, deque
from threading import Lock

# 每个直方图最多保留的数据点。observe() 之前无上限地 append，长跑进程
# （服务常驻数周）内存会持续增长；改成有界滑动窗口，语义上也更贴近
# "近期表现"而非"自启动以来的全量样本"。
_MAX_HISTOGRAM_SAMPLES = 1000


class MetricsCollector:
    """轻量级指标收集器。

    提供 counters / histograms / gauges 三种指标类型。
    线程安全，使用双检查锁实现单例。
    """

    _lock = Lock()
    _instance = None

    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.histograms: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=_MAX_HISTOGRAM_SAMPLES)
        )
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
                    # 只排序一次：原实现对每个百分位各排一次，样本多时是
                    # O(n log n) 的重复开销。
                    ordered = sorted(values)
                    n = len(ordered)
                    hist_summary[name] = {
                        "count": n,
                        "avg": sum(ordered) / n,
                        "min": ordered[0],
                        "max": ordered[-1],
                        "p50": ordered[n // 2],
                        "p95": ordered[int(n * 0.95)] if n > 20 else ordered[-1],
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
