"""真实 HTTP 压测基线（零新依赖，纯 stdlib）。

用于在**真实运行中的服务**上测延迟分位数 / 错误率 / 吞吐，建立性能基线，
而不是依赖 TestClient 的内存调用（TestClient 不经过真实 socket，无法反映
并发下的网络与事件循环调度）。

用法（先启动服务，再另开终端）：
    python scripts/load_test.py --base-url http://localhost:8000         --concurrency 20 --requests 500

默认打的端点：
- GET /api/health/live   轻量存活探针（新增，不穿透 DB/Redis/Chroma）
- GET /api/health        全量体检（带 TTL 缓存）
- POST /api/v1/search    检索（走 embedding 编码 + Chroma）

每个请求都记录耗时；最后输出 P50/P95/P99、错误率与吞吐。
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def _request(base_url: str, method: str, path: str, body: dict | None, timeout: float) -> tuple[float, int, str]:
    url = base_url.rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            return (time.perf_counter() - start) * 1000, resp.status, ""
    except urllib.error.HTTPError as exc:
        return (time.perf_counter() - start) * 1000, exc.code, str(exc)
    except Exception as exc:  # noqa: BLE001
        return (time.perf_counter() - start) * 1000, 0, str(exc)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    idx = min(len(data) - 1, int(len(data) * p))
    return data[idx]


def _run_plan(base_url: str, plan: list[tuple[str, str, dict | None]], concurrency: int, timeout: float) -> dict:
    results: list[tuple[float, int, str]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_request, base_url, m, p, b, timeout) for (m, p, b) in plan]
        for fut in as_completed(futures):
            results.append(fut.result())
    elapsed = time.perf_counter() - started

    latencies = [r[0] for r in results]
    errors = [r for r in results if r[2] or r[1] >= 400]
    return {
        "requests": len(results),
        "elapsed_s": round(elapsed, 3),
        "throughput_rps": round(len(results) / elapsed, 2) if elapsed > 0 else 0.0,
        "error_rate": round(len(errors) / len(results), 4) if results else 0.0,
        "p50_ms": round(_percentile(latencies, 0.50), 2),
        "p95_ms": round(_percentile(latencies, 0.95), 2),
        "p99_ms": round(_percentile(latencies, 0.99), 2),
        "avg_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "max_ms": round(max(latencies), 2) if latencies else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="真实 HTTP 压测基线")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    # 每个端点均摊请求数
    per_endpoint = max(1, args.requests // 3)
    plans = [
        ("GET", "/api/health/live", None),
        ("GET", "/api/health", None),
        ("POST", "/api/v1/search", {"query": "什么是 RAG", "top_k": 5}),
    ]

    print("压测目标：%s  并发=%d  总请求=%d" % (args.base_url, args.concurrency, args.requests))
    print("-" * 60)

    for method, path, body in plans:
        plan = [(method, path, body)] * per_endpoint
        summary = _run_plan(args.base_url, plan, args.concurrency, args.timeout)
        print("%-4s %-22s -> %s" % (method, path, json.dumps(summary, ensure_ascii=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

