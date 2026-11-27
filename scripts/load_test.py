#!/usr/bin/env python
"""由容量模型驱动的真实 HTTP 压测。

请求速率和并发来自 ``scripts/capacity_model.py``，不把 DAU 直接当作 QPS。
压测客户端为每个请求生成独立的基准测试网段地址，用于避免同一个前端用户
的 60/min 配额掩盖服务容量；限流本身另有专项测试。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from scripts.capacity_model import build_model
except ModuleNotFoundError:  # direct execution: python scripts/load_test.py
    from capacity_model import build_model


# 本机压测必须直连被测服务。Windows 上继承的全局 HTTP_PROXY 可能指向
# 已失效端口，urllib 会静默把 loopback 请求送往代理，污染延迟和错误率。
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # nearest-rank：n=1 时 P99 不会越界，也比向下取整更保守。
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * p) - 1))
    return ordered[index]


def _benchmark_ip(index: int) -> str:
    # 10/8 私有网段仅作为 X-Forwarded-For 的压测标识，不用于实际路由。
    # 使用约 1600 万个独立桶，避免不同轮次或长稳测试复用 IP 后撞上
    # 60/min 用户配额，把限流结果误判成服务吞吐。
    value = index % 16_777_216
    return f"10.{(value >> 16) & 0xFF}.{(value >> 8) & 0xFF}.{value & 0xFF}"


def _request(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None,
    timeout: float,
    client_index: int,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body else None
    headers = {
        "X-Forwarded-For": _benchmark_ip(client_index),
        "User-Agent": "zhige-capacity-test/1.0",
    }
    if extra_headers:
        headers.update(extra_headers)
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method=method,
        headers=headers,
    )
    started = time.perf_counter()
    payload = b""
    try:
        with _DIRECT_OPENER.open(request, timeout=timeout) as response:
            payload = response.read()
            status = response.status
            error = ""
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        status = exc.code
        error = payload[:300].decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        status = 0
        error = str(exc)[:500]
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "latency_ms": elapsed_ms,
        "status": status,
        "error": error,
        "path": path,
        "response_bytes": len(payload),
    }


def _workload_plan(
    *,
    profile: str,
    day: str,
    workload: str,
    phase: str,
    duration: float,
    burst_factor: float,
    query: str,
    current_docs: int,
) -> dict[str, Any]:
    model = build_model(current_docs=current_docs)
    metrics = model["scenarios"][profile][day]
    rate_multiplier = burst_factor if phase == "burst" else 1.0
    if workload == "search":
        modeled_qps = float(metrics["search_qps_peak_hour"])
        concurrency_key = "design_retrieval_concurrency"
    elif workload == "chat":
        modeled_qps = float(metrics["chat_qps_peak_hour"])
        concurrency_key = "design_generation_concurrency"
    else:
        modeled_qps = float(metrics["query_qps_peak_hour"])
        concurrency_key = "design_retrieval_concurrency"
    modeled_qps = max(0.01, modeled_qps)
    target_qps = modeled_qps * rate_multiplier
    concurrency = max(
        1,
        math.ceil(float(metrics[concurrency_key]) * rate_multiplier),
    )
    if phase == "ramp":
        # 线性爬坡的积分正好是恒定速率的一半；至少保留一个请求。
        total_requests = max(1, math.ceil(modeled_qps * duration / 2))
    else:
        total_requests = max(1, math.ceil(target_qps * duration))
    return {
        "model_version": model["model_version"],
        "profile": profile,
        "day": day,
        "workload": workload,
        "phase": phase,
        "modeled_qps": modeled_qps,
        "rate_multiplier": rate_multiplier,
        "target_qps": target_qps,
        "concurrency": concurrency,
        "duration_s": duration,
        "planned_requests": total_requests,
        "query": query,
    }


def _session_headers(cookie_files: list[Path]) -> list[dict[str, str]]:
    headers = []
    for path in cookie_files:
        try:
            cookie = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"无法读取会话 Cookie 文件: {path.name}") from exc
        if not cookie or "\r" in cookie or "\n" in cookie:
            raise ValueError(f"会话 Cookie 文件格式无效: {path.name}")
        headers.append({"Cookie": cookie})
    return headers


def _run(
    plan: dict[str, Any],
    base_url: str,
    timeout: float,
    session_headers: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    total = int(plan["planned_requests"])
    duration = float(plan["duration_s"])
    concurrency = int(plan["concurrency"])
    phase = str(plan["phase"])
    workload = str(plan["workload"])
    query = str(plan["query"])
    client_offset = random.SystemRandom().randrange(16_000_000)
    sessions = session_headers or [None]

    # 健康检查不会初始化 RAG/LLM。用两条唯一的真实业务请求预热各个 worker，
    # 预热结果不进入测量；否则首请求冷启动会被误算成稳定态容量。
    warmup_results: list[dict[str, Any]] = []
    for index in range(4):
        warmup_query = f"{query}（预热 {index + 1}）"
        if workload == "search" or (workload == "mixed" and index % 2 == 0):
            method, path = "POST", "/api/v1/search"
            body = {"query": warmup_query, "top_k": 5}
        else:
            method, path = "POST", "/api/v1/chat"
            body = {"query": warmup_query, "top_k": 5, "direct": False}
        warmup_results.append(
            _request(
                base_url,
                method,
                path,
                body,
                timeout,
                client_offset + index,
                sessions[index % len(sessions)],
            )
        )
    failed_warmups = [item for item in warmup_results if item["status"] != 200]
    if failed_warmups:
        raise RuntimeError(
            "真实业务预热失败；请检查团队会话、可信代理与依赖状态: "
            + ", ".join(str(item["status"]) for item in failed_warmups)
        )

    results: list[dict[str, Any]] = []
    schedule_lag_s: list[float] = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for index in range(total):
            if phase == "ramp":
                deadline = wall_start + duration * math.sqrt((index + 1) / total)
            else:
                deadline = wall_start + index / float(plan["target_qps"])
            wait = deadline - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            request_query = f"{query}（压测编号 {index + 1}）"
            if workload == "search" or (
                workload == "mixed" and index % 4 == 0
            ):
                method, path = "POST", "/api/v1/search"
                body = {"query": request_query, "top_k": 5}
            else:
                method, path = "POST", "/api/v1/chat"
                body = {"query": request_query, "top_k": 5, "direct": False}
            schedule_lag_s.append(max(0.0, time.perf_counter() - deadline))
            futures.append(
                pool.submit(
                    _request,
                    base_url,
                    method,
                    path,
                    body,
                    timeout,
                    client_offset + index,
                    sessions[index % len(sessions)],
                )
            )
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = max(0.000001, time.perf_counter() - wall_start)
    latencies = [float(item["latency_ms"]) for item in results]
    statuses: dict[str, int] = {}
    for item in results:
        key = str(item["status"])
        statuses[key] = statuses.get(key, 0) + 1
    protected = sum(1 for item in results if item["status"] == 429)
    errors = [
        item
        for item in results
        if item["status"] != 429 and (item["status"] == 0 or item["status"] >= 400)
    ]
    return {
        **plan,
        "warmup_status_codes": [item["status"] for item in warmup_results],
        "warmup_errors": [item["error"] for item in warmup_results if item["status"] >= 400],
        "completed_requests": len(results),
        "elapsed_s": round(elapsed, 3),
        "achieved_completed_rps": round(len(results) / elapsed, 4),
        "status_codes": statuses,
        "protected_429": protected,
        "protected_429_rate": round(protected / len(results), 5) if results else 0.0,
        "service_error_count": len(errors),
        "service_error_rate": round(len(errors) / len(results), 5) if results else 0.0,
        "p50_ms": round(_percentile(latencies, 0.50), 2),
        "p95_ms": round(_percentile(latencies, 0.95), 2),
        "p99_ms": round(_percentile(latencies, 0.99), 2),
        "avg_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "max_ms": round(max(latencies), 2) if latencies else 0.0,
        "schedule_lag_p95_ms": round(_percentile(schedule_lag_s, 0.95) * 1000, 2),
        "sample_errors": [item["error"] for item in errors[:5]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="模型驱动的知阁真实 HTTP 压测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--profile", choices=["low", "base", "high"], default="base")
    parser.add_argument("--day", choices=["normal_day", "peak_day"], default="peak_day")
    parser.add_argument("--workload", choices=["search", "chat", "mixed"], default="search")
    parser.add_argument("--phase", choices=["steady", "ramp", "burst", "soak"], default="steady")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--burst-factor", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, help="只允许调高模型并发，用于发现饱和点")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--current-docs", type=int, default=1000)
    parser.add_argument("--query", default="一线城市差旅住宿每晚的报销上限是多少？")
    parser.add_argument(
        "--cookie-file",
        action="append",
        default=[],
        help="包含 Cookie 请求头的文件；可重复传入多个团队会话，不输出 Cookie 内容",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration 必须大于 0")
    if args.burst_factor < 1:
        parser.error("--burst-factor 必须不小于 1")

    plan = _workload_plan(
        profile=args.profile,
        day=args.day,
        workload=args.workload,
        phase=args.phase,
        duration=args.duration,
        burst_factor=args.burst_factor,
        query=args.query,
        current_docs=args.current_docs,
    )
    if args.concurrency:
        if args.concurrency < plan["concurrency"]:
            parser.error("压测并发不得低于模型设计值；调低会把排队误当作容量")
        plan["model_concurrency"] = plan["concurrency"]
        plan["concurrency"] = args.concurrency

    sessions = _session_headers([Path(item) for item in args.cookie_file])
    summary = _run(plan, args.base_url, args.timeout, sessions)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if summary["service_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
