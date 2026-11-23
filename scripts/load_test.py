"""真实 HTTP 端到端并发压测（不依赖第三方库）。

示例：
    python scripts/load_test.py \
      --base-url https://zhige.example.com \
      --username admin \
      --password-file /run/secrets/loadtest_password \
      --endpoint /api/v1/search \
      --query "企业账号密码锁定规则" \
      --requests 500 --concurrency 30

本地回环模式可加 --no-auth。脚本不会打印密码或访问 Token。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * max(0.0, min(1.0, p))
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _login(base_url: str, username: str, password: str, timeout: float) -> str:
    payload = json.dumps(
        {"username": username, "password": password}, ensure_ascii=False
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/v1/auth/login",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    token = str(data.get("access_token") or "")
    if not token:
        raise RuntimeError("登录响应没有 access_token")
    return token


def _request(
    url: str,
    *,
    method: str,
    payload: Optional[Dict[str, Any]],
    token: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    data = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url, data=data, headers=headers, method=method.upper()
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "body_bytes": len(body),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "body_bytes": 0,
            "error": f"HTTP {exc.code}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": 0,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "body_bytes": 0,
            "error": type(exc).__name__,
        }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="知阁真实 HTTP 并发压测")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--username", default="")
    parser.add_argument("--password-file", default="")
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--endpoint", default="/api/v1/search")
    parser.add_argument("--method", default="POST")
    parser.add_argument("--query", default="企业账号密码锁定规则")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--max-p95-latency-ms", type=float, default=5000.0)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    token = None
    if not args.no_auth:
        if not args.username or not args.password_file:
            print("企业模式压测需要 --username 与 --password-file", file=sys.stderr)
            return 2
        password = Path(args.password_file).read_text(encoding="utf-8").strip()
        try:
            token = _login(args.base_url, args.username, password, args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"压测登录失败：{type(exc).__name__}", file=sys.stderr)
            return 2

    base = args.base_url.rstrip("/")
    endpoint = args.endpoint if args.endpoint.startswith("/") else "/" + args.endpoint
    url = base + endpoint
    method = args.method.upper()
    payload = None
    if method in {"POST", "PUT", "PATCH"}:
        payload = {"query": args.query, "top_k": 5}

    for _ in range(max(0, args.warmup)):
        _request(
            url, method=method, payload=payload, token=token, timeout=args.timeout
        )

    started = time.perf_counter()
    results: list[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                _request,
                url,
                method=method,
                payload=payload,
                token=token,
                timeout=args.timeout,
            )
            for _ in range(max(1, args.requests))
        ]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = max(1e-9, time.perf_counter() - started)

    latencies = [float(item["latency_ms"]) for item in results]
    successes = sum(1 for item in results if item["ok"])
    report = {
        "endpoint": url,
        "requests": len(results),
        "concurrency": args.concurrency,
        "success": successes,
        "failed": len(results) - successes,
        "success_rate": successes / len(results) if results else 0.0,
        "throughput_rps": len(results) / elapsed,
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "status_counts": {
            str(status): sum(1 for item in results if item["status"] == status)
            for status in sorted({item["status"] for item in results})
        },
    }
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["success_rate"] < args.min_success_rate:
        print("压测失败：成功率低于阈值", file=sys.stderr)
        return 1
    p95 = report["latency_ms"]["p95"] or 0.0
    if p95 > args.max_p95_latency_ms:
        print("压测失败：P95 延迟超过阈值", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
