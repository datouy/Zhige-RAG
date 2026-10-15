"""接口功能冒烟测试：通过 HTTP 调用真实运行的服务，逐接口验证是否正常。

服务不需要登录，所有请求直接打到业务接口即可。

用法（在 ChineseRAGKB 项目根目录运行）：
    .venv/Scripts/python.exe scripts/interface_smoke.py
输出：控制台摘要 + logs/interface_smoke.log 详细报告
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8000"
LOG = ROOT / "logs" / "interface_smoke.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

# 登录 / 注册已经彻底移除，这几条路径应稳定返回 404
REMOVED_AUTH_PATHS = (
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/refresh",
    "/api/v1/auth/me",
)

results: list[dict] = []


def call(method: str, path: str, body=None, timeout: int = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, raw, time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.time() - t
    except Exception as e:  # noqa: BLE001
        return "ERR", str(e).encode("utf-8", "replace"), time.time() - t


def check(name: str, method: str, path: str, body=None,
          timeout: int = 30, expect: tuple = (200,)):
    st, raw, dt = call(method, path, body=body, timeout=timeout)
    try:
        payload = json.loads(raw) if raw else None
    except Exception:
        payload = None
    ok = st in expect
    results.append({"name": name, "method": method, "path": path,
                    "status": st, "dt": dt, "ok": ok,
                    "detail": _brief(payload, raw)})
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name:28s} {method:5s} {path:34s} {st} ({dt:.2f}s)", flush=True)
    return st, payload, dt


def _brief(payload, raw, limit: int = 200):
    if isinstance(payload, dict):
        keys = list(payload.keys())[:8]
        return "keys=" + ",".join(keys)
    if isinstance(payload, list):
        return f"list[{len(payload)}]"
    if raw:
        return raw[:limit].decode("utf-8", "replace")
    return ""


def main() -> int:
    print("=" * 72, flush=True)
    print(f"接口冒烟测试  {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"目标: {BASE}", flush=True)
    print("=" * 72, flush=True)

    # 1) 健康检查与运维端点
    check("健康检查 health", "GET", "/api/health", timeout=15)
    check("就绪探针 ready", "GET", "/api/ready", timeout=15)
    check("Prometheus 指标", "GET", "/api/metrics", timeout=15)
    check("OpenAPI 文档", "GET", "/openapi.json", timeout=15)

    # 2) 登录 / 注册相关端点应已全部移除
    for path in REMOVED_AUTH_PATHS:
        check(f"已移除 {path.rsplit('/', 1)[-1]}", "POST", path, {}, timeout=15, expect=(404, 405))

    # 3) 业务接口
    check("文档列表", "GET", "/api/v1/documents", timeout=60)
    check("语义检索 search", "POST", "/api/v1/search",
          {"query": "RAG 是什么", "top_k": 3}, timeout=60)
    check("运行配置 config", "GET", "/api/v1/config", timeout=15)
    check("订阅档位 tiers", "GET", "/api/v1/subscription/tiers", timeout=15)
    check("当前订阅 current", "GET", "/api/v1/subscription/current", timeout=15)
    check("反馈统计 stats", "GET", "/api/v1/feedback/stats", timeout=15)
    check("长期记忆 memory", "GET", "/api/v1/memory", timeout=15)
    check("Agent 工具列表", "GET", "/api/v1/agent/tools", timeout=15)
    check("知识库列表 kb", "GET", "/api/v1/kb", timeout=15)
    check("模型设置 settings", "GET", "/api/v1/settings/model", timeout=15)

    # 4) 知识图谱（列表/查询类，不触发 LLM）
    check("KG 统计 stats", "GET", "/api/v1/kg/stats", timeout=15)
    check("KG 实体 entities", "GET", "/api/v1/kg/entities", timeout=15)
    check("KG 关系 relations", "GET", "/api/v1/kg/relations", timeout=15)
    check("KG 查询 query", "POST", "/api/v1/kg/query",
          {"cypher": "MATCH (n) RETURN n LIMIT 3", "params": {}}, timeout=15)

    # 5) 聊天问答（会触发真实 LLM 加载，给足超时）
    print("\n[...] 聊天接口将触发 LLM 加载，可能耗时数分钟 ...", flush=True)
    st, body, dt = check("聊天问答 chat", "POST", "/api/v1/chat",
                         {"query": "用一句话介绍中文 RAG 知识库", "top_k": 3},
                         timeout=600, expect=(200,))
    if isinstance(body, dict):
        ans = body.get("answer") or body.get("text") or ""
        print(f"      回答预览: {str(ans)[:120]}", flush=True)

    return _finish()


def _finish() -> int:
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    print("\n" + "=" * 72, flush=True)
    print(f"结果: {passed}/{total} 通过", flush=True)
    failed = [r for r in results if not r["ok"]]
    if failed:
        print("失败项:", flush=True)
        for r in failed:
            print(f"  - {r['name']} {r['method']} {r['path']} -> {r['status']} | {r['detail']}", flush=True)
    print("=" * 72, flush=True)
    # 写详细日志
    try:
        with open(LOG, "w", encoding="utf-8") as f:
            f.write(f"接口冒烟测试  {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"目标 {BASE}\n")
            f.write(f"结果 {passed}/{total}\n\n")
            for r in results:
                f.write(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['name']} "
                        f"{r['method']} {r['path']} -> {r['status']} "
                        f"({r['dt']:.2f}s) | {r['detail']}\n")
    except Exception as e:  # noqa: BLE001
        print(f"写日志失败: {e}", flush=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
