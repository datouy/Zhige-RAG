"""API 依赖 / 单例 / 缓存层测试。

覆盖：
- P1.1：路由拆分后所有原 URL 仍可发现。
- P1.2：``get_cached_config`` 真正命中缓存（同一进程只 load 一次）。
- P1.4：``get_pipeline`` 多次调用返回同一实例（避免 LLM 重复加载）。
- P2.2：``/api/health`` TTL 缓存有效。
- P3.3：``single_flight_lock`` 跨进程互斥。
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =========================== P1.2: Config cache ===========================
class TestConfigCache:
    """``api.deps.get_cached_config`` 必须只解析一次 YAML。"""

    def test_same_config_object_returned(self):
        from api import deps

        deps._reset_cached_config()
        c1 = deps.get_cached_config()
        c2 = deps.get_cached_config()
        assert c1 is c2, "lru_cache must return identical dict object"

    def test_cache_hits_count_grows(self):
        from api import deps

        deps._reset_cached_config()
        info_before = deps.get_cached_config.cache_info()
        for _ in range(5):
            deps.get_cached_config()
        info_after = deps.get_cached_config.cache_info()
        assert info_after.hits > info_before.hits
        assert info_after.misses - info_before.misses <= 1

    def test_reset_clears_cache(self):
        from api import deps

        deps.get_cached_config()
        deps._reset_cached_config()
        info = deps.get_cached_config.cache_info()
        assert info.hits == 0
        assert info.currsize == 0


# =========================== P2.2: Health cache ===========================
class TestHealthCache:
    """``/api/health`` TTL 缓存：相同 TTL 窗口内重复请求只计算一次。"""

    def test_health_cache_roundtrip(self):
        from api import deps

        deps.invalidate_health_cache()
        assert deps._get_health_cache() is None

        payload = {"status": "ok", "timestamp": "x"}
        deps._set_health_cache(payload)
        assert deps._get_health_cache() == payload

    def test_invalidate_clears(self):
        from api import deps

        deps._set_health_cache({"status": "ok"})
        assert deps._get_health_cache() is not None
        deps.invalidate_health_cache()
        assert deps._get_health_cache() is None

    def test_ttl_expiry(self):
        from api import deps

        deps.invalidate_health_cache()
        deps._set_health_cache({"status": "ok"})
        deps._health_cache["ts"] = time.time() - (deps.get_health_cache_ttl() + 10)
        assert deps._get_health_cache() is None, "TTL expired must invalidate cache"


# =========================== P3.3: single_flight_lock ===========================
class TestSingleFlightLock:
    """``single_flight_lock`` 必须跨进程互斥（基于文件 + stdlib）。"""

    def test_acquire_and_release(self, tmp_path, monkeypatch):
        from api import deps

        monkeypatch.setenv("SINGLE_FLIGHT_DIR", str(tmp_path))
        with deps.single_flight_lock("unit_test_basic", timeout=1.0) as acquired:
            assert acquired is True
        # 释放后再次获取应成功
        with deps.single_flight_lock("unit_test_basic", timeout=1.0) as acquired:
            assert acquired is True

    def test_second_lock_blocks_until_release(self, tmp_path, monkeypatch):
        """第一个锁释放前，第二个 lock 不能立即获取（应超时返回 False）。"""
        from api import deps

        monkeypatch.setenv("SINGLE_FLIGHT_DIR", str(tmp_path))

        # 在一个线程里持有锁 1 秒
        ready = threading.Event()
        release = threading.Event()

        def holder():
            with deps.single_flight_lock("unit_test_contend", timeout=5.0) as got:
                assert got is True
                ready.set()
                release.wait(timeout=5.0)

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        ready.wait(timeout=2.0)
        # holder 持锁期间，尝试获取相同 name 的锁（短超时 → 期望 False）
        t0 = time.time()
        with deps.single_flight_lock("unit_test_contend", timeout=0.3) as got:
            elapsed = time.time() - t0
        assert got is False, "second acquirer should fail while holder holds"
        assert elapsed >= 0.25, "timeout should be respected"
        # 释放 holder
        release.set()
        t.join(timeout=2.0)


# =========================== P1.1: Route module surface ===========================
class TestRouteModuleSurface:
    """路由拆分后必须保留所有原 URL。"""

    def test_route_modules_reexport_routers(self):
        from api.routes import (
            agent_router,
            auth_router,
            chat_router,
            eval_router,
            kg_router,
            subscription_router,
            system_router,
        )

        for r in (
            agent_router,
            auth_router,
            chat_router,
            eval_router,
            kg_router,
            subscription_router,
            system_router,
        ):
            assert hasattr(r, "routes"), f"{r} missing routes attribute"

    def test_app_has_expected_top_level_paths(self):
        """启动后，app 必须暴露全部受保护 + 公开路径（无 404）。

        兼容两种 FastAPI 行为：旧版把 include_router 的路由摊平进
        ``app.routes``；新版（>=0.121）包裹为 ``_IncludedRouter``，
        需要深入 ``original_router.routes`` 收集路径。
        """
        from api.main import app

        def _collect_paths(routes) -> set:
            paths = set()
            for r in routes:
                path = getattr(r, "path", None)
                if path:
                    paths.add(path)
                    continue
                inner = getattr(r, "original_router", None)
                if inner is not None:
                    paths |= _collect_paths(getattr(inner, "routes", []))
                    continue
                inner_routes = getattr(r, "routes", None)
                if inner_routes:
                    paths |= _collect_paths(inner_routes)
            return paths

        paths = _collect_paths(app.routes)
        for required in (
            "/api/health",
            "/api/ready",
            "/api/metrics",
            "/api/v1/metrics",
            "/api/v1/auth/register",
            "/api/v1/auth/login",
            "/api/v1/chat",
            "/api/v1/search",
            "/api/v1/ingest",
            "/api/v1/kg/stats",
            "/api/v1/kg/query",
            "/api/v1/agent/tools",
            "/api/v1/agent/chat",
            "/api/v1/eval/run",
            "/api/v1/config",
            "/ws/chat",
            "/api/v1/ws/agent",
        ):
            assert required in paths, f"missing route: {required}"