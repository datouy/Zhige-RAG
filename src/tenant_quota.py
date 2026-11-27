"""Tenant-scoped request quotas with Redis and single-process fallback.

Quota state is shared by every API worker when Redis is configured. The
tenant key is the knowledge-base owner, so users sharing a team knowledge base
cannot multiply capacity by creating additional sessions. The local fallback
is intentionally visible as degraded in readiness because per-worker limits
do not enforce a true tenant quota.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    scope: str
    window: str
    limit: int
    used: int
    retry_after: int


_LOCAL: Dict[str, Tuple[int, float]] = {}
_LOCAL_LOCK = threading.Lock()
_REDIS_CLIENT: Any = None
_REDIS_LOCK = threading.Lock()


def enabled() -> bool:
    return os.getenv("ZHIGE_TENANT_QUOTA_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _limit(scope: str, window: str) -> int:
    defaults = {
        ("search", "minute"): 120,
        ("search", "day"): 100000,
        ("chat", "minute"): 60,
        ("chat", "day"): 20000,
        ("ingest", "minute"): 20,
        ("ingest", "day"): 2000,
    }
    env_name = f"ZHIGE_QUOTA_{scope.upper()}_{window.upper()}"
    try:
        return max(0, int(os.getenv(env_name, str(defaults.get((scope, window), 0)) )))
    except (TypeError, ValueError):
        return int(defaults.get((scope, window), 0))


def _redis_url() -> str:
    return (os.getenv("REDIS_URL", "").strip() or os.getenv("RATE_LIMIT_STORAGE_URL", "").strip())


def _redis():
    global _REDIS_CLIENT
    url = _redis_url()
    if not url:
        return None
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    with _REDIS_LOCK:
        if _REDIS_CLIENT is not None:
            return _REDIS_CLIENT
        try:
            import redis

            _REDIS_CLIENT = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
                retry_on_timeout=False,
            )
            _REDIS_CLIENT.ping()
        except Exception:
            _REDIS_CLIENT = None
    return _REDIS_CLIENT


def _tenant_key(tenant: str) -> str:
    return hashlib.sha256(str(tenant).encode("utf-8")).hexdigest()[:24]


def _window_key(tenant: str, scope: str, window: str, now: int) -> str:
    bucket = now // 60 if window == "minute" else now // 86400
    return f"zhige:quota:{window}:{bucket}:{_tenant_key(tenant)}:{scope}"


def _check_redis(tenant: str, scope: str, amount: int) -> QuotaDecision:
    client = _redis()
    minute_limit = _limit(scope, "minute")
    day_limit = _limit(scope, "day")
    now = int(time.time())
    keys = [
        _window_key(tenant, scope, "minute", now),
        _window_key(tenant, scope, "day", now),
    ]
    script = """
local minute = tonumber(redis.call('GET', KEYS[1]) or '0')
local day = tonumber(redis.call('GET', KEYS[2]) or '0')
if minute + ARGV[1] > tonumber(ARGV[2]) then
  return {0, minute, day, 1}
end
if day + ARGV[1] > tonumber(ARGV[3]) then
  return {0, minute, day, 2}
end
local nm = redis.call('INCRBY', KEYS[1], ARGV[1])
if nm == tonumber(ARGV[1]) then redis.call('EXPIRE', KEYS[1], 65) end
local nd = redis.call('INCRBY', KEYS[2], ARGV[1])
if nd == tonumber(ARGV[1]) then redis.call('EXPIRE', KEYS[2], 86405) end
return {1, nm, nd, 0}
"""
    allowed, minute, day, blocked = client.eval(
        script, 2, *keys, int(amount), minute_limit, day_limit
    )
    if bool(int(allowed)):
        return QuotaDecision(True, scope, "minute", minute_limit, int(minute), 0)
    if int(blocked) == 1:
        return QuotaDecision(False, scope, "minute", minute_limit, int(minute), 60)
    return QuotaDecision(False, scope, "day", day_limit, int(day), 3600)


def _check_local(tenant: str, scope: str, amount: int) -> QuotaDecision:
    now = time.time()
    decisions = []
    with _LOCAL_LOCK:
        # Opportunistically remove expired keys; bounded by caller traffic.
        for key, (_, expires_at) in list(_LOCAL.items()):
            if expires_at <= now:
                _LOCAL.pop(key, None)
        for window in ("minute", "day"):
            limit = _limit(scope, window)
            key = _window_key(tenant, scope, window, int(now))
            used, expires_at = _LOCAL.get(
                key,
                (0, now + (65 if window == "minute" else 86405)),
            )
            decisions.append((window, limit, used, expires_at, key))
        if any(used + amount > limit for _, limit, used, _, _ in decisions):
            window, limit, used, _, _ = next(
                item for item in decisions if item[2] + amount > item[1]
            )
            return QuotaDecision(
                False,
                scope,
                window,
                limit,
                used,
                60 if window == "minute" else 3600,
            )
        for _, _, used, expires_at, key in decisions:
            _LOCAL[key] = (used + amount, expires_at)
    return QuotaDecision(
        True,
        scope,
        "minute",
        _limit(scope, "minute"),
        decisions[0][2] + amount,
        0,
    )


def check_tenant_quota(tenant: str, scope: str, amount: int = 1) -> QuotaDecision:
    """Consume ``amount`` quota units or return a deterministic denial."""
    scope = str(scope).strip().lower()
    if scope not in {"search", "chat", "ingest"}:
        raise ValueError(f"unknown quota scope: {scope}")
    amount = max(1, int(amount))
    if not enabled() or not str(tenant or "").strip():
        return QuotaDecision(True, scope, "disabled", 0, 0, 0)
    client = _redis()
    try:
        decision = _check_redis(tenant, scope, amount) if client else _check_local(
            tenant, scope, amount
        )
    except Exception:
        # Capacity protection must not turn a Redis outage into a full query
        # outage. Readiness and metrics make the weaker local fallback visible.
        decision = _check_local(tenant, scope, amount)
    if not decision.allowed:
        try:
            from src.monitoring.metrics import metrics_collector

            metrics_collector.record_quota_rejection(scope, decision.window)
        except Exception:
            pass
    return decision


def quota_health() -> Dict[str, Any]:
    if not enabled():
        return {"enabled": False, "backend": "disabled", "fallback": False}
    client = _redis()
    return {
        "enabled": True,
        "backend": "redis" if client else "memory",
        "fallback": client is None,
    }


__all__ = ["QuotaDecision", "check_tenant_quota", "enabled", "quota_health"]
