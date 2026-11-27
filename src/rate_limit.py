"""可信代理配置与请求限流。

P0-2 修复：反向代理下限流失效
- 问题1：所有用户共用一个反代IP桶
- 问题2：X-Forwarded-For 可伪造

修复方案：
- 可信代理白名单（只从可信代理读取 X-Forwarded-For）
- 以真实客户端 IP 作为限流键

本模块同时提供全局限流器 ``limiter``：``api/main.py`` 用它装配限流中间件，
路由模块用 ``@limiter.limit(...)`` 声明各自的配额。
"""
from __future__ import annotations

import hashlib
import os
from ipaddress import IPv4Network, IPv6Network, ip_address
from typing import Optional

from fastapi import Request
from slowapi import Limiter

# 可信代理网段（从环境变量或配置读取）
# 格式：逗号分隔的 CIDR，例如 "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
# 默认只信任本机回环地址。私网地址既可能是反向代理，也可能是普通内网
# 客户端；若默认信任整个私网，直连客户端可伪造 X-Forwarded-For 绕过限流。
# 部署在 Nginx / Ingress 后方时，必须显式填写代理实际所在的网段。
_TRUSTED_PROXIES_STR = os.getenv(
    "TRUSTED_PROXIES",
    "127.0.0.1/32,::1/128",
)

_TRUSTED_NETWORKS: list[IPv4Network | IPv6Network] = []


def _init_trusted_networks():
    """初始化可信代理网段列表。"""
    global _TRUSTED_NETWORKS
    if _TRUSTED_NETWORKS:
        return
    
    for cidr in _TRUSTED_PROXIES_STR.split(","):
        cidr = cidr.strip()
        if not cidr:
            continue
        try:
            if "/" not in cidr and ":" not in cidr:
                cidr = f"{cidr}/32"
            network = IPv6Network(cidr) if ":" in cidr else IPv4Network(cidr)
            _TRUSTED_NETWORKS.append(network)
        except Exception:
            # 忽略无效的 CIDR
            pass


def is_trusted_proxy(ip: str) -> bool:
    """判断 IP 是否属于可信代理。"""
    _init_trusted_networks()
    try:
        addr = ip_address(ip)
        return any(addr in network for network in _TRUSTED_NETWORKS)
    except Exception:
        return False


def get_real_client_ip(request: Request) -> str:
    """获取真实客户端 IP（防止伪造）。
    
    逻辑：
    1. 如果直连客户端是可信代理，从右向左剔除可信代理，取最后一个不可信地址
    2. 否则直接用连接的客户端 IP（不信任 X-Forwarded-For）
    
    Args:
        request: FastAPI Request 对象
    
    Returns:
        客户端真实 IP
    """
    # 获取直连客户端 IP
    if request.client:
        direct_ip = request.client.host
    else:
        return "unknown"
    
    # 如果不是可信代理，直接返回（不信任 X-Forwarded-For）
    if not is_trusted_proxy(direct_ip):
        return direct_ip
    
    # 是可信代理，从右向左剔除可信代理。不能直接取最左边：Nginx 默认使用
    # ``$proxy_add_x_forwarded_for``，客户端自带的伪造值会留在链首。
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        hops = [part.strip() for part in forwarded.split(",") if part.strip()]
        for hop in reversed(hops):
            if not is_trusted_proxy(hop):
                try:
                    return str(ip_address(hop))
                except ValueError:
                    # 非法地址不可作为可信的客户端身份，继续看链路右侧。
                    continue
        if hops:
            try:
                return str(ip_address(hops[0]))
            except ValueError:
                pass
    
    # 回退到 X-Real-IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        try:
            return str(ip_address(real_ip.strip()))
        except ValueError:
            pass
    
    return direct_ip


def get_rate_limit_key(request: Request, username: Optional[str] = None) -> str:
    """获取限流键（智能选择）。

    优先级：
    1. 显式传入用户名 -> "user:{username}"
    2. 团队模式且存在不透明会话令牌 -> "session:{sha256 前缀}"
    3. 否则 -> "ip:{real_ip}"

    团队模式用会话令牌分桶，避免多个员工经同一企业出口 NAT 时共用
    60/min 的 IP 配额。令牌只以短哈希形式作为内存/Redis 键，不写原值。

    Args:
        request: FastAPI Request 对象
        username: 用户名（可选）

    Returns:
        限流键
    """
    if username:
        return f"user:{username}"

    # 延迟导入，避免 rate_limit 在应用初始化阶段引入数据库依赖。
    try:
        from src.team_auth import request_token, team_mode

        if team_mode():
            token = request_token(request)
            if token:
                digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
                return f"session:{digest}"
    except Exception:
        # 令牌解析失败时安全回退到真实连接 IP。
        pass

    return f"ip:{get_real_client_ip(request)}"


#: 全局限流器实例。所有路由与 ``api/main.py`` 共用同一个，
#: 这样 ``app.state.limiter`` 与各端点的 ``@limiter.limit`` 才作用在同一份计数上。
_RATE_LIMIT_STORAGE_URL = (
    os.getenv("RATE_LIMIT_STORAGE_URL", "").strip()
    or os.getenv("REDIS_URL", "").strip()
    or "memory://"
)

#: 全局限流器实例。所有路由与 api/main.py 共用同一个，这样 app.state.limiter
#: 与各端点的 @limiter.limit 才作用在同一份计数上。
#: 显式传入 storage_uri：此前没传，RATE_LIMIT_STORAGE_URL 形同虚设，
#: 多副本下各自用独立内存桶，限流会失效。
limiter = Limiter(
    key_func=get_rate_limit_key,
    storage_uri=_RATE_LIMIT_STORAGE_URL,
    # Redis 短暂不可用时，单实例降级到本地桶继续提供核心成功路径；
    # 不能用 500 让所有查询直接失败。降级期间多 worker 只能各自限流，
    # 监控和运行手册必须明确标记这一容量与配额边界。
    in_memory_fallback=["60/minute"],
    in_memory_fallback_enabled=True,
    storage_options={
        "socket_connect_timeout": 0.5,
        "socket_timeout": 0.5,
        "retry_on_timeout": False,
    },
)


def rate_limit_health() -> dict[str, object]:
    """Return readiness-safe limiter state without exposing storage credentials."""
    backend = "memory" if _RATE_LIMIT_STORAGE_URL.startswith("memory://") else "redis"
    if backend == "memory":
        return {"status": "connected", "backend": "memory", "fallback": False}
    try:
        connected = bool(limiter._storage.check())
    except Exception:
        connected = False
    return {
        "status": "connected" if connected else "degraded",
        "backend": backend,
        "fallback": not connected and limiter._in_memory_fallback_enabled,
    }
