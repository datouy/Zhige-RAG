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

import os
from ipaddress import IPv4Address, IPv4Network, ip_address
from typing import Optional

from fastapi import Request
from slowapi import Limiter

# 可信代理网段（从环境变量或配置读取）
# 格式：逗号分隔的 CIDR，例如 "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
_TRUSTED_PROXIES_STR = os.getenv(
    "TRUSTED_PROXIES",
    "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,127.0.0.1/32"
)

_TRUSTED_NETWORKS: list[IPv4Network] = []


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
            _TRUSTED_NETWORKS.append(IPv4Network(cidr))
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
    1. 如果直连客户端是可信代理，从 X-Forwarded-For 取第一个IP
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
    
    # 是可信代理，尝试从 X-Forwarded-For 获取原始 IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # X-Forwarded-For 格式：client, proxy1, proxy2
        # 取第一个（最左边的是原始客户端）
        return forwarded.split(",")[0].strip()
    
    # 回退到 X-Real-IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    return direct_ip


def get_rate_limit_key(request: Request, username: Optional[str] = None) -> str:
    """获取限流键（智能选择）。

    优先级：
    1. 显式传入用户名 -> "user:{username}"
    2. 否则 -> "ip:{real_ip}"

    Args:
        request: FastAPI Request 对象
        username: 用户名（可选）

    Returns:
        限流键
    """
    if username:
        return f"user:{username}"

    return f"ip:{get_real_client_ip(request)}"


#: 全局限流器实例。所有路由与 ``api/main.py`` 共用同一个，
#: 这样 ``app.state.limiter`` 与各端点的 ``@limiter.limit`` 才作用在同一份计数上。
limiter = Limiter(key_func=get_rate_limit_key)
