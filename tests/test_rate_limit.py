"""测试限流修复（P0-2）。

验证：
1. 可信代理白名单正确识别
2. X-Forwarded-For 伪造防护
3. 按账号限流功能
"""
from src.rate_limit import (
    is_trusted_proxy,
    get_real_client_ip,
    get_rate_limit_key,
)


class MockRequest:
    """模拟 FastAPI Request。"""
    
    def __init__(self, client_host: str, headers: dict = None):
        self.client = type('obj', (object,), {'host': client_host})
        self.headers = headers or {}


def test_trusted_proxy_detection():
    """测试可信代理识别。"""
    # 内网地址应该是可信的
    assert is_trusted_proxy("10.0.0.1")
    assert is_trusted_proxy("172.16.0.1")
    assert is_trusted_proxy("192.168.1.1")
    assert is_trusted_proxy("127.0.0.1")
    
    # 公网地址不应该是可信的
    assert not is_trusted_proxy("8.8.8.8")
    assert not is_trusted_proxy("1.1.1.1")


def test_real_ip_from_trusted_proxy():
    """测试从可信代理获取真实 IP。"""
    # 场景：Nginx (10.0.0.1) 转发用户请求
    req = MockRequest(
        client_host="10.0.0.1",
        headers={"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}
    )
    assert get_real_client_ip(req) == "203.0.113.5"


def test_real_ip_from_untrusted_source():
    """测试防止 X-Forwarded-For 伪造。"""
    # 场景：恶意用户直连，伪造 X-Forwarded-For
    req = MockRequest(
        client_host="203.0.113.100",  # 公网 IP
        headers={"X-Forwarded-For": "127.0.0.1"}  # 尝试伪造本地 IP
    )
    # 应该忽略伪造的 header，使用真实连接 IP
    assert get_real_client_ip(req) == "203.0.113.100"


def test_rate_limit_key_by_user():
    """测试限流键：给了用户名就用 user:，否则用 ip:。"""
    req = MockRequest(client_host="203.0.113.5")

    # 显式传入用户名
    key = get_rate_limit_key(req, username="alice")
    assert key == "user:alice"

    # 未传用户名
    key = get_rate_limit_key(req, username=None)
    assert key == "ip:203.0.113.5"


def test_x_real_ip_fallback():
    """测试 X-Real-IP 回退。"""
    req = MockRequest(
        client_host="10.0.0.1",
        headers={"X-Real-IP": "203.0.113.10"}
    )
    assert get_real_client_ip(req) == "203.0.113.10"
