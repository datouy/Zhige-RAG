"""共享 HTTP 客户端工具 —— 供 LLM / Embedding 的 OpenAI 兼容后端复用。

为什么要单独抽出来
------------------
LLM 与 Embedding 的远程后端都需要同一套东西：

1. ``POST`` JSON 并解析响应；
2. 解析 SSE 流式响应（LLM 需要，Embedding 不需要但应预留）；
3. **本地/内网地址自动绕过系统代理** —— 这是真实踩过的坑：国内开发机常配置
   全局 ``HTTP_PROXY``，urllib 默认会把 ``localhost`` 请求也发给代理，导致
   Ollama 明明在运行却永远连不上（表现为 502 Bad Gateway）；
4. 把 HTTP 错误翻译成**可操作的排查提示**（401/404/429/超时各有不同建议）。

放在一处实现，避免两个 provider 各写一套后逐渐漂移。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Generator, Optional
from urllib.parse import urlparse


class RemoteServiceError(RuntimeError):
    """远程服务调用失败。消息中会附带可操作的排查提示。"""


# ----------------------------------------------------------------------
#  代理
# ----------------------------------------------------------------------
def is_local_endpoint(url: str) -> bool:
    """判断 URL 是否指向本机 / 内网地址。

    本地与私网地址必须显式清空代理，否则配了全局 ``HTTP_PROXY`` 的机器
    永远连不上本地服务。
    """
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return True
    if host.startswith("192.168.") or host.startswith("10."):
        return True
    if host.startswith("172."):  # 私网段 172.16.0.0/12
        try:
            return 16 <= int(host.split(".")[1]) <= 31
        except (IndexError, ValueError):
            return False
    return host.endswith(".local") or host.endswith(".internal")


_NO_PROXY_OPENER: Optional[urllib.request.OpenerDirector] = None
_DEFAULT_OPENER: Optional[urllib.request.OpenerDirector] = None


def opener_for(url: str) -> urllib.request.OpenerDirector:
    """按目标地址返回 HTTP opener（本地地址禁用代理）。opener 可复用。"""
    global _NO_PROXY_OPENER, _DEFAULT_OPENER
    if is_local_endpoint(url):
        if _NO_PROXY_OPENER is None:
            _NO_PROXY_OPENER = urllib.request.build_opener(
                urllib.request.ProxyHandler({})
            )
        return _NO_PROXY_OPENER
    if _DEFAULT_OPENER is None:
        _DEFAULT_OPENER = urllib.request.build_opener()
    return _DEFAULT_OPENER


# ----------------------------------------------------------------------
#  请求构造
# ----------------------------------------------------------------------
def build_headers(api_key: str = "", extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update(extra)
    return headers


def make_request(
    url: str, payload: Dict[str, Any], headers: Dict[str, str]
) -> urllib.request.Request:
    return urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )


# ----------------------------------------------------------------------
#  错误翻译
# ----------------------------------------------------------------------
_HTTP_HINTS = {
    401: "鉴权失败：请检查 api_key 是否正确或已过期。",
    403: "无权限：该 Key 可能未开通此模型。",
    404: "路径不存在：base_url 是否正确？（需包含 /v1 这类前缀）",
    422: "请求参数被拒绝：请检查模型名是否受支持。",
    429: "触发限流或余额不足：请稍后重试或检查配额。",
}


def describe_http_error(
    exc: urllib.error.HTTPError,
    *,
    label: str,
    url: str,
    model: str = "",
) -> str:
    detail = ""
    try:
        detail = exc.read().decode("utf-8", errors="ignore")[:300]
    except Exception:  # noqa: BLE001 - 读错误体失败不应掩盖原始错误
        pass

    hint = _HTTP_HINTS.get(exc.code)
    if hint is None:
        if exc.code in (502, 503, 504) and is_local_endpoint(url):
            hint = (
                "本地服务不可达。若本机配置了 HTTP_PROXY/HTTPS_PROXY，"
                "请将本地地址加入 NO_PROXY，并确认服务已启动。"
            )
        else:
            hint = "请检查 base_url、模型名与账号状态。"

    return (
        f"{label} 返回 HTTP {exc.code} {exc.reason}。{hint} "
        f"[url={url}, model={model}] 响应：{detail}"
    )


def describe_url_error(
    exc: urllib.error.URLError,
    *,
    label: str,
    base_url: str,
    model: str = "",
) -> str:
    reason = getattr(exc, "reason", exc)
    if label == "ollama":
        return (
            f"无法连接 Ollama（{base_url}）：{reason}。"
            f"请确认 ① Ollama 已启动（`ollama serve`）；"
            f"② 已拉取模型（`ollama pull {model}`）；"
            f"③ 若在容器内运行，base_url 不能写 localhost，应指向宿主机。"
        )
    return (
        f"无法连接服务（{base_url}）：{reason}。"
        f"请检查网络、base_url 拼写与代理设置。"
    )


# ----------------------------------------------------------------------
#  请求执行
# ----------------------------------------------------------------------
def post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Dict[str, str],
    timeout: float,
    label: str,
    model: str = "",
) -> Dict[str, Any]:
    """POST 并返回解析后的 JSON 对象。"""
    request = make_request(url, payload, headers)
    try:
        with opener_for(url).open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        raise RemoteServiceError(
            describe_http_error(exc, label=label, url=url, model=model)
        ) from exc
    except urllib.error.URLError as exc:
        raise RemoteServiceError(
            describe_url_error(exc, label=label, base_url=url, model=model)
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RemoteServiceError(f"{label} 返回了非 JSON 内容：{raw[:200]}") from exc
    if not isinstance(data, dict):
        raise RemoteServiceError(f"{label} 返回体不是对象：{raw[:200]}")
    return data


def post_sse(
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Dict[str, str],
    timeout: float,
    label: str,
    model: str = "",
) -> Generator[Dict[str, Any], None, None]:
    """POST 并按 SSE 逐条产出解析后的 JSON 分片。

    会忽略注释行（``: keep-alive``）与空行，遇 ``data: [DONE]`` 结束。
    """
    request = make_request(url, payload, headers)
    try:
        response = opener_for(url).open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RemoteServiceError(
            describe_http_error(exc, label=label, url=url, model=model)
        ) from exc
    except urllib.error.URLError as exc:
        raise RemoteServiceError(
            describe_url_error(exc, label=label, base_url=url, model=model)
        ) from exc

    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                return
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


__all__ = [
    "RemoteServiceError",
    "build_headers",
    "describe_http_error",
    "describe_url_error",
    "is_local_endpoint",
    "make_request",
    "opener_for",
    "post_json",
    "post_sse",
]
