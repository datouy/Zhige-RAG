"""网页抓取与正文提取（URL 导入）。

用途
----
用户常希望"把这个网页喂给知识库"而不想手工复制粘贴。本模块负责抓取并把
HTML 转成可入库的纯文本。

安全（重要）
------------
URL 导入是典型的 **SSRF 高危入口** —— 攻击者让服务器去请求它本不该访问的地址，
从而探测内网、读取云元数据（如 ``169.254.169.254``）、甚至攻击内网服务。
这里的防护是**多层**的：

1. **协议白名单**：仅 http / https；
2. **解析后校验 IP**：不允许私网 / 回环 / 链路本地 / 保留地址。
   注意必须校验**解析后的 IP**而不是域名本身 —— 否则
   ``evil.com`` 解析到 ``127.0.0.1``（DNS rebinding）即可绕过；
3. **逐跳校验重定向**：每次 3xx 跳转都重新做第 2 步检查，
   防止"公网地址 302 到内网";
4. **限制响应体大小**：流式读取并在超限时中断，避免被大文件拖垮；
5. **限制 Content-Type**：只接受文本类，拒绝二进制；
6. **超时**。

这些检查在生产环境关闭会有实际风险，因此默认全部开启。
"""
from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

from .utils import clean_text, get_logger

logger = get_logger("web_loader")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; ZhigeBot/1.0; +https://github.com/datouy/rag) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
ALLOWED_SCHEMES = ("http", "https")
ALLOWED_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")

# 正文里丢弃的标签
_DROP_TAGS = (
    "script", "style", "nav", "footer", "header", "aside",
    "noscript", "form", "iframe", "svg", "button", "select",
)


class WebFetchError(ValueError):
    """URL 抓取失败（消息面向用户，可直接回传）。"""


# ----------------------------------------------------------------------
#  SSRF 防护
# ----------------------------------------------------------------------
def is_public_ip(ip: str) -> bool:
    """判断 IP 是否可公开访问（非私网/回环/链路本地/保留/组播）。"""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def validate_url(url: str, allow_private: bool = False) -> urllib.parse.SplitResult:
    """校验 URL 是否允许抓取，返回解析结果。

    Args:
        url: 待校验的 URL。
        allow_private: 是否允许内网地址（**仅供内网部署/测试使用**，
            默认 False，公网服务切勿打开）。

    Raises:
        WebFetchError: 协议不允许、缺少主机、或解析到内网地址。
    """
    url = (url or "").strip()
    if not url:
        raise WebFetchError("URL 不能为空")
    if "://" not in url:
        url = "https://" + url  # 容忍省略协议

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise WebFetchError(f"仅支持 http/https 协议，收到：{parsed.scheme or '(空)'}")
    host = parsed.hostname
    if not host:
        raise WebFetchError("URL 缺少主机名")

    if allow_private:
        return parsed

    # 主机本身就是 IP 时直接判定；否则解析域名
    try:
        ipaddress.ip_address(host)
        candidates = [host]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
        except socket.gaierror as exc:
            raise WebFetchError(f"域名解析失败：{host}（{exc}）") from exc
        candidates = sorted({info[4][0] for info in infos})

    blocked = [ip for ip in candidates if not is_public_ip(ip)]
    if blocked:
        raise WebFetchError(
            f"出于安全考虑，不允许抓取内网/保留地址（{host} → {', '.join(blocked)}）"
        )
    return parsed


class _SsrfSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """重定向时逐跳重新校验，避免"公网 302 到内网"绕过。"""

    def __init__(self, allow_private: bool = False) -> None:
        super().__init__()
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        try:
            validate_url(newurl, allow_private=self.allow_private)
        except WebFetchError as exc:
            raise urllib.error.URLError(f"重定向目标被拒绝：{exc}") from exc
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# ----------------------------------------------------------------------
#  抓取
# ----------------------------------------------------------------------
def fetch_html(
    url: str,
    timeout: float = 15.0,
    max_bytes: int = 5 * 1024 * 1024,
    allow_private: bool = False,
) -> Tuple[str, str]:
    """抓取网页并返回 ``(最终URL, HTML文本)``。

    Raises:
        WebFetchError: 校验失败或抓取失败。
    """
    validate_url(url, allow_private=allow_private)
    if "://" not in url:
        url = "https://" + url

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),  # 抓外网也走直连：本地服务的代理常不可用
        _SsrfSafeRedirectHandler(allow_private=allow_private),
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )

    try:
        with opener.open(request, timeout=timeout) as resp:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if not any(ct in content_type for ct in ALLOWED_CONTENT_TYPES):
                raise WebFetchError(
                    f"不支持的内容类型：{content_type or '(未知)'}（仅支持网页文本）"
                )

            # 先看 Content-Length，再流式读并兜底，双重限制
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise WebFetchError(
                    f"页面过大（{int(declared) // 1024} KB），超过上限 {max_bytes // 1024} KB"
                )

            chunks: List[bytes] = []
            total = 0
            while True:
                block = resp.read(65536)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise WebFetchError(f"页面超过大小上限（{max_bytes // 1024} KB）")
                chunks.append(block)
            raw = b"".join(chunks)
            final_url = resp.geturl() or url
    except urllib.error.HTTPError as exc:
        raise WebFetchError(f"目标站点返回 HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise WebFetchError(f"无法访问该地址：{getattr(exc, 'reason', exc)}") from exc
    except TimeoutError as exc:
        raise WebFetchError("抓取超时，请稍后重试或换一个地址") from exc

    # 编码探测：优先响应头，再 html meta，最后 utf-8 兜底
    charset = None
    m = re.search(r"charset=([\w\-]+)", content_type)
    if m:
        charset = m.group(1)
    if not charset:
        m = re.search(rb'charset=["\']?([\w\-]+)', raw[:4096], re.IGNORECASE)
        if m:
            charset = m.group(1).decode("ascii", "ignore")
    for enc in filter(None, [charset, "utf-8", "gb18030"]):
        try:
            return final_url, raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return final_url, raw.decode("utf-8", errors="ignore")


# ----------------------------------------------------------------------
#  正文提取
# ----------------------------------------------------------------------
def _make_soup(html: str):
    from bs4 import BeautifulSoup

    try:
        return BeautifulSoup(html, "lxml")
    except Exception:  # noqa: BLE001 - lxml 未安装时退回标准库解析器
        return BeautifulSoup(html, "html.parser")


def extract_main_text(html: str) -> Tuple[str, str]:
    """从 HTML 提取 ``(标题, 正文文本)``。

    优先 ``<article>`` / ``<main>`` 等正文容器，尽量剥离导航、页脚、脚本等噪声；
    找不到容器时退回 ``<body>``。
    """
    soup = _make_soup(html)
    for tag in soup(_DROP_TAGS):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)

    node = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )
    text = node.get_text("\n", strip=True)
    # 压缩空行（网页正文常带大量空白）
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, clean_text(text)


def fetch_and_extract(
    url: str,
    timeout: float = 15.0,
    max_bytes: int = 5 * 1024 * 1024,
    allow_private: bool = False,
    max_chars: int = 200_000,
) -> Dict[str, str]:
    """抓取 URL 并提取正文，返回可直接入库的字段。

    Returns:
        ``{"url", "final_url", "title", "text", "chars"}``
    """
    final_url, html = fetch_html(
        url, timeout=timeout, max_bytes=max_bytes, allow_private=allow_private
    )
    title, text = extract_main_text(html)
    if len(text) > max_chars:
        logger.warning("网页正文过长（%d 字符），已截断到 %d", len(text), max_chars)
        text = text[:max_chars]
    if not text.strip():
        raise WebFetchError("未能从该页面提取到正文（可能是纯前端渲染或空页面）")

    host = urllib.parse.urlsplit(final_url).hostname or "web"
    return {
        "url": url,
        "final_url": final_url,
        "title": title or host,
        "text": text,
        "chars": str(len(text)),
    }


__all__ = [
    "WebFetchError",
    "extract_main_text",
    "fetch_and_extract",
    "fetch_html",
    "is_public_ip",
    "validate_url",
]
