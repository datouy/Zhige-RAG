"""P1-3：可选异步 HTTP 后端（httpx）—— 回归测试。

原始诉求：远程 ollama/openai 后端走标准库 urllib 同步 I/O，异步问答接口只能
把整个 pipeline 丢进线程里桥接。新增 ``http_client.async_post_json`` /
``async_post_sse`` 与 ``OpenAICompatLLM.achat`` / ``astream_chat``，供异步
消费者在事件循环内做非阻塞请求（用 ``httpx.MockTransport`` 验证，不依赖真实服务）。
"""
from __future__ import annotations

import asyncio

import httpx

from src.http_client import async_post_json, async_post_sse
from src.llm_provider import OpenAICompatLLM


def _patch_client(monkeypatch, handler):
    """把 httpx.AsyncClient 替换为使用 MockTransport 的工厂。"""
    transport = httpx.MockTransport(handler)
    orig = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return orig(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


_URL = "http://localhost:11434/v1/chat/completions"


class TestAsyncPostJson:
    def test_returns_parsed_json(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": "你好"}}]})

        _patch_client(monkeypatch, handler)

        async def run():
            return await async_post_json(
                _URL, {"model": "m"}, headers={}, timeout=10, label="ollama", model="m"
            )

        data = asyncio.run(run())
        assert data["choices"][0]["message"]["content"] == "你好"

    def test_http_401_translates_to_hint(self, monkeypatch):
        def handler(request):
            return httpx.Response(401)

        _patch_client(monkeypatch, handler)

        async def run():
            try:
                await async_post_json(
                    _URL, {"model": "m"}, headers={}, timeout=10, label="openai", model="m"
                )
            except Exception as exc:
                return str(exc)
            return ""

        msg = asyncio.run(run())
        assert "鉴权失败" in msg


class TestAsyncPostSse:
    def test_streams_deltas_until_done(self, monkeypatch):
        body = (
            'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"好"}}]}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request):
            return httpx.Response(200, content=body.encode("utf-8"))

        _patch_client(monkeypatch, handler)

        async def run():
            out = []
            async for chunk in async_post_sse(
                _URL, {"model": "m"}, headers={}, timeout=10, label="ollama", model="m"
            ):
                out.append(chunk["choices"][0]["delta"]["content"])
            return out

        assert asyncio.run(run()) == ["你", "好"]


class TestOpenAICompatLLMAsync:
    def _llm(self):
        return OpenAICompatLLM(
            model_name="m",
            base_url="http://localhost:11434/v1",
            provider_label="ollama",
            http_backend="httpx",
        )

    def test_supports_async_flag(self):
        assert self._llm().supports_async is True
        sync_llm = OpenAICompatLLM(
            model_name="m", base_url="http://localhost:11434/v1", provider_label="ollama"
        )
        assert sync_llm.supports_async is False

    def test_achat(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": "答案"}}]})

        _patch_client(monkeypatch, handler)
        llm = self._llm()

        async def run():
            return await llm.achat([{"role": "user", "content": "问题"}])

        assert asyncio.run(run()) == "答案"

    def test_astream_chat(self, monkeypatch):
        body = (
            'data: {"choices":[{"delta":{"content":"分"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"片"}}]}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request):
            return httpx.Response(200, content=body.encode("utf-8"))

        _patch_client(monkeypatch, handler)
        llm = self._llm()

        async def run():
            return [tok async for tok in llm.achat([{"role": "user", "content": "问题"}], stream=True)]

        assert asyncio.run(run()) == ["分", "片"]
