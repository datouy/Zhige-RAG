"""P1-3 收尾：异步流式问答端到端接入 httpx 非阻塞 LLM —— 回归测试。

``astream_answer`` 现在在检索/上下文组装（同步）之后，若 LLM 后端支持异步
HTTP（``supports_async``），就把生成切回事件循环用 ``_emit_llm_tail_async``
流式输出；否则仍在线程内同步生成（旧行为）。
"""
from __future__ import annotations

import asyncio

from src.rag_pipeline import RAGPipeline


class _AsyncLLM:
    supports_async = True
    gen_cfg = None

    def achat(self, messages, generation=None, stream=False, timeout=None):
        # 与 OpenAICompatLLM.achat 一致：stream=True 返回异步生成器
        async def _tokens():
            for t in ["你", "好"]:
                yield t

        if stream:
            return _tokens()

        async def _once():
            return "你好"

        return _once()


def _pipeline():
    p = object.__new__(RAGPipeline)
    p._verify = lambda answer, ctx: None
    p._visible_citations = lambda answer, cites, ver: cites
    p._answer_cache_put = lambda key, res: None
    return p


class TestEmitLlmTailAsync:
    def test_streams_tokens_then_done(self):
        p = _pipeline()
        p.llm = _AsyncLLM()

        async def run():
            return [e async for e in p._emit_llm_tail_async([], [], [], [], None)]

        events = asyncio.run(run())
        tokens = [e["data"] for e in events if e["event"] == "token"]
        assert tokens == ["你", "好"]
        assert events[-1]["event"] == "done"


class TestAstremAnswerDispatch:
    def _pipeline_with(self, supports_async):
        p = object.__new__(RAGPipeline)
        p.llm = type("L", (), {"supports_async": supports_async})()
        p.calls = {"async": 0, "sync": 0}

        def before_llm(question, top_k=None, history=None, user_id=None, allowed_groups=None, direct=False):
            yield {"event": "hits", "data": [{"index": 1}]}
            yield {"event": "__llm_ready__", "data": {
                "messages": [], "hits": [], "cites": [], "context_chunks": [], "ans_key": None,
            }}

        def tail_sync(messages, hits, cites, context_chunks, ans_key):
            p.calls["sync"] += 1
            yield {"event": "token", "data": "S"}
            yield {"event": "done", "data": [], "verification": None}

        async def tail_async(messages, hits, cites, context_chunks, ans_key):
            p.calls["async"] += 1
            yield {"event": "token", "data": "A"}
            yield {"event": "done", "data": [], "verification": None}

        p._stream_answer_before_llm = before_llm
        p._emit_llm_tail = tail_sync
        p._emit_llm_tail_async = tail_async
        return p

    def test_async_backend_uses_async_tail(self):
        p = self._pipeline_with(supports_async=True)

        async def run():
            return [e async for e in p.astream_answer("q")]

        events = asyncio.run(run())
        assert p.calls == {"async": 1, "sync": 0}
        tokens = [e["data"] for e in events if e["event"] == "token"]
        assert tokens == ["A"]
        assert events[-1]["event"] == "done"

    def test_sync_backend_still_uses_sync_tail(self):
        p = self._pipeline_with(supports_async=False)

        async def run():
            return [e async for e in p.astream_answer("q")]

        events = asyncio.run(run())
        assert p.calls == {"async": 0, "sync": 1}
        tokens = [e["data"] for e in events if e["event"] == "token"]
        assert tokens == ["S"]
        assert events[-1]["event"] == "done"
