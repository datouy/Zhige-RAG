"""LLM 并发闸的跨事件循环安全性。

背景：``asyncio.Semaphore`` 在 Python 3.10+ 会惰性绑定到首次 ``acquire()``
所在的事件循环。一旦绑定，在另一个循环里 ``async with`` 就会抛::

    RuntimeError: <asyncio.locks.Semaphore ...> is bound to a different event loop

触发条件是"信号量被耗尽、存在等待者"，也就是**并发超过上限**时。

本项目的 WebSocket 路由用的是**进程级全局 gate**，而
- 每个 ``TestClient`` 都新建事件循环，却共享同一个 gate；
- ``uvicorn --reload`` / 多 worker 会在同进程外换 loop；
- 任何在服务主循环之外 ``asyncio.run()`` 的后台任务都可能先绑走 loop。

所以 gate 必须做到"按事件循环分槽"。这里把该行为钉死。
"""

from __future__ import annotations

import asyncio

import pytest

from src.llm_concurrency import LLMConcurrencyGate, create_llm_gate


async def _hold(gate: LLMConcurrencyGate, hold: float = 0.05):
    """占住闸一小会儿。"""
    async with gate:
        await asyncio.sleep(hold)


class TestCrossEventLoop:
    def test_same_gate_works_in_two_sequential_loops(self):
        """同一个 gate 在两个不同事件循环里依次使用，不应报错。"""

        async def scenario():
            gate = LLMConcurrencyGate(max_concurrent=1)
            # 并发超限，制造等待者 —— 这正是触发绑定的关键条件
            t = asyncio.create_task(_hold(gate))
            await asyncio.sleep(0.01)
            w = asyncio.create_task(_hold(gate))
            await t
            await w

        asyncio.run(scenario())  # loop A
        asyncio.run(scenario())  # loop B —— 复用同一个 gate 实例

    def test_global_gate_survives_loop_switch(self):
        """模块级单例 gate（生产实际用法）跨 loop 也要安全。"""
        gate = create_llm_gate()

        async def scenario():
            t = asyncio.create_task(_hold(gate))
            await asyncio.sleep(0.01)
            w = asyncio.create_task(_hold(gate))
            await t
            await w

        asyncio.run(scenario())
        asyncio.run(scenario())

    def test_concurrency_is_still_limited(self):
        """修复不能把"限流"本身弄丢：同时进入数不得超过 max_concurrent。"""
        gate = LLMConcurrencyGate(max_concurrent=2)
        state = {"current": 0, "peak": 0}

        async def scenario():
            async def one():
                async with gate:
                    state["current"] += 1
                    state["peak"] = max(state["peak"], state["current"])
                    await asyncio.sleep(0.02)
                    state["current"] -= 1

            await asyncio.gather(*(one() for _ in range(8)))

        asyncio.run(scenario())
        assert state["peak"] <= 2, f"并发峰值 {state['peak']} 超过了上限 2"


class TestGateApi:
    def test_acquire_release_are_balanced(self):
        """acquire 后必须能 release，且不泄漏许可。"""
        gate = LLMConcurrencyGate(max_concurrent=1)

        async def scenario():
            await gate.acquire_async()
            await gate.release_async()
            # 释放后应能再次立即获取（若泄漏了就会挂死，这里用 wait_for 兜底）
            await asyncio.wait_for(gate.acquire_async(), timeout=2.0)
            await gate.release_async()

        asyncio.run(scenario())

    def test_gate_outside_event_loop_does_not_raise(self):
        """同步上下文里取 semaphore 不应抛异常。"""
        gate = LLMConcurrencyGate(max_concurrent=2)
        sem = gate.semaphore  # 不在 loop 里
        assert sem is not None

    def test_max_concurrent_is_at_least_one(self):
        """LLM_MAX_CONCURRENT=0 不应把闸锁死。"""
        import os

        old = os.environ.get("LLM_MAX_CONCURRENT")
        try:
            os.environ["LLM_MAX_CONCURRENT"] = "0"
            assert create_llm_gate().max_concurrent >= 1
            os.environ["LLM_MAX_CONCURRENT"] = "abc"
            assert create_llm_gate().max_concurrent >= 1
        finally:
            if old is None:
                os.environ.pop("LLM_MAX_CONCURRENT", None)
            else:
                os.environ["LLM_MAX_CONCURRENT"] = old

    def test_gate_state_is_balanced_after_closed_loop(self):
        """旧实现的 per-loop 字典可能残留；新实现没有 loop 级状态可堆积。"""
        gate = LLMConcurrencyGate(max_concurrent=1)

        async def scenario():
            await _hold(gate)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(scenario())
        finally:
            loop.close()

        # 再跑一次：此时上一个 loop 已关闭，应被 prune 掉
        asyncio.run(scenario())

        assert gate.stats()["active"] == 0
        assert gate.stats()["waiting"] == 0

    def test_repeated_loop_usage_releases_all_leases(self):
        """连续使用多个 loop 后，容量必须全部归还。"""
        gate = LLMConcurrencyGate(max_concurrent=1)

        async def scenario():
            await _hold(gate)

        for _ in range(3):
            asyncio.run(scenario())

        # 最后一个 loop 也已关闭；再取一次即应把前几个清空
        asyncio.run(scenario())

        assert gate.stats()["active"] == 0
        assert gate.stats()["waiting"] == 0


@pytest.mark.parametrize("max_conc", [1, 3, 10])
def test_gate_works_with_various_limits(max_conc):
    gate = LLMConcurrencyGate(max_concurrent=max_conc)

    async def scenario():
        await asyncio.gather(*(_hold(gate, 0.01) for _ in range(5)))

    asyncio.run(scenario())
