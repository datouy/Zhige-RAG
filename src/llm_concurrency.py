"""LLM 并发控制。

限制"同时处理的 LLM 请求数"，避免单实例被并发流量击穿
（每个请求都会触发一次分钟级的阻塞推理）。

设计约束（踩过的坑）::

    全局单例 + asyncio.Semaphore  =  跨事件循环必炸

``asyncio.Semaphore`` 在 Python 3.10+ 会**惰性绑定**到首次 ``acquire()``
时所在的事件循环。一旦绑定，在另一个循环里 ``async with`` 就会抛::

    RuntimeError: <asyncio.locks.Semaphore ...> is bound to a different event loop

触发条件是"信号量被耗尽、存在等待者"——也就是**并发超过上限**时。
本项目里这不是理论风险：WebSocket 路由是单进程全局 gate，而
- ``uvicorn --reload`` / 多 worker 会重建进程（gate 随之重建，安全）
- 测试里每个 ``TestClient`` 都新建事件循环，却共享进程级 gate
- 任何在服务主循环之外 ``asyncio.run()`` 的后台任务都可能先绑走 loop

所以这里改成**按事件循环分槽**（loop-local）：每个 loop 各自持有一把
信号量，跨 loop 时自动换一把。这样既保留了"限制并发"的效果，
又彻底消除跨循环绑定的 RuntimeError。
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, Dict


class LLMConcurrencyGate:
    """LLM 并发闸，限制同时处理的 LLM 请求数。

    用法::

        gate = LLMConcurrencyGate(max_concurrent=5)
        async with gate:
            result = await llm.chat(messages)
    """

    def __init__(self, max_concurrent: int = 5):
        """初始化并发闸。

        Args:
            max_concurrent: 同时处理的最大 LLM 请求数
        """
        self.max_concurrent = max_concurrent
        # 每个事件循环一把信号量。WeakKeyDictionary 不行 —— 事件循环对象
        # 不一定支持弱引用；用普通 dict + 显式清理（见 `_prune`）。
        self._semaphores: Dict[Any, asyncio.Semaphore] = {}
        # 当前持有许可的协程所对应的信号量（按 loop 记），
        # 保证 __aexit__ 在同一把上释放。
        self._held: Dict[Any, asyncio.Semaphore] = {}
        self._lock = threading.Lock()

    def _semaphore_for_current_loop(self) -> asyncio.Semaphore:
        """取当前事件循环对应的信号量；没有就新建一把。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 不在事件循环里（同步上下文）。返回一个永不阻塞的替身。
            return _NullSemaphore()

        with self._lock:
            sem = self._semaphores.get(loop)
            if sem is None:
                sem = asyncio.Semaphore(self.max_concurrent)
                self._semaphores[loop] = sem
            # 每次取用都清理一次已关闭 loop 的残留 —— 只在新建时清理的话，
            # 长期运行（每个 loop 用完就关）会一直堆积。
            self._prune(keep=loop)
            return sem

    def _prune(self, keep: Any) -> None:
        """清理已关闭的事件循环对应的信号量，防止长期运行后泄漏。

        事件循环 ``close()`` 之后 ``is_closed()`` 变 True。
        """
        stale = [
            loop for loop in self._semaphores
            if loop is not keep and getattr(loop, "is_closed", lambda: False)()
        ]
        for loop in stale:
            self._semaphores.pop(loop, None)

    async def __aenter__(self):
        """进入异步上下文管理器，获取信号量。"""
        sem = self._semaphore_for_current_loop()
        await sem.acquire()
        # 记住本次持有的是哪一把：__aexit__ 必须在**同一把**上释放，
        # 否则中途换了 loop 就会泄漏许可（后来的请求永远等不到闸）。
        self._held[asyncio.get_running_loop()] = sem
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出异步上下文管理器，释放信号量。"""
        loop = asyncio.get_running_loop()
        sem = self._held.pop(loop, None)
        if sem is None:
            sem = self._semaphore_for_current_loop()
        sem.release()
        return False

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """当前事件循环的底层信号量（向后兼容旧调用方）。

        注意：返回的是**当前循环**那把，不是全局单例 —— 这正是修复
        "bound to a different event loop" 的关键。
        """
        return self._semaphore_for_current_loop()

    async def acquire_async(self) -> None:
        """异步获取信号量（显式接口）。"""
        await self.__aenter__()

    async def release_async(self) -> None:
        """异步释放信号量（显式接口）。"""
        await self.__aexit__(None, None, None)


class _NullSemaphore:
    """无事件循环时的替身：永不阻塞，也不报错。"""

    async def acquire(self) -> None:
        return None

    def release(self) -> None:
        return None


def create_llm_gate() -> LLMConcurrencyGate:
    """创建 LLM 并发闸（工厂函数）。

    并发数从环境变量 LLM_MAX_CONCURRENT 读取，默认 5。
    """
    try:
        max_concurrent = max(1, int(os.getenv("LLM_MAX_CONCURRENT", "5")))
    except ValueError:
        max_concurrent = 5
    return LLMConcurrencyGate(max_concurrent=max_concurrent)
