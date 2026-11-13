"""LLM 并发闸：所有模型调用共享同一份容量、等待队列与统计。

设计目标
--------
1. 全局限量：HTTP、WebSocket、Agent、RAG 与后台压缩最终都会调用
   LocalLLM.chat 或 OpenAICompatLLM.chat。闸放在模型实现内，
   避免路由层各自加锁后模型实际承受成倍并发。
2. 异步安全：异步许可属于具体 task（ContextVar 保存租约），不再把
   事件循环当作持有者，避免同循环并发互相覆盖释放状态。
3. 有界排队：等待队列有上限与超时，避免请求无限堆积拖垮线程池。
4. 可观测：提供 active/waiting/rejected/timeouts 统计。

本地 HuggingFace 推理线程无法被强制杀死；超时后调用方停止等待，
LocalLLM 通过 StoppingCriteria 让正在进行的 generate 尽快停止。
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterator, Optional


class LLMConcurrencyLimitError(RuntimeError):
    """等待队列已满，调用应快速失败而不是继续排队。"""


class LLMConcurrencyTimeoutError(TimeoutError):
    """排队或执行等待超时。"""


@dataclass
class _Waiter:
    loop: Any
    future: asyncio.Future
    event: Optional[threading.Event] = None
    cancelled: bool = False


class _Lease:
    def __init__(self, gate: "LLMConcurrencyGate") -> None:
        self.gate = gate
        self.released = False
        self.context_token: Any = None

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.gate._release()


class LLMConcurrencyGate:
    """线程安全与事件循环安全的全局并发闸。

    异步路径使用 per-loop Future，同步路径使用 threading.Event。
    两条路径共用同一份容量、等待队列和统计。
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        *,
        max_queue: int = 64,
        queue_timeout: float = 120.0,
    ) -> None:
        self.max_concurrent = max(1, int(max_concurrent))
        self.max_queue = max(0, int(max_queue))
        self.queue_timeout = max(0.1, float(queue_timeout))

        self._lock = threading.Lock()
        self._active = 0
        self._waiters: Deque[_Waiter] = deque()
        self._stats = {"rejected": 0, "timeouts": 0, "releases": 0}
        self._lease_var: ContextVar[Optional[_Lease]] = ContextVar(
            "llm_concurrency_lease", default=None
        )

    @property
    def semaphore(self) -> "LLMConcurrencyGate":
        """兼容旧调用：返回自身，不再暴露事件循环信号量。"""
        return self

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "max_concurrent": self.max_concurrent,
                "max_queue": self.max_queue,
                "active": self._active,
                "waiting": len(self._waiters),
                **self._stats,
            }

    @contextmanager
    def blocking(self, timeout: Optional[float] = None) -> Iterator[None]:
        """同步模型调用的全局闸。"""
        wait_timeout = self.queue_timeout if timeout is None else float(timeout)
        waiter = self._enqueue_or_acquire(event=threading.Event())
        if waiter is not None:
            if not waiter.event.wait(wait_timeout):
                self._cancel_waiter(waiter)
                with self._lock:
                    self._stats["timeouts"] += 1
                raise LLMConcurrencyTimeoutError(
                    f"LLM concurrency queue timeout after {wait_timeout}s"
                )
        try:
            yield
        finally:
            self._release()

    async def __aenter__(self) -> _Lease:
        return await self.acquire_async()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        lease = self._lease_var.get()
        if lease is not None:
            self._lease_var.reset(lease.context_token)
            lease.release()
        return False

    async def acquire_async(self, timeout: Optional[float] = None) -> _Lease:
        wait_timeout = self.queue_timeout if timeout is None else float(timeout)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError("acquire_async must be called in an event loop") from exc

        waiter = self._enqueue_or_acquire(future=loop.create_future())
        if waiter is None:
            return self._new_lease()
        try:
            await asyncio.wait_for(asyncio.shield(waiter.future), timeout=wait_timeout)
        except asyncio.TimeoutError as exc:
            self._cancel_waiter(waiter)
            with self._lock:
                self._stats["timeouts"] += 1
            raise LLMConcurrencyTimeoutError(
                f"LLM concurrency queue timeout after {wait_timeout}s"
            ) from exc
        except asyncio.CancelledError:
            self._cancel_waiter(waiter)
            raise
        return self._new_lease()

    async def release_async(self) -> None:
        lease = self._lease_var.get()
        if lease is None:
            raise RuntimeError("release_async called without acquire_async")
        self._lease_var.reset(lease.context_token)
        lease.release()

    def _new_lease(self) -> _Lease:
        lease = _Lease(self)
        lease.context_token = self._lease_var.set(lease)
        return lease

    def _enqueue_or_acquire(
        self,
        *,
        future: Optional[asyncio.Future] = None,
        event: Optional[threading.Event] = None,
    ) -> Optional[_Waiter]:
        with self._lock:
            if self._active < self.max_concurrent:
                self._active += 1
                return None
            if len(self._waiters) >= self.max_queue:
                self._stats["rejected"] += 1
                raise LLMConcurrencyLimitError(
                    "LLM concurrency queue is full "
                    f"(active={self._active}, waiting={len(self._waiters)})"
                )
            waiter = _Waiter(
                loop=asyncio.get_running_loop() if future is not None else None,
                future=future,
                event=event,
            )
            self._waiters.append(waiter)
            return waiter

    def _release(self) -> None:
        with self._lock:
            self._stats["releases"] += 1
            while self._waiters:
                waiter = self._waiters.popleft()
                if waiter.cancelled:
                    continue
                if waiter.event is not None:
                    waiter.event.set()
                    return
                if waiter.loop.is_closed():
                    continue
                try:
                    waiter.loop.call_soon_threadsafe(_complete_future, waiter.future)
                    return
                except RuntimeError:
                    continue
            self._active = max(0, self._active - 1)

    def _cancel_waiter(self, waiter: _Waiter) -> None:
        with self._lock:
            if waiter in self._waiters:
                waiter.cancelled = True
                self._waiters.remove(waiter)


def _complete_future(future: asyncio.Future) -> None:
    if not future.done():
        future.set_result(None)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def create_llm_gate() -> LLMConcurrencyGate:
    """创建独立闸，供测试与兼容旧导入。"""
    return LLMConcurrencyGate(
        max_concurrent=_env_int("LLM_MAX_CONCURRENT", 5),
        max_queue=_env_int("LLM_MAX_QUEUE", 64),
        queue_timeout=_env_float("LLM_QUEUE_TIMEOUT", 120.0),
    )


_GLOBAL_GATE: Optional[LLMConcurrencyGate] = None
_GLOBAL_GATE_LOCK = threading.Lock()


def get_llm_call_gate() -> LLMConcurrencyGate:
    """模型实现共享的全局闸。"""
    global _GLOBAL_GATE
    if _GLOBAL_GATE is None:
        with _GLOBAL_GATE_LOCK:
            if _GLOBAL_GATE is None:
                _GLOBAL_GATE = create_llm_gate()
    return _GLOBAL_GATE


def reset_llm_call_gate() -> None:
    global _GLOBAL_GATE
    with _GLOBAL_GATE_LOCK:
        _GLOBAL_GATE = None
