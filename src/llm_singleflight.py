"""LLM 预加载的单飞行锁（P3.3）。

避免两个进程同时构造 LLM 把显存打爆（每一份都会占用 VRAM）。
策略：``filelock.FileLock(LOCK_PATH)`` + 非阻塞 ``acquire``。第一个进程拿到锁后
跑构造；其它进程拿不到锁就直接 log 然后 ``return``，由调用方决定重试。

选用 (b) "log + skip" 模式，不共享内存，避免 IPC 复杂度。锁文件位置：
``<project>/logs/llm_load.lock``。
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from filelock import FileLock

logger = logging.getLogger(__name__)

# 默认锁路径：logs/llm_load.lock。可被环境变量覆盖。
DEFAULT_LOCK_PATH = Path(__file__).resolve().parent.parent / "logs" / "llm_load.lock"


def get_lock_path() -> Path:
    """获取锁路径，允许通过环境变量 ``CHINESERAGKB_LLM_LOCK`` 覆盖。"""
    custom = os.getenv("CHINESERAGKB_LLM_LOCK")
    if custom:
        return Path(custom)
    return DEFAULT_LOCK_PATH


def _ensure_lock_dir(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)


class LlmLoadSkipped(RuntimeError):
    """第二个进程的退出信号 — ``acquire_llm_load_lock`` 失败时抛出。"""


@contextmanager
def acquire_llm_load_lock(timeout: float = 0.0) -> Iterator[None]:
    """进入受保护的 LLM 加载块。

    Args:
        timeout: 等待锁的最长秒数。``0`` 表示 **不等待**（直接返回未拿到锁）。

    Yields:
        进入临界区时 ``yield None``；未拿到锁时抛 :class:`LlmLoadSkipped`。

    Example:
        >>> from src.llm_singleflight import acquire_llm_load_lock, LlmLoadSkipped
        >>> try:
        ...     with acquire_llm_load_lock():
        ...         do_llm_loading()
        ... except LlmLoadSkipped:
        ...     print("另一个进程正在加载，跳过")
    """
    lock_path = get_lock_path()
    _ensure_lock_dir(lock_path)
    lock = FileLock(str(lock_path))

    # ``filelock`` 的 ``acquire(timeout=...)`` 在超时时会抛 Timeout。
    # 我们用 try/except 把 timeout 包成 LlmLoadSkipped，让调用方用统一的 try 块。
    try:
        with lock.acquire(timeout=timeout):
            yield
    except TimeoutError as exc:
        # 当 timeout=0 时，filelock 可能直接抛 TimeoutError；也可能是其异常基类
        logger.info(
            "另一个进程正在加载 LLM（lock=%s），本进程 backoff 跳过", lock_path
        )
        raise LlmLoadSkipped(str(exc)) from exc


__all__ = [
    "acquire_llm_load_lock",
    "LlmLoadSkipped",
    "get_lock_path",
    "DEFAULT_LOCK_PATH",
]
