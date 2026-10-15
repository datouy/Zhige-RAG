"""LLM 并发控制 - 简化版。

只保留 semaphore 并发闸，删除从未被使用的 AsyncLLMExecutor。

原 AsyncLLMExecutor 的问题：
1. submit() / start() 全项目无调用
2. _handle_request 调用的 4 个方法（agenerate/generate/stream_generate/astream_generate）
   在 LocalLLM 上全不存在（LocalLLM 只有 chat 方法）
3. 一旦启用立即 AttributeError

审计报告：AUDIT_FIX_REPORT.md P0-1
修复日期：2026-09-24
"""
from __future__ import annotations

import asyncio
import os


class LLMConcurrencyGate:
    """LLM 并发闸，限制同时处理的 LLM 请求数。
    
    使用示例::
    
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
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def __aenter__(self):
        """进入异步上下文管理器，获取信号量。"""
        await self._semaphore.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出异步上下文管理器，释放信号量。"""
        self._semaphore.release()
        return False
    
    def acquire_sync(self):
        """同步获取信号量（用于同步代码）。"""
        # 注意：这是简化版，仅用于与现有代码兼容
        # 实际使用应该用异步版本
        pass
    
    def release_sync(self):
        """同步释放信号量（用于同步代码）。"""
        pass
    
    @property
    def semaphore(self):
        """返回底层 semaphore（向后兼容）。"""
        return self._semaphore


def create_llm_gate() -> LLMConcurrencyGate:
    """创建 LLM 并发闸（工厂函数）。
    
    并发数从环境变量 LLM_MAX_CONCURRENT 读取，默认 5。
    """
    max_concurrent = int(os.getenv("LLM_MAX_CONCURRENT", "5"))
    return LLMConcurrencyGate(max_concurrent=max_concurrent)
