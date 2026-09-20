"""异步 LLM 执行器 + 请求队列。

支持：
- 令牌桶限流（per-user + 全局）
- 优先级队列（pro > free）
- 多 worker 并发处理
- Streaming token 流式回调
"""
from __future__ import annotations
import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
from threading import Lock


class TaskPriority(Enum):
    LOW = 0      # free 用户
    NORMAL = 1   # pro 用户
    HIGH = 2     # enterprise 用户


@dataclass
class LLMRequest:
    """LLM 请求封装。"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    prompt: str = ""
    priority: TaskPriority = TaskPriority.LOW
    created_at: float = field(default_factory=time.time)
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    stream: bool = False
    on_token: Optional[Callable[[str], None]] = None
    future: Optional[asyncio.Future] = field(default=None)


class TokenBucket:
    """令牌桶限流器。"""

    def __init__(self, rate: float, capacity: float):
        self.rate = rate          # 每秒补充的 token 数
        self.capacity = capacity  # 桶容量
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = Lock()

    def consume(self, tokens: int = 1) -> bool:
        """尝试消耗 tokens。返回是否成功。"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def available(self) -> float:
        """查看当前可用 token 数。"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            return min(self.capacity, self.tokens + elapsed * self.rate)


class AsyncLLMExecutor:
    """异步 LLM 执行器。

    Features:
    - Per-user rate limiting（令牌桶）
    - Priority queue（优先级队列）
    - Configurable concurrency（可配置并发数）
    - Token streaming support（流式回调）
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        free_rate: float = 2.0,        # tokens/sec
        pro_rate: float = 20.0,
        enterprise_rate: float = 100.0,
        queue_capacity: int = 1000,
    ):
        self.max_concurrent = max_concurrent
        self.queue: List[LLMRequest] = []
        self.queue_capacity = queue_capacity
        self.lock = Lock()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.running = False

        self.user_buckets: Dict[str, TokenBucket] = {}
        self.rate_map = {
            "free": free_rate,
            "pro": pro_rate,
            "enterprise": enterprise_rate,
        }
        self.user_tiers: dict[str, str] = {}
        self._task: Optional[asyncio.Task] = None
        self._llm_factory: Optional[Callable] = None
        # 追踪的 user bucket 上限：防止 user_buckets 随用户数无限增长（内存泄漏）
        self.max_tracked_users = int(os.getenv("LLM_MAX_TRACKED_USERS", "1000"))

    def _bucket_for(self, user_id: str) -> TokenBucket:
        """获取用户的限流桶，惰性创建；超过上限时淘汰最早插入的桶。"""
        bucket = self.user_buckets.get(user_id)
        if bucket is None:
            tier = self.user_tiers.get(user_id, "free")
            rate = self.rate_map.get(tier, self.rate_map["free"])
            if len(self.user_buckets) >= self.max_tracked_users:
                # dict 保持插入序，弹出到上限以下即可（粗粒度 LRU）
                for old_id in list(self.user_buckets.keys()):
                    self.user_buckets.pop(old_id, None)
                    if len(self.user_buckets) < self.max_tracked_users:
                        break
            bucket = self.user_buckets[user_id] = TokenBucket(rate, rate)
        return bucket

    def _insert_by_priority(self, request: LLMRequest) -> None:
        """按优先级插入队列（高优先级在前）。

        P5 审计修复：限流被拒的请求之前固定 insert(0) 回队首——一个 free
        用户耗尽配额时会把后面所有 enterprise 请求堵在身后（队头阻塞）。
        """
        with self.lock:
            for i, r in enumerate(self.queue):
                if r.priority.value < request.priority.value:
                    self.queue.insert(i, request)
                    return
            self.queue.append(request)

    def set_user_tier(self, user_id: str, tier: str) -> None:
        """设置用户订阅层，同时更新限流速率。"""
        rate = self.rate_map.get(tier, self.rate_map["free"])
        self.user_tiers[user_id] = tier
        self.user_buckets[user_id] = TokenBucket(rate, rate)

    def submit(self, request: LLMRequest) -> asyncio.Future:
        """提交 LLM 请求。返回 Future。"""
        if len(self.queue) >= self.queue_capacity:
            raise RuntimeError(f"请求队列已满（{self.queue_capacity}），请稍后重试")

        future = asyncio.Future()
        request.future = future

        # 根据用户订阅层设置优先级
        tier = self.user_tiers.get(request.user_id, "free")
        priority_map = {
            "free": TaskPriority.LOW,
            "pro": TaskPriority.NORMAL,
            "enterprise": TaskPriority.HIGH,
        }
        request.priority = priority_map.get(tier, TaskPriority.LOW)

        with self.lock:
            # 按优先级插入队列（高优先级插前面）
            inserted = False
            for i, r in enumerate(self.queue):
                if r.priority.value < request.priority.value:
                    self.queue.insert(i, request)
                    inserted = True
                    break
            if not inserted:
                self.queue.append(request)

        return future

    def get_queue_position(self, request_id: str) -> int:
        """查询请求在队列中的位置（用于进度展示）。"""
        with self.lock:
            for i, r in enumerate(self.queue):
                if r.request_id == request_id:
                    return i
            return -1

    async def start(self, llm_factory: Callable[[], Any]) -> None:
        """启动执行器主循环。

        Args:
            llm_factory: 返回 LLM 实例的工厂函数
        """
        self._llm_factory = llm_factory
        self.running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """停止执行器。"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        """主事件循环：不断从队列取请求执行。"""
        while self.running:
            request = None

            with self.lock:
                if self.queue:
                    request = self.queue.pop(0)

            if request:
                bucket = self._bucket_for(request.user_id)
                if bucket and not bucket.consume():
                    # 限流：按优先级放回队列原位，避免队头阻塞高优先级请求
                    self._insert_by_priority(request)
                    await asyncio.sleep(0.1)
                    continue

                asyncio.create_task(self._handle_request(request))
            else:
                await asyncio.sleep(0.05)

    async def _handle_request(self, request: LLMRequest) -> None:
        """处理单个 LLM 请求。"""
        async with self.semaphore:
            if self._llm_factory is None:
                request.future.set_exception(RuntimeError("LLM factory not set"))
                return

            try:
                llm = self._llm_factory()
                if request.stream:
                    tokens: List[str] = []
                    if hasattr(llm, "astream_generate"):
                        async for tok in llm.astream_generate(
                            request.prompt,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                            top_p=request.top_p,
                        ):
                            tokens.append(tok)
                            if request.on_token:
                                request.on_token(tok)
                    else:
                        result = await self._sync_stream_fallback(llm, request)
                        request.future.set_result(result)
                        return
                    request.future.set_result("".join(tokens))
                else:
                    if hasattr(llm, "agenerate"):
                        result = await llm.agenerate(
                            request.prompt,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                        )
                    else:
                        result = llm.generate(
                            request.prompt,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                        )
                    request.future.set_result(result)
            except Exception as e:
                request.future.set_exception(e)

    async def _sync_stream_fallback(self, llm: Any, request: LLMRequest) -> str:
        """同步流式生成的异步包装（当 LLM 无 astream_generate 时）。"""
        loop = asyncio.get_event_loop()
        tokens: List[str] = []
        for tok in llm.stream_generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            tokens.append(tok)
            if request.on_token:
                await loop.run_in_executor(None, lambda t=tok: request.on_token(t))
        return "".join(tokens)
