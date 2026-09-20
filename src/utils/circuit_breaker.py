"""熔断器模式实现，防止级联故障和外部服务过载。

熔断器三种状态：
- CLOSED（关闭）: 正常调用，失败计数
- OPEN（打开）: 快速失败，不调用目标服务
- HALF_OPEN（半开）: 允许有限请求测试服务是否恢复

适用于：
- LLM API 调用
- 外部 HTTP API
- 第三方服务集成
"""
from __future__ import annotations

import asyncio
import functools
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Generic

from src.utils import get_logger

logger = get_logger("circuit_breaker")


class CircuitState(str, Enum):
    """熔断器状态枚举。"""
    CLOSED = "closed"      # 正常，允许请求通过
    OPEN = "open"          # 熔断，快速失败
    HALF_OPEN = "half_open"  # 半开，允许测试请求


@dataclass
class CircuitStats:
    """熔断器统计信息。"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    @property
    def failure_rate(self) -> float:
        """计算失败率。"""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    @property
    def success_rate(self) -> float:
        """计算成功率。"""
        return 1.0 - self.failure_rate


@dataclass
class CircuitBreakerConfig:
    """熔断器配置。"""
    failure_threshold: int = 5          # 触发熔断的连续失败次数
    success_threshold: int = 3          # 从半开恢复到关闭需要的连续成功次数
    timeout: float = 60.0               # 熔断持续时间（秒）
    half_open_max_calls: int = 3        # 半开状态允许的最大并发调用数
    excluded_exceptions: tuple = ()      # 不计入失败的异常类型


class CircuitBreakerOpenError(Exception):
    """熔断器打开异常。"""
    def __init__(self, circuit_name: str, remaining_timeout: float):
        self.circuit_name = circuit_name
        self.remaining_timeout = remaining_timeout
        super().__init__(
            f"Circuit breaker '{circuit_name}' is OPEN. "
            f"Retry in {remaining_timeout:.1f} seconds."
        )


class CircuitBreaker:
    """熔断器实现。
    
    特性：
    - 线程安全
    - 支持异步和同步函数
    - 可配置失败阈值和恢复策略
    - 提供统计信息
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout: float = 60.0,
        half_open_max_calls: int = 3,
        excluded_exceptions: tuple = (),
    ):
        """初始化熔断器。
        
        Args:
            name: 熔断器名称
            failure_threshold: 触发熔断的连续失败次数
            success_threshold: 从半开恢复到关闭的连续成功次数
            timeout: 熔断持续时间（秒）
            half_open_max_calls: 半开状态允许的最大并发调用数
            excluded_exceptions: 不计入失败的异常类型
        """
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            half_open_max_calls=half_open_max_calls,
            excluded_exceptions=excluded_exceptions,
        )
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._lock = threading.RLock()
        self._half_open_semaphore: Optional[asyncio.Semaphore] = None
    
    @property
    def state(self) -> CircuitState:
        """获取当前熔断器状态。"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否应该转换到半开状态
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_semaphore = asyncio.Semaphore(
                        self.config.half_open_max_calls
                    )
                    logger.info(
                        "Circuit breaker '%s' transitioned from OPEN to HALF_OPEN",
                        self.name,
                    )
            return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """获取熔断器统计信息。"""
        with self._lock:
            return CircuitStats(
                total_calls=self._stats.total_calls,
                successful_calls=self._stats.successful_calls,
                failed_calls=self._stats.failed_calls,
                rejected_calls=self._stats.rejected_calls,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                consecutive_failures=self._stats.consecutive_failures,
                consecutive_successes=self._stats.consecutive_successes,
            )
    
    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试从 OPEN 状态恢复。"""
        if self._stats.last_failure_time is None:
            return True
        elapsed = time.time() - self._stats.last_failure_time
        return elapsed >= self.config.timeout
    
    def _record_success(self) -> None:
        """记录成功调用。"""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes += 1
            self._stats.last_success_time = time.time()
            
            # 从半开状态恢复
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._stats.consecutive_successes = 0
                    self._half_open_semaphore = None
                    logger.info(
                        "Circuit breaker '%s' transitioned from HALF_OPEN to CLOSED",
                        self.name,
                    )
    
    def _record_failure(self, exc: Exception) -> None:
        """记录失败调用。"""
        # 检查是否应该跳过（某些异常不计入失败）
        if isinstance(exc, self.config.excluded_exceptions):
            return
        
        with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.time()
            
            # 从关闭状态打开
            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "Circuit breaker '%s' transitioned from CLOSED to OPEN "
                        "(%d consecutive failures)",
                        self.name,
                        self._stats.consecutive_failures,
                    )
            # 从半开状态打开
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_semaphore = None
                logger.warning(
                    "Circuit breaker '%s' transitioned from HALF_OPEN to OPEN "
                    "(failure in half-open state)",
                    self.name,
                )
    
    def _can_execute(self) -> tuple[bool, Optional[float]]:
        """检查是否可以执行请求。
        
        Returns:
            (can_execute, remaining_timeout)
        """
        state = self.state
        
        if state == CircuitState.CLOSED:
            return True, None
        
        if state == CircuitState.OPEN:
            remaining = self.config.timeout - (
                time.time() - self._stats.last_failure_time
            )
            return False, max(0, remaining)
        
        # HALF_OPEN
        return True, None
    
    def record_rejected(self) -> None:
        """记录被拒绝的调用（熔断打开时）。"""
        with self._lock:
            self._stats.rejected_calls += 1
    
    def reset(self) -> None:
        """重置熔断器到关闭状态。"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._stats = CircuitStats()
            self._half_open_semaphore = None
            logger.info("Circuit breaker '%s' has been reset", self.name)
    
    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """同步调用函数（带熔断保护）。
        
        Args:
            func: 要调用的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数返回值
            
        Raises:
            CircuitBreakerOpenError: 熔断器打开时抛出
            Exception: 函数执行时的异常
        """
        can_execute, remaining_timeout = self._can_execute()
        
        if not can_execute:
            self.record_rejected()
            raise CircuitBreakerOpenError(self.name, remaining_timeout)
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    async def call_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """异步调用函数（带熔断保护）。
        
        Args:
            func: 要调用的异步函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数返回值
            
        Raises:
            CircuitBreakerOpenError: 熔断器打开时抛出
            Exception: 函数执行时的异常
        """
        can_execute, remaining_timeout = self._can_execute()
        
        if not can_execute:
            self.record_rejected()
            raise CircuitBreakerOpenError(self.name, remaining_timeout)
        
        # 半开状态限制并发
        if self._state == CircuitState.HALF_OPEN and self._half_open_semaphore:
            async with self._half_open_semaphore:
                return await self._execute_async(func, *args, **kwargs)
        
        return await self._execute_async(func, *args, **kwargs)
    
    async def _execute_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """执行异步函数并记录结果。"""
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # 如果是同步函数，在线程池中执行
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: func(*args, **kwargs),
                )
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise


# 类型变量用于泛型装饰器
T = TypeVar("T")


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    success_threshold: int = 3,
    timeout: float = 60.0,
    excluded_exceptions: tuple = (),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """熔断器装饰器。
    
    用法：
        @circuit_breaker(name="llm_call", failure_threshold=3)
        async def call_llm(prompt: str) -> str:
            ...
    
    Args:
        name: 熔断器名称，默认使用函数名
        failure_threshold: 触发熔断的连续失败次数
        success_threshold: 从半开恢复到关闭的连续成功次数
        timeout: 熔断持续时间（秒）
        excluded_exceptions: 不计入失败的异常类型
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cb_name = name or func.__name__
        _breaker: Optional[CircuitBreaker] = None
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal _breaker
            if _breaker is None:
                _breaker = CircuitBreaker(
                    name=cb_name,
                    failure_threshold=failure_threshold,
                    success_threshold=success_threshold,
                    timeout=timeout,
                    excluded_exceptions=excluded_exceptions,
                )
            return _breaker.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal _breaker
            if _breaker is None:
                _breaker = CircuitBreaker(
                    name=cb_name,
                    failure_threshold=failure_threshold,
                    success_threshold=success_threshold,
                    timeout=timeout,
                    excluded_exceptions=excluded_exceptions,
                )
            return await _breaker.call_async(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# 全局熔断器注册表
_cb_registry: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    **config: Any,
) -> CircuitBreaker:
    """获取或创建命名的熔断器。
    
    Args:
        name: 熔断器名称
        **config: 熔断器配置参数
        
    Returns:
        熔断器实例
    """
    if name not in _cb_registry:
        _cb_registry[name] = CircuitBreaker(name=name, **config)
    return _cb_registry[name]


def get_all_circuit_breakers() -> dict[str, CircuitStats]:
    """获取所有熔断器的状态统计。"""
    return {
        name: cb.stats
        for name, cb in _cb_registry.items()
    }


def reset_all_circuit_breakers() -> None:
    """重置所有熔断器。"""
    for cb in _cb_registry.values():
        cb.reset()
