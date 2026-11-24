"""跨进程共享缓存（可选 Redis）与缓存值序列化。

默认部署（单 worker）走进程内 ``OrderedDict`` + 锁，零新增依赖、零配置。
当设置 ``CACHE_STORAGE_URL``（或回退到 ``REDIS_URL``）且安装了 ``redis`` 包时，
检索缓存与答案缓存会**额外**写入 Redis，供多 worker 共享命中：

- 多副本 / 多 worker 部署下，一个进程算好的答案能被其它进程复用，降低重复
  检索与 LLM 调用；
- 任一进程缓存未命中时仍回退到本进程内存缓存，Redis 不可用（未装包 / 网络
  抖动 / 服务宕机）时自动降级为"只有进程内缓存"，不阻塞主链路。

Redis 只存本服务自己写入的 JSON 值，不 ``pickle``，避免反序列化不可信数据。
"""
from __future__ import annotations

import json
import os
from typing import Any, List, Optional

# 本模块只依赖 vector_store.Hit 做序列化；RAGResult 在反序列化时惰性导入，
# 避免与 rag_pipeline 形成模块级循环导入。
from .vector_store import Hit


# ----------------------------------------------------------------------
#  序列化
# ----------------------------------------------------------------------
def serialize_hits(hits: List[Hit]) -> str:
    """把命中列表编码为 JSON 字符串（跨 worker 共享检索缓存用）。"""
    return json.dumps(
        [h.to_dict() for h in hits],
        ensure_ascii=False,
        default=str,
        sort_keys=True,
    )


def deserialize_hits(raw: str) -> List[Hit]:
    """把 :func:`serialize_hits` 的输出还原为命中列表。"""
    data = json.loads(raw)
    hits: List[Hit] = []
    for d in data:
        hits.append(
            Hit(
                id=d.get("id", ""),
                text=d.get("text", ""),
                score=float(d.get("score", 0.0) or 0.0),
                metadata=dict(d.get("metadata") or {}),
            )
        )
    return hits


def serialize_result(res: Any) -> str:
    """把 :class:`RAGResult` 编码为 JSON 字符串（答案缓存共享用）。"""
    return json.dumps(
        {
            "answer": res.answer,
            "sources": res.sources,
            "raw_hits": [h.to_dict() for h in (res.raw_hits or [])],
            "timings": res.timings,
            "verification": res.verification,
        },
        ensure_ascii=False,
        default=str,
        sort_keys=True,
    )


def deserialize_result(raw: str) -> Any:
    """把 :func:`serialize_result` 的输出还原为 :class:`RAGResult`。"""
    from .rag_pipeline import RAGResult

    d = json.loads(raw)
    return RAGResult(
        answer=d.get("answer", ""),
        sources=list(d.get("sources") or []),
        raw_hits=[
            Hit(
                id=h.get("id", ""),
                text=h.get("text", ""),
                score=float(h.get("score", 0.0) or 0.0),
                metadata=dict(h.get("metadata") or {}),
            )
            for h in (d.get("raw_hits") or [])
        ],
        timings=dict(d.get("timings") or {}),
        verification=d.get("verification"),
    )


# ----------------------------------------------------------------------
#  共享缓存（Redis）
# ----------------------------------------------------------------------
class SharedCache:
    """Redis 共享缓存。所有操作都吞掉异常并降级（返回 None / 不写入）。

    Args:
        url: Redis 连接串（``redis://...`` / ``rediss://...``）。
        namespace: 顶层键名前缀，用于区分不同部署 / 避免误删。
        client: 可注入的 redis 客户端（测试用）；不传则按 ``url`` 惰性创建。
    """

    def __init__(self, url: str, namespace: str = "zhige:rag", client: Any = None) -> None:
        self._url = url
        self._namespace = namespace.rstrip(":")
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(self._url, decode_responses=False)
        return self._client

    def _full_key(self, scope: str, key: str) -> str:
        return f"{self._namespace}:{scope}:{key}"

    def get(self, scope: str, key: str) -> Optional[bytes]:
        try:
            return self.client.get(self._full_key(scope, key))
        except Exception:
            return None

    def set(self, scope: str, key: str, value: bytes, ttl: float) -> None:
        try:
            self.client.set(self._full_key(scope, key), value, ex=max(1, int(ttl)))
        except Exception:
            pass

    def clear_scope(self, scope: str) -> None:
        """删除某个 scope（一个知识库）下的全部缓存键。"""
        try:
            pattern = f"{self._namespace}:{scope}:*"
            keys = list(self.client.scan_iter(match=pattern, count=1000))
            if keys:
                self.client.delete(*keys)
        except Exception:
            pass


def build_shared_cache() -> Optional[SharedCache]:
    """按环境变量构造共享缓存；不可用时返回 ``None``（纯进程内缓存）。

    读取顺序与 ``rate_limit.py`` 一致：优先 ``CACHE_STORAGE_URL``，回退
    ``REDIS_URL``；两者都空、或 ``redis`` 包未安装时返回 ``None``。
    """
    url = (
        os.getenv("CACHE_STORAGE_URL", "").strip()
        or os.getenv("REDIS_URL", "").strip()
    )
    if not url:
        return None
    try:
        import redis  # noqa: F401
    except ImportError:
        return None
    return SharedCache(url)


__all__ = [
    "SharedCache",
    "build_shared_cache",
    "serialize_hits",
    "deserialize_hits",
    "serialize_result",
    "deserialize_result",
]
