"""API 共享依赖：全局单例 + 缓存配置加载器。

设计目标
--------
- 所有跨路由共享的对象（Pipeline / Embedding / VectorStore / LLM Executor /
  Agent 句柄 / KG 组件 / 拆分器工厂）集中在此，避免在 ``api/main.py`` 或各
  路由文件中重复实例化。
- 配置加载走 ``lru_cache``，避免每次请求都重新 ``yaml.safe_load`` 同一文件
  （P1.2 — backend-architect 优化）。
- 提供 ``apply_runtime_config(cfg)`` 工具，把当前活跃的 dict 一次性注入到
  各组件，避免下游重复 ``load_config``（P1.4 — ai-engineer 优化）。
- 提供 ``single_flight_lock``，用于启动期预热等需要"全进程只跑一次"的场景
  （P3.3 — sre 优化）。该锁使用 stdlib（``msvcrt`` on Windows, ``fcntl``
  on POSIX），不引入新依赖。

所有路由模块按以下约定使用本文件::

    from api.deps import get_pipeline, get_embedding, get_cached_config

严禁在新代码里再次 ``RAGPipeline.from_config(...)`` 重建 Pipeline ——
应该复用 ``get_pipeline()`` / ``get_runtime_pipeline(user_id)``。
"""
from __future__ import annotations

import functools
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterator, Optional

# 确保项目根目录在 path 中（与 api/main.py 一致）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embeddings import EmbeddingModel
from src.kg import GraphRAG, GraphRetriever, KGExtractor, create_kg_store
from src.llm import LocalLLM
from src.rag_pipeline import RAGPipeline
from src.reranker import BgeReranker
from src.text_splitter import ChineseTextSplitter, RecursiveTextSplitter
from src.utils import apply_env_overrides, load_config, merge_dict
from src.vector_store import ChromaStore, hybrid_kwargs


# =====================================================================
#  配置缓存（P1.2）
# =====================================================================
@functools.lru_cache(maxsize=4)
def get_cached_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """读取并缓存配置 dict。

    说明：
    - 同一进程内，``load_config`` 只会真正解析 YAML 一次。
    - 通过 ``apply_env_overrides`` 把环境变量叠加（不变性，所以缓存安全）。
    - 再叠加**界面写入的模型设置**（``config/local_overrides.yaml``，优先级最高）——
      这样用户在界面上切换后端后无需改 config.yaml，也不用重启进程。
    - 若需强制重新加载（如刚保存了模型设置），调用 ``_reset_cached_config()``。
    """
    from src.settings_service import load_overrides
    from src.utils import merge_dict

    cfg = load_config(path)
    # 界面写入的模型设置覆盖 config.yaml
    overrides = load_overrides()
    if overrides:
        cfg = merge_dict(cfg, overrides)
    # 环境变量最后叠加 —— 容器 / CI 场景下部署者的意图优先，也符合 12-factor。
    # 因此若同一项同时被环境变量与界面设置指定，以环境变量为准；
    # src.settings_service 会在保存时检测并提醒用户。
    return apply_env_overrides(cfg)


def _reset_cached_config() -> None:
    """清空配置缓存。仅用于测试或运维热更新。"""
    get_cached_config.cache_clear()


def get_runtime_config() -> Dict[str, Any]:
    """获取项目根目录下的默认配置。``get_cached_config`` 的便捷包装。"""
    return get_cached_config(str(ROOT / "config" / "config.yaml"))


# =====================================================================
#  LLM 执行器（P1.3）
# =====================================================================
_llm_executor: Optional[Any] = None
_llm_executor_lock = Lock()


def get_llm_executor():
    """获取或创建全局 LLM 执行器（异步 + 限流）。"""
    global _llm_executor
    if _llm_executor is None:
        with _llm_executor_lock:
            if _llm_executor is None:
                from src.executor.llm_executor import AsyncLLMExecutor

                _llm_executor = AsyncLLMExecutor(
                    max_concurrent=int(os.getenv("LLM_MAX_CONCURRENT", "3")),
                    free_rate=float(os.getenv("LLM_FREE_RATE", "2.0")),
                    pro_rate=float(os.getenv("LLM_PRO_RATE", "20.0")),
                    enterprise_rate=float(os.getenv("LLM_ENTERPRISE_RATE", "100.0")),
                )
    return _llm_executor


def reset_llm_executor() -> None:
    """清空执行器引用（测试 / 关闭时使用）。"""
    global _llm_executor
    _llm_executor = None


# =====================================================================
#  共享组件单例（P1.4）
# =====================================================================
_pipeline: Optional[RAGPipeline] = None
_embedding: Optional[EmbeddingModel] = None
_vector_store: Optional[ChromaStore] = None
_singletons_lock = Lock()


def get_embedding() -> Any:
    """获取或创建全局 Embedding 实例。

    后端由 ``embedding.backend`` 决定（``local`` / ``ollama`` / ``openai``），
    实现见 :mod:`src.embeddings_provider`。返回值只需满足
    ``encode(texts, batch_size=...) -> ndarray`` 协议即可被 ``ChromaStore`` 使用，
    因此不再强绑 :class:`EmbeddingModel`。
    """
    global _embedding
    if _embedding is None:
        with _singletons_lock:
            if _embedding is None:
                from src.embeddings_provider import create_embedding

                cfg = get_runtime_config()
                _embedding = create_embedding(cfg.get("embedding", {}))
    return _embedding


def get_vector_store() -> ChromaStore:
    """获取或创建全局 VectorStore 实例。"""
    global _vector_store
    if _vector_store is None:
        # 锁外先建 embedding（get_embedding 有自己的锁；threading.Lock
        # 不可重入，锁内调用会同线程二次加锁死锁）
        embed = get_embedding()
        cfg = get_runtime_config()
        from src.embeddings_provider import embedding_collection_name

        # 非 local 后端会自动带上后端与模型指纹：换 Embedding 模型会改变向量
        # 维度，复用旧 collection 会直接报错，也会让检索结果不可信。
        collection = embedding_collection_name(
            cfg["vector_store"].get("collection_name", "chinese_rag_kb"),
            cfg.get("embedding", {}),
        )
        with _singletons_lock:
            if _vector_store is None:
                _vector_store = ChromaStore(
                    persist_directory=cfg["vector_store"]["persist_directory"],
                    collection_name=collection,
                    embedding_model=embed,
                    distance_fn=cfg["vector_store"].get("distance_fn", "cosine"),
            **hybrid_kwargs(cfg.get("vector_store", {})),
                )
    return _vector_store


def get_pipeline() -> RAGPipeline:
    """获取或创建全局 Pipeline 实例（懒加载，单例）。"""
    global _pipeline
    if _pipeline is None:
        # 底层组件各自带锁，全部在 _singletons_lock 之外构造，
        # 避免不可重入锁死锁
        from src.utils import resolve_path
        from src.prompt_template import PromptTemplate

        embedding = get_embedding()
        vs = get_vector_store()
        cfg = get_runtime_config()

        # 默认 lazy_llm=True（避免冷启动慢），首次 answer 时按需加载
        llm: Optional[LocalLLM] = None
        if not bool(os.getenv("RAG_LAZY_LLM", "1")):
            # P1.4：与 RAGPipeline.from_config / ensure_llm 共用同一构造逻辑
            llm = RAGPipeline._build_llm_from_cfg(cfg.get("llm", {}))

        reranker: Optional[BgeReranker] = None
        if cfg.get("reranker", {}).get("enabled"):
            reranker = BgeReranker(
                model_name=cfg["reranker"]["model_name"],
                device=cfg["reranker"].get("device", "auto"),
                cache_dir=cfg["reranker"].get(
                    "cache_dir", cfg["embedding"].get("cache_dir")
                ),
            )

        prompts = PromptTemplate.from_yaml(
            str(resolve_path(cfg.get("rag", {}).get("system_prompt_file", "config/prompts.yaml")))
        )

        with _singletons_lock:
            if _pipeline is None:
                _pipeline = RAGPipeline(
                    config=cfg,
                    embedding=embedding,
                    vector_store=vs,
                    llm=llm,
                    reranker=reranker,
                    prompts=prompts,
                )
    return _pipeline


def get_runtime_pipeline(
    user_id: Optional[str] = None,
    kb: Optional[Dict[str, str]] = None,
) -> RAGPipeline:
    """为指定 user（可指定知识库）返回 ``RAGPipeline``。

    - ``user_id`` 为 None：直接返回全局单例。
    - ``user_id`` 不为 None：复制全局单例的 config / llm / reranker / prompts，
      但 ``vector_store`` / ``kg_store`` 替换为该租户的隔离实例。这样既复用
      LLM（避免重复加载），又保证检索 / 图谱按租户隔离（P1.4 关键）。
    - ``kb`` 不为 None：``vector_store`` 进一步切换到该**知识库**的 collection
      （多知识库隔离，见 :mod:`src.kb_service`）。参数用
      ``{"id", "collection_name"}`` 形式的普通 dict，避免与 ORM 会话耦合。
    """
    import copy

    base = get_pipeline()
    if not user_id:
        return base

    from src.factories import TenantAwareFactory

    cfg = dict(base.config)
    if kb and kb.get("collection_name"):
        vs = TenantAwareFactory.get_chroma_store_by_name(kb["collection_name"], cfg)
    else:
        vs = TenantAwareFactory.get_chroma_store(user_id, cfg)
    kg = TenantAwareFactory.get_kg_store(user_id, cfg)

    # 浅拷贝 pipeline，确保不污染全局
    tenant_pipeline = copy.copy(base)
    tenant_pipeline.vector_store = vs
    # 注入租户专属 KG store。注意：copy.copy 是浅拷贝，base 上已经探测过的
    # _graph_state_checked / _graph_retriever 会被一并带过来；若不重置，
    # _get_graph_retriever 会直接返回全局图谱的检索器（跨租户数据泄露）。
    tenant_pipeline.kg_store = kg
    tenant_pipeline._graph_state_checked = False
    tenant_pipeline._graph_retriever = None
    return tenant_pipeline


# =====================================================================
#  拆分器工厂
# =====================================================================
def get_splitter(cfg: Optional[Dict[str, Any]] = None):
    """根据配置构建分块器。``cfg`` 缺省时取全局缓存配置。"""
    if cfg is None:
        cfg = get_runtime_config()
    sp = cfg.get("text_splitter", {})
    strategy = sp.get("strategy", "chinese")
    if strategy == "chinese":
        return ChineseTextSplitter(
            chunk_size=sp.get("chunk_size", 256),
            chunk_overlap=sp.get("chunk_overlap", 32),
            separators=sp.get("chinese_separators") or sp.get("separators"),
            keep_separator=sp.get("keep_separator", True),
            min_chunk_size=sp.get("min_chunk_size", 32),
        )
    return RecursiveTextSplitter(
        chunk_size=sp.get("chunk_size", 512),
        chunk_overlap=sp.get("chunk_overlap", 64),
    )


# =====================================================================
#  KG 共享组件
# =====================================================================
_kg_store = None
_kg_retriever: Optional[GraphRetriever] = None
_kg_extractor: Optional[KGExtractor] = None
_graph_rag: Optional[GraphRAG] = None
_kg_lock = Lock()


def _load_kg_config() -> Dict[str, Any]:
    return get_runtime_config().get("knowledge_graph", {}) or {}


def get_kg_store():
    """获取（或创建）共享 KG 存储实例（admin 全局视角）。"""
    global _kg_store
    if _kg_store is None:
        with _kg_lock:
            if _kg_store is None:
                kg_cfg = _load_kg_config()
                if not kg_cfg.get("enabled", True):
                    kg_cfg = {**kg_cfg, "enabled": True}
                _kg_store = create_kg_store(kg_cfg)
    return _kg_store


def get_kg_retriever():
    """获取（或创建）图谱检索器（admin 全局）。"""
    global _kg_retriever
    if _kg_retriever is None:
        with _kg_lock:
            if _kg_retriever is None:
                _kg_retriever = GraphRetriever(get_kg_store())
    return _kg_retriever


def get_kg_extractor() -> Optional[KGExtractor]:
    """获取（或创建）KG 抽取器（需要 LLM）。LLM 不可用时返回 None。"""
    global _kg_extractor
    if _kg_extractor is None:
        with _kg_lock:
            if _kg_extractor is None:
                try:
                    pipeline = get_pipeline()
                    pipeline.ensure_llm()
                    llm = pipeline.llm
                except Exception:
                    return None
                if llm is None:
                    return None
                cfg = _load_kg_config()
                ex_cfg = cfg.get("extractor", {}) or {}
                _kg_extractor = KGExtractor(
                    llm=llm,
                    max_entities_per_chunk=int(ex_cfg.get("max_entities_per_chunk", 20)),
                    max_relations_per_chunk=int(ex_cfg.get("max_relations_per_chunk", 30)),
                )
    return _kg_extractor


def get_graph_rag() -> GraphRAG:
    """获取（或创建）GraphRAG 实例。"""
    global _graph_rag
    if _graph_rag is None:
        with _kg_lock:
            if _graph_rag is None:
                try:
                    vs = get_vector_store()
                except Exception:
                    vs = None
                kg_store = get_kg_store()
                retriever = get_kg_retriever()
                extractor = get_kg_extractor()
                try:
                    pipeline = get_pipeline()
                    pipeline.ensure_llm()
                    llm = pipeline.llm
                except Exception:
                    llm = None
                _graph_rag = GraphRAG(
                    vector_store=vs,
                    kg_store=kg_store,
                    extractor=extractor,
                    retriever=retriever,
                    llm=llm,
                )
    return _graph_rag


# =====================================================================
#  健康检查缓存（P2.2 — sre）
# =====================================================================
_health_cache: Dict[str, Any] = {"ts": 0.0, "payload": None}
_health_cache_lock = Lock()
_HEALTH_TTL_SECONDS = float(os.getenv("HEALTH_CACHE_TTL", "5.0"))


def get_health_cache_ttl() -> float:
    """健康检查缓存 TTL（秒），可通过 HEALTH_CACHE_TTL 调整。"""
    return _HEALTH_TTL_SECONDS


def _set_health_cache(payload: Dict[str, Any]) -> None:
    _health_cache["ts"] = time.time()
    _health_cache["payload"] = payload


def _get_health_cache() -> Optional[Dict[str, Any]]:
    if _health_cache["payload"] is None:
        return None
    if (time.time() - _health_cache["ts"]) > _HEALTH_TTL_SECONDS:
        return None
    return _health_cache["payload"]


def invalidate_health_cache() -> None:
    """主动失效健康检查缓存（测试 / 运维）。"""
    with _health_cache_lock:
        _health_cache["ts"] = 0.0
        _health_cache["payload"] = None


# =====================================================================
#  Single-flight 文件锁（P3.3 — sre）
# =====================================================================
@contextmanager
def single_flight_lock(name: str, timeout: float = 30.0) -> Iterator[bool]:
    """跨进程的 single-flight 互斥锁（基于文件）。

    实现要点：
    - 跨平台 stdlib：Windows 用 ``msvcrt.locking``，POSIX 用 ``fcntl.flock``。
    - 不引入 ``filelock`` 第三方依赖（避免污染 requirements.txt）。
    - 失败 / 超时返回 ``False``；成功获取返回 ``True``。

    用法::

        with single_flight_lock("preload_llm") as acquired:
            if acquired:
                do_expensive_work()
            else:
                # 别的进程正在做，正常返回
                pass
    """
    lock_path = Path(os.getenv("SINGLE_FLIGHT_DIR", str(ROOT / ".locks"))) / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not lock_path.exists():
        lock_path.touch()

    fd = None
    acquired = False
    try:
        fd = open(lock_path, "r+")
        if os.name == "nt":
            import msvcrt

            # msvcrt.locking 阻塞；用 LK_NBLCK + 重试模拟非阻塞
            LK_NBLCK = 2
            deadline = time.time() + timeout
            while True:
                try:
                    msvcrt.locking(fd.fileno(), LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError:
                    if time.time() >= deadline:
                        break
                    time.sleep(0.1)
        else:
            import fcntl

            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                # 等待释放：尝试加共享锁直到 timeout
                deadline = time.time() + timeout
                while time.time() < deadline:
                    time.sleep(0.1)
                    try:
                        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except BlockingIOError:
                        continue
        yield acquired
    finally:
        if fd is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    LK_UNLCK = 0
                    try:
                        msvcrt.locking(fd.fileno(), LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl

                    try:
                        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
            finally:
                fd.close()


# =====================================================================
#  测试 / 重启钩子
# =====================================================================
def reset_all_singletons() -> None:
    """清空所有进程级单例。供测试与运维热重启使用。"""
    global _pipeline, _embedding, _vector_store, _kg_store
    global _kg_retriever, _kg_extractor, _graph_rag
    with _singletons_lock, _kg_lock:
        _pipeline = None
        _embedding = None
        _vector_store = None
        _kg_store = None
        _kg_retriever = None
        _kg_extractor = None
        _graph_rag = None
    reset_llm_executor()
    _reset_cached_config()
    invalidate_health_cache()