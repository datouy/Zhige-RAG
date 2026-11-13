"""Chroma 向量数据库封装 - 优化版。

能力：
- 初始化持久化客户端
- 创建/获取 collection
- 批量写入 Document（带 embedding 与 metadata）
- 按查询检索 Top-K
- 批量检索（多 query 并行）
- 按 metadata 过滤
- 删除文档（按 id 或 metadata 条件）
- 统计信息（chunk 数、来源数）

优化点：
- BM25 多粒度分词（完整词 + bigram + unigram）
- 自定义领域词典支持
- 混合检索 RRF 融合
- count() 结果缓存
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .embeddings import EmbeddingModel
from .utils import Timer, get_logger, resolve_path

logger = get_logger("vector_store")

_CHROMA_CLIENTS: Dict[str, Any] = {}
_CHROMA_CLIENT_LOCK = threading.Lock()


def get_shared_chroma_client(persist_directory: str) -> Any:
    """Return one persistent Chroma client per database directory.

    Chroma 1.x initializes its default tenant lazily. Creating several clients
    from the startup warmup and the first request at the same time can race and
    fail once with ``Could not connect to tenant default_tenant``. A shared,
    serialized client removes that race and avoids duplicate runtime state.
    """
    import chromadb
    from chromadb.config import Settings

    path = resolve_path(persist_directory).resolve()
    path.mkdir(parents=True, exist_ok=True)
    key = str(path)
    with _CHROMA_CLIENT_LOCK:
        client = _CHROMA_CLIENTS.get(key)
        if client is None:
            client = chromadb.PersistentClient(
                path=key,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False,
                ),
            )
            _CHROMA_CLIENTS[key] = client
        return client


class VectorStoreHealthError(RuntimeError):
    """向量库 collection 无法完成基础读写探测。"""


# ======================================================================
#  P1-4 向量库抽象：BaseStore 接口 + pgvector 选项
# ======================================================================
class BaseVectorStore:
    """向量库统一接口（Chroma / pgvector 共用）。

    RAGFlow/Dify 均用抽象层隔离 pg/vector 引擎，业务只依赖本接口，
    切换引擎无需改检索链路。新增引擎只需实现这 6 个方法。
    """

    def add_chunks(self, chunks, embeddings=None, batch_size: int = 128) -> int:  # pragma: no cover
        raise NotImplementedError

    def replace_chunks_for_sources(self, chunks, sources, on_commit=None) -> int:  # pragma: no cover
        raise NotImplementedError

    def query(self, query_text: str, top_k: int = 5, where=None, score_threshold: float = 0.0):  # pragma: no cover
        raise NotImplementedError

    def delete_by_ids(self, ids) -> int:  # pragma: no cover
        raise NotImplementedError

    def delete_by_metadata(self, where) -> int:  # pragma: no cover
        raise NotImplementedError

    def count(self) -> int:  # pragma: no cover
        raise NotImplementedError


def create_vector_store(
    backend: str = "chroma",
    pgvector_dsn: str = "",
    **kwargs,
) -> "BaseVectorStore":
    """按 backend 创建向量库：chroma（默认）| pgvector（需 psycopg + pgvector 扩展）。

    pgvector 未安装/无 DSN 时抛错并提示安装方式，不静默回落，避免“以为切库成功
    实际还在 chroma”的数据分裂。
    """
    backend = (backend or "chroma").strip().lower()
    if backend in ("pgvector", "pg", "postgres"):
        if not pgvector_dsn:
            raise VectorStoreHealthError("backend=pgvector 但未配置 pgvector_dsn")
        try:
            from .vector_store_pgvector import PgVectorStore  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreHealthError(
                f"pgvector 后端不可用：{exc}（需 pip install psycopg[binary] pgvector）"
            ) from exc
        return PgVectorStore(dsn=pgvector_dsn, **kwargs)
    return ChromaStore(**kwargs)


def probe_collections(client: Any) -> Dict[str, Any]:
    """探测客户端中的全部 collection。

    ``collection.count()`` 仍可能读取缓存或只走轻量元数据。真实故障（例如
    Chroma 升级后 metadata segment 编码不兼容）通常要到 ``get()`` 才暴露，
    因此对非空 collection 额外读取一条最小记录。这里不改数据，也不会加载
    embedding 模型。
    """
    collections = list(client.list_collections())
    reports: List[Dict[str, Any]] = []
    chunks_total = 0
    for collection in collections:
        name = str(getattr(collection, "name", collection))
        try:
            count = int(collection.count())
            if count > 0:
                sample = collection.get(
                    limit=1, include=["metadatas"]
                )
                ids = sample.get("ids") or []
                if not ids:
                    raise VectorStoreHealthError("count > 0 但读取样本为空")
            chunks_total += count
            reports.append({"name": name, "status": "healthy", "chunks": count})
        except Exception as exc:
            reports.append(
                {
                    "name": name,
                    "status": "unhealthy",
                    "error": str(exc)[:500],
                }
            )

    failures = [item for item in reports if item["status"] != "healthy"]
    return {
        "status": "healthy" if not failures else "unhealthy",
        "collections": len(reports),
        "chunks": chunks_total,
        "details": reports,
        "errors": failures,
    }


# ----------------------------------------------------------------------
#  BM25 多粒度分词（优化版）
# ----------------------------------------------------------------------
# 
# 多粒度策略：
# 1. 完整词（jieba 精准分词）- 主要匹配
# 2. 双字 bigram - 召回主力
# 3. 单字 unigram - 兜底防漏检
#
# 自定义词典支持行业术语、专有名词等
#

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")

# 中文停用词表（高频无区分力虚词）
_CJK_STOPWORDS = frozenset(
    "的 了 和 是 就 都 而 及 与 或 在 有 对 从 被 把 让 向 往 于 到 给 为 "
    "以 之 其 这 那 一个 一些 我们 你们 他们 她们 它们 什么 怎么 怎样 如何 "
    "以及 通过 对于 由于 因此 但是 可以 可能 进行 已经 关于 根据 按照 同时 "
    "另外 然后 所以 如果 虽然 并且 还是 只是 还有 等等 以上 以下 这些 那些 "
    "没有 是不是 为什么 多少 哪里 哪个 哪些 怎样 如何".split()
)

# jieba 探测状态
_JIEBA_STATE = "unknown"
_JIEBA_CUSTOM_WORDS: Set[str] = set()
_JIEBA_INITIALIZED = False


def _init_jieba_with_custom_dict() -> None:
    """初始化 jieba 并加载自定义词典。"""
    global _JIEBA_STATE, _JIEBA_CUSTOM_WORDS, _JIEBA_INITIALIZED
    
    if _JIEBA_INITIALIZED:
        return
    _JIEBA_INITIALIZED = True
    
    try:
        import jieba
        jieba.setLogLevel(logging.ERROR)

        if not _JIEBA_CUSTOM_WORDS:
            dictionary = resolve_path("config/custom_dict.txt")
            if dictionary.exists():
                for line in dictionary.read_text(encoding="utf-8").splitlines():
                    word = line.strip()
                    if word and not word.startswith("#"):
                        _JIEBA_CUSTOM_WORDS.add(word)

        # 加载自定义词典
        for word in _JIEBA_CUSTOM_WORDS:
            jieba.add_word(word)
        
        _JIEBA_STATE = "on"
        if _JIEBA_CUSTOM_WORDS:
            logger.info(f"BM25 分词器：jieba + 自定义词典({len(_JIEBA_CUSTOM_WORDS)}词)")
        else:
            logger.info("BM25 分词器：jieba 精确模式")
    except ImportError:
        _JIEBA_STATE = "off"
        logger.info("BM25 分词器：jieba 未安装，回退 unigram+bigram")
    except Exception as exc:
        _JIEBA_STATE = "off"
        logger.warning(f"jieba 初始化失败: {exc}")


def add_custom_words(words: List[str]) -> None:
    """添加自定义词典词汇（用于领域术语、专有名词等）。"""
    global _JIEBA_CUSTOM_WORDS
    _JIEBA_CUSTOM_WORDS.update(words)
    if _JIEBA_INITIALIZED and _JIEBA_STATE == "on":
        import jieba

        for word in words:
            jieba.add_word(word)
    logger.info(f"已添加 {len(words)} 个自定义词典词")


def _jieba_cut(seg: str) -> Optional[List[str]]:
    """用 jieba 精确模式切分 CJK 段。"""
    global _JIEBA_STATE, _JIEBA_INITIALIZED
    
    if not _JIEBA_INITIALIZED:
        _init_jieba_with_custom_dict()
    
    if _JIEBA_STATE != "on":
        return None
    
    import jieba
    return [w for w in jieba.lcut(seg) if w.strip() and w not in _CJK_STOPWORDS]


def _multi_granularity_tokens(seg: str) -> List[str]:
    """
    多粒度分词策略：
    1. jieba 完整词
    2. 双字 bigram
    3. 单字 unigram（兜底）
    """
    # 1. 完整词（优先）
    words = _jieba_cut(seg)
    if words is not None:
        return words
    
    # 2. 回退到 unigram + bigram
    chars = list(seg)
    tokens = []
    
    # unigram（所有字符）
    for c in chars:
        if c not in _CJK_STOPWORDS:
            tokens.append(c)
    
    # bigram（相邻字符对）
    for i in range(len(chars) - 1):
        bigram = chars[i] + chars[i + 1]
        if bigram not in _CJK_STOPWORDS:
            tokens.append(bigram)
    
    return tokens


def _tokenize(text: str) -> List[str]:
    """
    多粒度分词主函数。
    对中英文采用不同策略，输出去重。
    """
    tokens: List[str] = []
    seen: Set[str] = set()
    
    for m in _TOKEN_RE.finditer(text or ""):
        seg = m.group(0)
        if seg[0].isascii():
            # 英文/数字：整词匹配
            token = seg.lower()
            if token not in seen:
                tokens.append(token)
                seen.add(token)
        else:
            # 中文：多粒度
            for tok in _multi_granularity_tokens(seg):
                if tok not in seen:
                    tokens.append(tok)
                    seen.add(tok)
    
    return tokens


class BM25Index:
    """内存 BM25 索引（k1/b 为标准参数）。

    优化：
    - 使用多粒度分词，提升中文检索召回率。
    - 线程安全：search / upsert / remove 用 RLock 互斥，避免查询与入库并发读写同一份 ids/metadatas/texts 时出现撕裂。
    - 维护 _id_to_pos 映射，避免每次检索重建 O(N) 的 id->位置 表。
    """

    def __init__(
        self, 
        ids: Sequence[str], 
        texts: Sequence[str], 
        metadatas: Sequence[Dict[str, Any]], 
        k1: float = 1.5, 
        b: float = 0.75
    ) -> None:
        self._lock = threading.RLock()
        self.ids = list(ids)
        self.metadatas = list(metadatas)
        self.texts = list(texts)
        self.k1, self.b = k1, b
        self.docs_tokens: List[List[str]] = [_tokenize(t) for t in self.texts]
        self._id_to_pos: Dict[str, int] = {cid: i for i, cid in enumerate(self.ids)}
        self.N = len(self.ids)
        self.avgdl = (sum(len(d) for d in self.docs_tokens) / self.N) if self.N else 0.0
        self.df: Counter = Counter()
        self.tfs: List[Counter] = []
        
        for toks in self.docs_tokens:
            tf = Counter(toks)
            self.tfs.append(tf)
            for term in tf:
                self.df[term] += 1

    def _rebuild_stats(self) -> None:
        self.N = len(self.ids)
        self.avgdl = (
            sum(len(tokens) for tokens in self.docs_tokens) / self.N
            if self.N
            else 0.0
        )
        self.df = Counter()
        self.tfs = []
        for tokens in self.docs_tokens:
            tf = Counter(tokens)
            self.tfs.append(tf)
            self.df.update(tf.keys())

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ids": list(self.ids),
                "texts": list(self.texts),
                "metadatas": list(self.metadatas),
                "k1": self.k1,
                "b": self.b,
                "version": 1,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25Index":
        return cls(
            ids=data.get("ids") or [],
            texts=data.get("texts") or [],
            metadatas=data.get("metadatas") or [],
            k1=float(data.get("k1", 1.5)),
            b=float(data.get("b", 0.75)),
        )

    def save(self, path: Any) -> None:
        """原子写：先写临时文件再 os.replace，避免进程被 kill 时留下半截 pkl。"""
        try:
            import os
            import pickle
            import tempfile

            p = str(path)
            directory = os.path.dirname(p) or "."
            os.makedirs(directory, exist_ok=True)
            data = self.to_dict()
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(data, f)
                os.replace(tmp, p)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception:
            pass

    @classmethod
    def load(cls, path: Any) -> Optional["BM25Index"]:
        try:
            import os
            import pickle

            p = str(path)
            if not os.path.exists(p):
                return None
            with open(p, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                return None
            return cls.from_dict(data)
        except Exception:
            return None

    def upsert_documents(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        """增量覆盖指定 chunk，不重新读取或分词整个 collection。"""
        with self._lock:
            for chunk_id, text, metadata in zip(ids, texts, metadatas):
                tokens = _tokenize(text)
                position = self._id_to_pos.get(chunk_id)
                if position is None:
                    position = len(self.ids)
                    self._id_to_pos[chunk_id] = position
                    self.ids.append(chunk_id)
                    self.texts.append(text)
                    self.metadatas.append(metadata)
                    self.docs_tokens.append(tokens)
                else:
                    self.texts[position] = text
                    self.metadatas[position] = metadata
                    self.docs_tokens[position] = tokens
            self._rebuild_stats()

    def remove_ids(self, ids: Sequence[str]) -> None:
        """删除指定 chunk 并重建词频统计（不重新分词）。"""
        remove = set(ids)
        if not remove:
            return
        with self._lock:
            kept = [
                index
                for index, chunk_id in enumerate(self.ids)
                if chunk_id not in remove
            ]
            self.ids = [self.ids[index] for index in kept]
            self.texts = [self.texts[index] for index in kept]
            self.metadatas = [self.metadatas[index] for index in kept]
            self.docs_tokens = [self.docs_tokens[index] for index in kept]
            self._id_to_pos = {cid: i for i, cid in enumerate(self.ids)}
            self._rebuild_stats()

    def search(self, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        """返回 [(chunk_id, bm25_score)]，按分数降序。"""
        with self._lock:
            if not self.N:
                return []
            
            q_terms = set(_tokenize(query_text))
            if not q_terms:
                return []
            
            scores: Dict[int, float] = {}
            for term in q_terms:
                df = self.df.get(term)
                if not df:
                    continue
                idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
                
                for i, tf in enumerate(self.tfs):
                    f = tf.get(term)
                    if not f:
                        continue
                    dl = len(self.docs_tokens[i])
                    norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
                    scores[i] = scores.get(i, 0.0) + idf * (f * (self.k1 + 1.0)) / (f + norm)
            
            ranked = sorted(scores.items(), key=lambda x: -x[1])[:max(1, int(top_k))]
            return [(self.ids[i], s) for i, s in ranked]

    def get_by_id(self, chunk_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """按 id 取 (text, metadata)；不存在返回 None。线程安全。"""
        with self._lock:
            position = self._id_to_pos.get(chunk_id)
            if position is None:
                return None
            return (self.texts[position], self.metadatas[position])

    def matching_ids(self, where: Optional[Dict[str, Any]]) -> List[str]:
        """返回满足 where 条件的 chunk id（线程安全）。"""
        with self._lock:
            if not where:
                return list(self.ids)
            return [
                self.ids[i]
                for i, metadata in enumerate(self.metadatas)
                if _metadata_matches(metadata, where)
            ]


@dataclass
class Hit:
    """检索命中结果。"""

    id: str
    text: str
    score: float
    metadata: Dict[str, Any]

    @property
    def fusion_rank(self) -> Optional[int]:
        """融合名次（RRF 序）。

        混合检索里 ``score`` 的量纲是**混合的**：向量侧是余弦相似度
        （0~1），关键词侧是 ``s / (1 + s)``（BM25 原始分的压缩），两者
        不可直接比较。RRF 融合用的是"名次"而不是"分数"，所以排序的真值
        应该是融合名次，而不是 ``score``。

        融合层会把 ``_fusion_rank`` 写进 metadata；这里做一次防御性读取，
        没有则返回 None（表示"未参与融合，保持原序"）。
        """
        v = self.metadata.get("_fusion_rank")
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def score_kind(self) -> str:
        """``score`` 的语义类型：``vector`` / ``keyword`` / ``fusion``。"""
        return str(self.metadata.get("_score_kind") or "vector")

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text, "score": self.score, "metadata": self.metadata}


def _make_id(source: str, page: int, chunk_index: int) -> str:
    """生成稳定 id。"""
    raw = f"{source}|{page}|{chunk_index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_security_where(
    extra_where: Optional[Dict[str, Any]] = None,
    status_filter: bool = True,
    acl_groups: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """构造数据层安全过滤条件（Chroma where 语法）。"""
    clauses: List[Dict[str, Any]] = []
    if status_filter:
        clauses.append({"doc_status": "active"})
    if acl_groups:
        clauses.append({"acl": {"$in": [str(g) for g in acl_groups]}})
    if extra_where:
        clauses.append(extra_where)
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def hybrid_kwargs(vs_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """从 vector_store 配置段构造 ChromaStore 的混合检索参数。"""
    h = (vs_cfg or {}).get("hybrid") or {}
    return {
        "hybrid_enabled": bool(h.get("enabled", False)),
        "hybrid_fetch_k": int(h.get("fetch_k", 20)),
        "hybrid_bm25_weight": float(h.get("bm25_weight", 0.5)),
    }


def _metadata_matches(metadata: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    """判断一条 metadata 是否满足 Chroma 的 ``where`` 条件。

    只覆盖本项目实际会生成的算子：``=`` / ``$eq`` / ``$ne`` / ``$in`` / ``$nin``
    以及 ``$and`` / ``$or`` 组合（见 :func:`build_security_where`）。

    **认不出来就返回 False（fail-closed）**：宁可少一条关键词召回，也不能让被
    ``doc_status`` / ``acl`` 排除掉的文档从关键词这一路溜回结果里。
    """
    if not where:
        return True
    meta = metadata or {}
    for key, cond in where.items():
        if key == "$and":
            if not all(_metadata_matches(meta, c) for c in (cond or [])):
                return False
            continue
        if key == "$or":
            if not any(_metadata_matches(meta, c) for c in (cond or [])):
                return False
            continue
        if key.startswith("$"):
            return False  # 不认识的顶层算子：保守排除
        value = meta.get(key)
        if isinstance(cond, dict):
            if len(cond) != 1:
                return False
            op, operand = next(iter(cond.items()))
            seq = operand if isinstance(operand, (list, tuple, set)) else None
            if op == "$eq":
                ok = value == operand
            elif op == "$ne":
                ok = value != operand
            elif op == "$in":
                ok = seq is not None and value in seq
            elif op == "$nin":
                ok = seq is not None and value not in seq
            else:
                return False
            if not ok:
                return False
            continue
        if value != cond:
            return False
    return True


class ChromaStore(BaseVectorStore):
    """Chroma 向量库封装。"""

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "chinese_rag_kb",
        embedding_model: Optional[EmbeddingModel] = None,
        distance_fn: str = "cosine",
        hybrid_enabled: bool = False,
        hybrid_fetch_k: int = 20,
        hybrid_bm25_weight: float = 0.5,
    ) -> None:
        try:
            import chromadb  # noqa: F401
        except ImportError as exc:
            raise ImportError("未安装 chromadb，请运行 `pip install chromadb`") from exc

        self.persist_directory = resolve_path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = get_shared_chroma_client(str(self.persist_directory))

        if embedding_model is None:
            logger.info("未提供 embedding_model，使用默认的 Chroma 内置 embedding")
            self.embedding_model = None
        else:
            self.embedding_model = embedding_model

        distance_map = {"cosine": "cosine", "l2": "l2", "ip": "ip"}
        self.distance_fn = distance_map.get(distance_fn, "cosine")
        metadata_cfg = {"hnsw:space": self.distance_fn}
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata_cfg,
        )
        logger.info("Chroma 集合已就绪：%s (当前 %d 条记录)", collection_name, self.collection.count())

        # count 缓存
        self._count_cache: Optional[int] = None
        self._count_cache_valid = False
        # 内容变更版本号：每次写/删递增，用于答案缓存键（替换等量文档也能失效）
        self._mutation_version = 0

        # 混合检索（BM25 + 向量 RRF 融合）
        self.hybrid_enabled = bool(hybrid_enabled)
        self.hybrid_fetch_k = max(int(hybrid_fetch_k), 1)
        # 关键词路在 RRF 里的权重：只做召回补充，不压过向量相似度
        # （取值理由见 _fuse_hybrid 的说明）
        self.hybrid_bm25_weight = max(0.0, float(hybrid_bm25_weight))
        self._bm25: Optional[BM25Index] = None
        self._bm25_lock = threading.Lock()
        # P1：BM25 磁盘持久化路径（与 collection 同目录，避免冷启全量 get+分词）
        self._bm25_path = self.persist_directory / f"bm25_{collection_name}.pkl"

    def health_check(self, query_probe: bool = True) -> Dict[str, Any]:
        """执行不会写数据的一致性探测，失败时抛出明确异常。"""
        report = probe_collections(self.client)
        report["active_collection"] = self.collection_name
        if report["status"] != "healthy":
            errors = "; ".join(
                f"{item['name']}: {item.get('error', 'unknown')}"
                for item in report["errors"][:3]
            )
            raise VectorStoreHealthError(errors or "向量库健康检查失败")

        target_collection = self.collection
        if query_probe and self.count() == 0:
            non_empty = next(
                (item for item in report.get("details", []) if item.get("chunks")),
                None,
            )
            if non_empty is not None:
                target_collection = self.client.get_collection(non_empty["name"])

        can_query_probe = query_probe and int(target_collection.count()) > 0
        if can_query_probe:
            try:
                sample = target_collection.get(limit=1, include=["embeddings"])
                embeddings = sample.get("embeddings")
                if embeddings is None or len(embeddings) == 0:
                    raise VectorStoreHealthError("无法读取已存向量")
                dimension = len(embeddings[0])
                expected_dim = getattr(self.embedding_model, "dim", None)
                if expected_dim and int(expected_dim) != dimension:
                    raise VectorStoreHealthError(
                        f"向量维度不一致：collection={dimension}, model={expected_dim}"
                    )
                # 使用非零探针向量验证 HNSW 查询路径，不触发模型加载。
                probe_vector = [1.0] + [0.0] * max(0, dimension - 1)
                target_collection.query(query_embeddings=[probe_vector], n_results=1)
            except Exception as exc:
                raise VectorStoreHealthError(
                    f"collection {target_collection.name} 查询探测失败: {exc}"
                ) from exc
        report["query_probe"] = bool(can_query_probe)
        return report

    def _invalidate_bm25(self) -> None:
        """数据变更后使 BM25 索引失效（并尽力落盘，下次冷启直接复用）。"""
        try:
            if self._bm25 is not None:
                self._bm25.save(getattr(self, "_bm25_path", "bm25.pkl"))
        except Exception:
            pass
        self._bm25 = None

    def _ensure_bm25(self) -> BM25Index:
        if self._bm25 is None:
            # 并发首查时只允许一个线程构建索引，避免重复全量读取和分词。
            with self._bm25_lock:
                if self._bm25 is None:
                    # P1：优先读磁盘持久化，N 对不上再全量重建（万级以上冷启分钟级→秒级）
                    try:
                        cached = BM25Index.load(getattr(self, "_bm25_path", "bm25.pkl"))
                        if cached is not None:
                            try:
                                cur_n = int(self.collection.count())
                            except Exception:
                                cur_n = -1
                            if cur_n < 0 or int(cached.N) == cur_n:
                                self._bm25 = cached
                                return self._bm25
                    except Exception:
                        pass
                    got = self.collection.get(include=["documents", "metadatas"])
                    self._bm25 = BM25Index(
                        ids=got.get("ids") or [],
                        texts=got.get("documents") or [],
                        metadatas=got.get("metadatas") or [],
                    )
                    try:
                        self._bm25.save(getattr(self, "_bm25_path", "bm25.pkl"))
                    except Exception:
                        pass
        return self._bm25

    def _fuse_hybrid(
        self,
        query_text: str,
        vector_hits: List[Hit],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Hit]:
        """向量命中与 BM25 命中做 RRF 倒数排名融合。

        ``where`` 必须**同时**作用在两条路上：向量侧交给 Chroma 过滤，关键词侧
        由 :func:`_metadata_matches` 过滤。此前只过滤了向量侧，于是被
        ``doc_status`` / ``acl`` 排除的文档会从 BM25 溜回结果 —— 既越权，也让
        无关文档拿到高分、把真正相关的内容挤出上下文。

        关键词路的权重由 ``hybrid.bm25_weight`` 控制（默认 0.5）。理由：本项目的
        中文分词是多粒度 unigram+bigram，BM25 分数是命中词条的累加、**没有做查询
        长度归一**，短查询下长文档极易拿高分 —— 实测一条 Node.js 面试题对
        「用的是什么模型 完成的是什么功能？」能排到关键词第 1，折算出的"相似度"
        0.90 压过了真正相关片段的 0.55。让它只做召回补充，才不至于颠倒是非。
        """
        vec_by_id = {h.id: h for h in vector_hits}
        rrf: Dict[str, float] = {}
        RRF_K = 60

        for rank, h in enumerate(vector_hits):
            rrf[h.id] = rrf.get(h.id, 0.0) + 1.0 / (RRF_K + rank + 1)

        bm = self._ensure_bm25()
        bm_score: Dict[str, float] = {}
        kept = 0
        for cid, s in bm.search(query_text, top_k=max(self.hybrid_fetch_k * 4, 20)):
            rec = bm.get_by_id(cid)
            if rec is None or not _metadata_matches(rec[1], where):
                continue
            bm_score[cid] = s
            rrf[cid] = rrf.get(cid, 0.0) + self.hybrid_bm25_weight / (RRF_K + kept + 1)
            kept += 1
            if kept >= self.hybrid_fetch_k:
                break

        if not rrf:
            return vector_hits

        fused_ids = sorted(rrf, key=lambda cid: -rrf[cid])[: max(1, int(top_k))]

        hits: List[Hit] = []
        for rank, cid in enumerate(fused_ids):
            if cid in vec_by_id:
                h = vec_by_id[cid]
                md = dict(h.metadata or {})
                md["_fusion_rank"] = rank
                md["_fusion_score"] = float(rrf[cid])
                md.setdefault("_score_kind", "vector")
                hits.append(Hit(id=h.id, text=h.text, score=h.score, metadata=md))
                continue
            rec = bm.get_by_id(cid)
            if rec is None:
                continue
            text, md_src = rec
            s = bm_score.get(cid, 0.0)
            md = dict(md_src or {})
            md["_score_kind"] = "keyword"
            md["_fusion_rank"] = rank
            md["_fusion_score"] = float(rrf[cid])
            hits.append(
                Hit(
                    id=cid,
                    text=text,
                    score=(s / (1.0 + s)) if s > 0 else 0.0,
                    metadata=md,
                )
            )
        return hits

    def add_chunks(
        self,
        chunks: Sequence[Any],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        batch_size: int = 128,
    ) -> int:
        """批量写入分块。"""
        chunks = list(chunks)
        if not chunks:
            return 0

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            unique: Dict[str, Dict[str, Any]] = {}
            
            for c in batch:
                md = dict(c.metadata or {})
                src = str(md.get("source") or md.get("filepath") or "unknown")
                page = int(md.get("page") or 1)
                cid = _make_id(src, page, c.index)
                safe_md = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in md.items()}
                # 落库时记住分块在文档内的原始序号。检索结果默认按相似度排序，
                # 直接拿它当上下文会把原文顺序打乱 —— 因果链、时间线都断了，
                # 模型只能做平行罗列。有了 chunk_index，_build_context 才能
                # 按原文顺序装配上下文（老索引没有该字段，保持原行为）。
                if "chunk_index" not in safe_md:
                    try:
                        safe_md["chunk_index"] = int(getattr(c, "index", 0) or 0)
                    except (TypeError, ValueError):
                        pass
                unique[cid] = {"text": c.text, "metadata": safe_md}

            ids = list(unique.keys())
            texts = [v["text"] for v in unique.values()]
            metadatas = [v["metadata"] for v in unique.values()]

            if embeddings is None:
                if self.embedding_model is None:
                    raise RuntimeError("未提供 embeddings 且未配置 embedding 模型")
                embeds = self.embedding_model.encode(texts, batch_size=batch_size).tolist()
            else:
                aligned: Dict[str, Sequence[float]] = {}
                for c, e in zip(batch, embeddings[start:start + batch_size]):
                    md = dict(c.metadata or {})
                    src = str(md.get("source") or md.get("filepath") or "unknown")
                    page = int(md.get("page") or 1)
                    aligned[_make_id(src, page, c.index)] = list(map(float, e))
                embeds = [list(map(float, aligned[i])) for i in ids]

            with Timer(f"chroma upsert x{len(batch)}"):
                self.collection.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=embeds,
                    metadatas=metadatas,
                )
            if self._bm25 is not None:
                self._bm25.upsert_documents(ids, texts, metadatas)
            total += len(ids)

        self._invalidate_count_cache(invalidate_bm25=False)
        self._bump_mutation()
        logger.info("写入完成：%d 条", total)
        return total

    def replace_chunks_for_sources(
        self,
        chunks: Sequence[Any],
        sources: Sequence[str],
        on_commit: Optional[Callable[[], None]] = None,
    ) -> int:
        """先写新分块、再删旧分块，失败时恢复变更前的 collection。

        ``on_commit`` 用于把数据库台账更新纳入同一补偿边界：回调抛错时，
        本方法会恢复旧向量并删除本次新增的 chunk id，避免“向量已更新但台账
        仍是旧数据”的半成功状态。
        """
        chunks = list(chunks)
        source_set = sorted({str(source) for source in sources if source})
        if not chunks:
            if on_commit is not None:
                on_commit()
            return 0
        if not source_set:
            raise ValueError("replace_chunks_for_sources 需要至少一个 source")

        where = {"source": {"$in": source_set}}
        old = self.collection.get(
            where=where,
            include=["documents", "metadatas", "embeddings"],
        )
        old_ids = [str(chunk_id) for chunk_id in (old.get("ids") or [])]
        old_id_set = set(old_ids)

        new_ids: List[str] = []
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            source = str(metadata.get("source") or metadata.get("filepath") or "unknown")
            page = int(metadata.get("page") or 1)
            new_ids.append(_make_id(source, page, chunk.index))
        new_id_set = set(new_ids)

        def _restore_old() -> None:
            new_only = list(new_id_set - old_id_set)
            if new_only:
                self.collection.delete(ids=new_only)
            if old_ids:
                self.collection.upsert(
                    ids=old_ids,
                    documents=old.get("documents") or [],
                    metadatas=old.get("metadatas") or [],
                    embeddings=[
                        list(map(float, vector))
                        for vector in (old.get("embeddings") or [])
                    ],
                )
            self._invalidate_count_cache()
            self._bump_mutation()

        try:
            inserted = self.add_chunks(chunks)
            stale_ids = list(old_id_set - new_id_set)
            if stale_ids:
                self.collection.delete(ids=stale_ids)
                if self._bm25 is not None:
                    self._bm25.remove_ids(stale_ids)
                self._invalidate_count_cache(invalidate_bm25=False)
            if on_commit is not None:
                on_commit()
            return inserted
        except Exception as exc:
            try:
                _restore_old()
            except Exception as rollback_exc:
                raise RuntimeError(
                    "分块写入失败，且旧数据恢复失败；请立即从 Chroma 备份恢复"
                    f"（sources={source_set}, write_error={exc}, rollback_error={rollback_exc})"
                ) from exc
            raise

    def get_by_source(self, source: str, where: Optional[Dict[str, Any]] = None) -> List[Hit]:
        """读取指定来源文档的全部分块，可叠加安全过滤条件。

        返回**按 chunk_index 升序**（即原文顺序）的分块：``collection.get``
        不保证顺序，拿乱序分块拼上下文会让"先说结论、后说原因"的叙述错位。
        老索引没有 ``chunk_index`` 字段时保持存储顺序不动。
        """
        filters: List[Dict[str, Any]] = [{"source": source}]
        if where:
            filters.append(where)
        combined = filters[0] if len(filters) == 1 else {"$and": filters}
        data = self.collection.get(where=combined, include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        hits = [
            Hit(id=ids[i], text=docs[i] or "", score=1.0,
                metadata=metas[i] if i < len(metas) else {})
            for i in range(len(ids))
        ]
        if len(hits) <= 1:
            return hits
        indices: List[Optional[int]] = []
        for hit in hits:
            try:
                indices.append(int((hit.metadata or {}).get("chunk_index")))
            except (TypeError, ValueError):
                indices.append(None)
        if all(index is not None for index in indices):
            hits = [h for _, h in sorted(zip(indices, hits), key=lambda pair: pair[0])]
        return hits

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Hit]:
        """检索 Top-K 命中文档。"""
        if self.count() == 0:
            return []
        results = self.query_batch([query_text], top_k=top_k, where=where, score_threshold=score_threshold)
        return results[0] if results else []

    def query_batch(
        self,
        query_texts: List[str],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[List[Hit]]:
        """批量检索：多个 query 并行编码，一次 Chroma 调用返回。"""
        if not query_texts:
            return []

        if self.embedding_model is None:
            raise RuntimeError("当前未配置 embedding 模型，无法进行语义检索")

        # 一次批量编码所有 query
        with Timer(f"embedding encode x{len(query_texts)}"):
            q_vecs = self.embedding_model.encode(query_texts, batch_size=len(query_texts)).tolist()

        kwargs: Dict[str, Any] = {
            "query_embeddings": q_vecs,
            "n_results": max(1, int(top_k)),
        }
        if where:
            kwargs["where"] = where

        with Timer(f"chroma batch query x{len(query_texts)}"):
            res = self.collection.query(**kwargs)

        results: List[List[Hit]] = []
        for qi, query_text in enumerate(query_texts):
            ids = (res.get("ids") or [[]])[qi]
            docs = (res.get("documents") or [[]])[qi]
            metas = (res.get("metadatas") or [[]])[qi]
            dists = (res.get("distances") or [[]])[qi]

            hits: List[Hit] = []
            for i, _id in enumerate(ids):
                dist = float(dists[i]) if i < len(dists) else 0.0
                if self.distance_fn == "l2":
                    score = 1.0 / (1.0 + max(0.0, dist))
                elif self.distance_fn == "ip":
                    score = dist
                else:
                    score = max(0.0, 1.0 - dist)
                
                if score < score_threshold:
                    continue
                hits.append(
                    Hit(
                        id=_id,
                        text=docs[i] if i < len(docs) else "",
                        score=score,
                        metadata=metas[i] if i < len(metas) else {},
                    )
                )
            
            if self.hybrid_enabled:
                hits = self._fuse_hybrid(query_text, hits, top_k, where=where)
            results.append(hits)
        
        return results

    def _bump_mutation(self) -> None:
        self._mutation_version += 1

    def _invalidate_count_cache(self, invalidate_bm25: bool = True) -> None:
        """写入/删除后调用，使 count 缓存失效。"""
        self._count_cache_valid = False
        if invalidate_bm25:
            self._invalidate_bm25()

    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        """按 metadata 条件删除文档，返回删除条数。"""
        if not where:
            raise ValueError("where 条件不能为空，避免误删")
        matching_ids = self._bm25.matching_ids(where) if self._bm25 is not None else []
        before = self.collection.count()
        self.collection.delete(where=where)
        after = self.collection.count()
        deleted = before - after
        if self._bm25 is not None:
            self._bm25.remove_ids(matching_ids)
        self._invalidate_count_cache(invalidate_bm25=False)
        self._bump_mutation()
        logger.info("按 %s 删除 %d 条", where, deleted)
        return deleted

    def delete_by_ids(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        before = self.collection.count()
        self.collection.delete(ids=list(ids))
        after = self.collection.count()
        if self._bm25 is not None:
            self._bm25.remove_ids(ids)
        self._invalidate_count_cache(invalidate_bm25=False)
        self._bump_mutation()
        return before - after

    def list_sources(self) -> List[Dict[str, Any]]:
        """列出所有来源文档及其分块数。"""
        if self.count() == 0:
            return []
        data = self.collection.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        agg: Dict[str, Dict[str, Any]] = {}
        for m in metas:
            src = str(m.get("source") or m.get("filepath") or "unknown")
            entry = agg.setdefault(src, {"source": src, "chunks": 0})
            entry["chunks"] += 1
        return sorted(agg.values(), key=lambda x: -x["chunks"])

    def count(self) -> int:
        """返回分块总数（带缓存）。"""
        if self._count_cache_valid:
            return self._count_cache
        self._count_cache = self.collection.count()
        self._count_cache_valid = True
        return self._count_cache

    def reset(self) -> None:
        """清空当前集合（谨慎使用）。"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._invalidate_count_cache()
        self._bump_mutation()
        logger.warning("集合 %s 已重置", self.collection_name)
