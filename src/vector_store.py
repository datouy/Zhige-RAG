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
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .embeddings import EmbeddingModel
from .utils import Timer, get_logger, resolve_path

logger = get_logger("vector_store")


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
    """
    内存 BM25 索引（k1/b 为标准参数）。
    
    优化：使用多粒度分词，提升中文检索召回率。
    """

    def __init__(
        self, 
        ids: Sequence[str], 
        texts: Sequence[str], 
        metadatas: Sequence[Dict[str, Any]], 
        k1: float = 1.5, 
        b: float = 0.75
    ) -> None:
        self.ids = list(ids)
        self.metadatas = list(metadatas)
        self.texts = list(texts)
        self.k1, self.b = k1, b
        self.docs_tokens: List[List[str]] = [_tokenize(t) for t in self.texts]
        self.N = len(self.ids)
        self.avgdl = (sum(len(d) for d in self.docs_tokens) / self.N) if self.N else 0.0
        self.df: Counter = Counter()
        self.tfs: List[Counter] = []
        
        for toks in self.docs_tokens:
            tf = Counter(toks)
            self.tfs.append(tf)
            for term in tf:
                self.df[term] += 1

    def search(self, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        """返回 [(chunk_id, bm25_score)]，按分数降序。"""
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


@dataclass
class Hit:
    """检索命中结果。"""

    id: str
    text: str
    score: float
    metadata: Dict[str, Any]

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
    return {"hybrid_enabled": bool(h.get("enabled", False)), "hybrid_fetch_k": int(h.get("fetch_k", 20))}


class ChromaStore:
    """Chroma 向量库封装。"""

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "chinese_rag_kb",
        embedding_model: Optional[EmbeddingModel] = None,
        distance_fn: str = "cosine",
        hybrid_enabled: bool = False,
        hybrid_fetch_k: int = 20,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise ImportError("未安装 chromadb，请运行 `pip install chromadb`") from exc

        self.persist_directory = resolve_path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )

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

        # 混合检索（BM25 + 向量 RRF 融合）
        self.hybrid_enabled = bool(hybrid_enabled)
        self.hybrid_fetch_k = max(int(hybrid_fetch_k), 1)
        self._bm25: Optional[BM25Index] = None

    def _invalidate_bm25(self) -> None:
        """数据变更后使 BM25 索引失效。"""
        self._bm25 = None

    def _ensure_bm25(self) -> BM25Index:
        if self._bm25 is None:
            got = self.collection.get(include=["documents", "metadatas"])
            self._bm25 = BM25Index(
                ids=got.get("ids") or [],
                texts=got.get("documents") or [],
                metadatas=got.get("metadatas") or [],
            )
        return self._bm25

    def _fuse_hybrid(
        self,
        query_text: str,
        vector_hits: List[Hit],
        top_k: int,
    ) -> List[Hit]:
        """向量命中与 BM25 命中做 RRF 倒数排名融合。"""
        vec_by_id = {h.id: h for h in vector_hits}
        rrf: Dict[str, float] = {}
        RRF_K = 60
        
        # 向量检索排名
        for rank, h in enumerate(vector_hits):
            rrf[h.id] = rrf.get(h.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        
        # BM25 检索排名
        bm_hits = self._ensure_bm25().search(query_text, top_k=self.hybrid_fetch_k)
        bm_score: Dict[str, float] = {}
        for rank, (cid, s) in enumerate(bm_hits):
            bm_score[cid] = s
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        
        if not rrf:
            return vector_hits

        fused_ids = sorted(rrf, key=lambda cid: -rrf[cid])[:max(1, int(top_k))]
        bm = self._ensure_bm25()
        bm_meta = {cid: i for i, cid in enumerate(bm.ids)}
        
        hits: List[Hit] = []
        for cid in fused_ids:
            if cid in vec_by_id:
                hits.append(vec_by_id[cid])
            else:
                idx = bm_meta.get(cid)
                if idx is None:
                    continue
                s = bm_score.get(cid, 0.0)
                hits.append(
                    Hit(
                        id=cid,
                        text=bm.texts[idx],
                        score=(s / (1.0 + s)) if s > 0 else 0.0,
                        metadata=bm.metadatas[idx] or {},
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
            total += len(ids)

        self._invalidate_count_cache()
        logger.info("写入完成：%d 条", total)
        return total

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
                hits = self._fuse_hybrid(query_text, hits, top_k)
            results.append(hits)
        
        return results

    def _invalidate_count_cache(self) -> None:
        """写入/删除后调用，使 count 缓存失效。"""
        self._count_cache_valid = False
        self._invalidate_bm25()

    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        """按 metadata 条件删除文档，返回删除条数。"""
        if not where:
            raise ValueError("where 条件不能为空，避免误删")
        before = self.collection.count()
        self.collection.delete(where=where)
        after = self.collection.count()
        deleted = before - after
        self._invalidate_count_cache()
        logger.info("按 %s 删除 %d 条", where, deleted)
        return deleted

    def delete_by_ids(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        before = self.collection.count()
        self.collection.delete(ids=list(ids))
        after = self.collection.count()
        self._invalidate_count_cache()
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
        logger.warning("集合 %s 已重置", self.collection_name)
