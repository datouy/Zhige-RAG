"""Chroma 向量数据库封装。

能力：
- 初始化持久化客户端
- 创建/获取 collection
- 批量写入 Document（带 embedding 与 metadata）
- 按查询检索 Top-K
- 按 metadata 过滤
- 删除文档（按 id 或 metadata 条件）
- 统计信息（chunk 数、来源数）

注意：写入的 id 默认基于 ``source + page + chunk_index`` 的 MD5，保证同一文档重复入库可幂等覆盖。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .embeddings import EmbeddingModel
from .utils import Timer, get_logger, resolve_path

logger = get_logger("vector_store")


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


class ChromaStore:
    """Chroma 向量库封装。

    Args:
        persist_directory: 持久化目录。
        collection_name: 集合名。
        embedding_model: 已初始化的 EmbeddingModel 实例。
        distance_fn: 距离函数 cosine / l2 / ip。
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "chinese_rag_kb",
        embedding_model: Optional[EmbeddingModel] = None,
        distance_fn: str = "cosine",
    ) -> None:
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore
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
            logger.info("未提供 embedding_model，使用默认的 Chroma 内置 embedding（不推荐）")
            self.embedding_model = None
        else:
            self.embedding_model = embedding_model

        distance_map = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip",
        }
        metadata_cfg = {"hnsw:space": distance_map.get(distance_fn, "cosine")}
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata_cfg,
        )
        logger.info(
            "Chroma 集合已就绪：%s (当前 %d 条记录)",
            collection_name,
            self.collection.count(),
        )

    # ------------------------------------------------------------------
    def add_chunks(
        self,
        chunks: Sequence[Any],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        batch_size: int = 128,
    ) -> int:
        """批量写入分块。

        Args:
            chunks: 可迭代对象，每个元素需要有 ``text, index, metadata`` 三个属性。
            embeddings: 与 chunks 对齐的 embedding 向量；若为空则调用内部 embedding 模型。
            batch_size: 写入批大小。

        Returns:
            实际写入条数。
        """
        # 物化列表
        chunks = list(chunks)
        if not chunks:
            return 0

        # 分批
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            ids: List[str] = []
            texts: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            for c in batch:
                md = dict(c.metadata or {})
                src = str(md.get("source") or md.get("filepath") or "unknown")
                page = int(md.get("page") or 1)
                cid = _make_id(src, page, c.index)
                # chroma 的 metadata 不支持 list 等复杂类型，统一转 str
                safe_md = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in md.items()}
                ids.append(cid)
                texts.append(c.text)
                metadatas.append(safe_md)

            if embeddings is None:
                if self.embedding_model is None:
                    raise RuntimeError("未提供 embeddings 且未配置 embedding 模型")
                embeds = self.embedding_model.encode(texts, batch_size=batch_size).tolist()
            else:
                embeds = [list(map(float, e)) for e in embeddings[start : start + batch_size]]

            with Timer(f"chroma upsert x{len(batch)}"):
                self.collection.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=embeds,
                    metadatas=metadatas,
                )
            total += len(batch)
        logger.info("写入完成：%d 条", total)
        return total

    # ------------------------------------------------------------------
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Hit]:
        """检索 Top-K 命中文档。

        Args:
            query_text: 查询文本。
            top_k: 返回条数。
            where: metadata 过滤条件。
            score_threshold: 相似度下限（cosine 距离下表示 1-distance >= threshold）。

        Returns:
            Hit 列表（按相似度从高到低）。
        """
        if self.collection.count() == 0:
            return []

        if self.embedding_model is None:
            raise RuntimeError("当前未配置 embedding 模型，无法进行语义检索")

        q_vec = self.embedding_model.encode([query_text], batch_size=1).tolist()

        kwargs: Dict[str, Any] = {
            "query_embeddings": q_vec,
            "n_results": max(1, int(top_k)),
        }
        if where:
            kwargs["where"] = where

        with Timer("chroma query"):
            res = self.collection.query(**kwargs)

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        hits: List[Hit] = []
        for i, _id in enumerate(ids):
            # cosine 距离下转相似度
            dist = float(dists[i]) if i < len(dists) else 0.0
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
        return hits

    # ------------------------------------------------------------------
    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        """按 metadata 条件删除文档，返回删除条数。"""
        if not where:
            raise ValueError("where 条件不能为空，避免误删")
        before = self.collection.count()
        self.collection.delete(where=where)
        after = self.collection.count()
        deleted = before - after
        logger.info("按 %s 删除 %d 条", where, deleted)
        return deleted

    def delete_by_ids(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        before = self.collection.count()
        self.collection.delete(ids=list(ids))
        after = self.collection.count()
        return before - after

    def list_sources(self) -> List[Dict[str, Any]]:
        """列出所有来源文档及其分块数（基于 metadata.source 聚合）。"""
        if self.collection.count() == 0:
            return []
        # 取全量 metadata（注意：数据量大时建议另存索引表）
        data = self.collection.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        agg: Dict[str, Dict[str, Any]] = {}
        for m in metas:
            src = str(m.get("source") or m.get("filepath") or "unknown")
            entry = agg.setdefault(src, {"source": src, "chunks": 0})
            entry["chunks"] += 1
        return sorted(agg.values(), key=lambda x: -x["chunks"])

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """清空当前集合（谨慎使用）。"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("集合 %s 已重置", self.collection_name)