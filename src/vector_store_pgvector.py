"""pgvector 向量库后端（P1-4 可选项）。

对应 RAGFlow/Dify 的 pgvector 部署形态：数据进 Postgres，运维统一备份/扩容。
未安装 ``psycopg`` / ``pgvector`` 或无 Postgres 时，本模块 import 即抛错，
由 :func:`src.vector_store.create_vector_store` 转为明确的健康检查错误，
绝不静默回落 chroma（避免数据分裂）。

表结构（ outputs 与 Chroma 字段对齐）::

    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS {table} (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        embedding vector({dim}),
        metadata JSONB NOT NULL DEFAULT '{}'
    );

检索：``ORDER BY embedding <=> %s::vector LIMIT k``（余弦距离），
metadata 过滤只支持 ``source = %s`` 等简单等值（复杂 where 由上层收敛）。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from .utils import get_logger
from .vector_store import BaseVectorStore, Hit, VectorStoreHealthError

logger = get_logger("vector_store_pgvector")


class PgVectorStore(BaseVectorStore):
    """Postgres + pgvector 实现的向量库。"""

    def __init__(
        self,
        dsn: str = "",
        collection_name: str = "chinese_rag_kb",
        embedding_model: Any = None,
        distance_fn: str = "cosine",
        **kwargs: Any,
    ) -> None:
        if not dsn:
            raise VectorStoreHealthError("pgvector 需要配置 pgvector_dsn")
        try:
            import psycopg  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreHealthError(
                f"未安装 psycopg：{exc}（pip install 'psycopg[binary]' pgvector）"
            ) from exc
        self._psycopg = psycopg
        self.dsn = dsn
        self.collection_name = collection_name
        self.table = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in collection_name)[:48] or "kb"
        self.embedding_model = embedding_model
        self.distance_fn = distance_fn
        self._dim: Optional[int] = getattr(embedding_model, "dim", None) or None
        self._init_table()

    def _connect(self):  # type: ignore[no-untyped-def]
        return self._psycopg.connect(self.dsn, autocommit=True)

    def _init_table(self) -> None:
        dim = int(self._dim or 0) or 384
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} "
                "(id TEXT PRIMARY KEY, text TEXT NOT NULL, "
                f"embedding vector({dim}), metadata JSONB NOT NULL DEFAULT '{{}}')"
            )

    # -- BaseVectorStore 接口 --
    def add_chunks(self, chunks: Sequence[Any], embeddings: Optional[Sequence[Sequence[float]]] = None, batch_size: int = 128) -> int:
        import hashlib as _hl

        items = list(chunks or [])
        if not items:
            return 0
        if embeddings is None:
            if self.embedding_model is None:
                raise RuntimeError("未提供 embeddings 且未配置 embedding 模型")
            texts = [str(getattr(c, "text", "") or "") for c in items]
            vecs = self.embedding_model.encode(texts, batch_size=batch_size).tolist()
        else:
            vecs = [list(map(float, v)) for v in embeddings]
        rows = []
        for c, v in zip(items, vecs):
            md = dict(getattr(c, "metadata", None) or {})
            src = str(md.get("source") or md.get("filepath") or "unknown")
            page = int(md.get("page") or 1)
            cid = _hl.md5(f"{src}|{page}|{getattr(c, 'index', 0)}".encode("utf-8")).hexdigest()
            if self._dim is None and v:
                self._dim = len(v)
            rows.append((cid, str(getattr(c, "text", "") or ""), v, json.dumps(md, ensure_ascii=False)))
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (id, text, embedding, metadata) VALUES (%s,%s,%s::vector,%s::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, embedding=EXCLUDED.embedding, metadata=EXCLUDED.metadata",
                rows,
            )
        return len(rows)

    def replace_chunks_for_sources(self, chunks: Sequence[Any], sources: Sequence[str], on_commit: Optional[Any] = None) -> int:
        sources = sorted({str(s) for s in sources if s})
        if not sources:
            raise ValueError("replace_chunks_for_sources 需要至少一个 source")
        self.delete_by_metadata({"source": {"$in": sources}})
        n = self.add_chunks(chunks)
        if on_commit is not None:
            on_commit()
        return n

    def query(self, query_text: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None, score_threshold: float = 0.0) -> List[Hit]:
        if self.embedding_model is None:
            raise RuntimeError("当前未配置 embedding 模型，无法进行语义检索")
        qv = self.embedding_model.encode([query_text], batch_size=1).tolist()[0]
        conds: List[str] = []
        params: List[Any] = [list(map(float, qv))]
        if where:
            src = (where.get("source") if isinstance(where, dict) else None)
            if isinstance(src, dict):
                if "$eq" in src:
                    conds.append("metadata->>'source' = %s")
                    params.append(str(src["$eq"]))
                elif "$in" in src and isinstance(src["$in"], (list, tuple)):
                    conds.append("metadata->>'source' = ANY(%s)")
                    params.append(list(map(str, src["$in"])))
        sql = f"SELECT id, text, metadata, embedding <=> %s::vector AS dist FROM {self.table}"
        params.append(list(map(float, qv)))
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += f" ORDER BY dist ASC LIMIT {max(1, int(top_k))}"
        hits: List[Hit] = []
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            for cid, text, md, dist in cur.fetchall():
                score = max(0.0, 1.0 - float(dist or 0.0))
                if score < float(score_threshold or 0.0):
                    continue
                try:
                    meta = json.loads(md) if isinstance(md, str) else dict(md or {})
                except Exception:
                    meta = {}
                hits.append(Hit(id=str(cid), text=str(text or ""), score=float(score), metadata=meta))
        return hits

    def delete_by_ids(self, ids: Sequence[str]) -> int:
        ids = list(ids or [])
        if not ids:
            return 0
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE id = ANY(%s)", (ids,))
            return int(cur.rowcount or 0)

    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        if not where:
            raise ValueError("where 条件不能为空，避免误删")
        src = (where.get("source") if isinstance(where, dict) else None)
        with self._connect() as conn, conn.cursor() as cur:
            if isinstance(src, dict) and "$in" in src:
                cur.execute(f"DELETE FROM {self.table} WHERE metadata->>'source' = ANY(%s)", (list(map(str, src["$in"])),))
            elif isinstance(src, dict) and "$eq" in src:
                cur.execute(f"DELETE FROM {self.table} WHERE metadata->>'source' = %s", (str(src["$eq"]),))
            elif isinstance(src, str):
                cur.execute(f"DELETE FROM {self.table} WHERE metadata->>'source' = %s", (src,))
            else:
                raise ValueError("pgvector 仅支持 source 等值/in 删除")
            return int(cur.rowcount or 0)

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            row = cur.fetchone()
            return int(row[0] if row else 0)


__all__ = ["PgVectorStore"]
