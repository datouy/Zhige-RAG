"""PostgreSQL + pgvector 向量后端（企业多副本部署）。

与 Chroma 后端共用 :class:`BaseVectorStore` 接口。所有 metadata ACL 条件都在
SQL 侧执行，向量召回与关键词召回都不能绕过安全过滤。
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from .utils import get_logger
from .vector_store import BaseVectorStore, Hit, VectorStoreHealthError

logger = get_logger("vector_store_pgvector")


def _safe_identifier(value: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", str(value or "kb"))[:40] or "kb"
    suffix = hashlib.sha256(str(value or "kb").encode("utf-8")).hexdigest()[:8]
    return f"{base}_{suffix}"


class PgVectorStore(BaseVectorStore):
    """Postgres 集中式向量库，可由多个 API 副本并发读写。"""

    def __init__(
        self,
        dsn: str = "",
        collection_name: str = "chinese_rag_kb",
        embedding_model: Any = None,
        distance_fn: str = "cosine",
        hybrid_enabled: bool = True,
        hybrid_fetch_k: int = 20,
        hybrid_bm25_weight: float = 0.5,
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
        self.table = _safe_identifier(collection_name)
        self.embedding_model = embedding_model
        self.distance_fn = distance_fn
        self.hybrid_enabled = bool(hybrid_enabled)
        self.hybrid_fetch_k = max(1, int(hybrid_fetch_k))
        self.hybrid_bm25_weight = max(0.0, float(hybrid_bm25_weight))
        self._dim: Optional[int] = getattr(embedding_model, "dim", None) or None
        self._init_table()

    def _connect(self):  # type: ignore[no-untyped-def]
        return self._psycopg.connect(self.dsn)

    def _init_table(self) -> None:
        dim = int(self._dim or 0)
        if not dim and self.embedding_model is not None:
            try:
                dim = len(self._embed(["dimension probe"], batch_size=1)[0])
                self._dim = dim
            except Exception:
                dim = 0
        dim = dim or 384
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} "
                "(id TEXT PRIMARY KEY, text TEXT NOT NULL, "
                f"embedding vector({dim}), metadata JSONB NOT NULL DEFAULT '{{}}')"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_metadata_gin "
                f"ON {self.table} USING GIN (metadata)"
            )
            try:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_hnsw "
                    f"ON {self.table} USING hnsw (embedding vector_cosine_ops)"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("创建 pgvector HNSW 索引失败（查询仍可用）: %s", exc)
            conn.commit()

    def _embed(self, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
        if self.embedding_model is None:
            raise RuntimeError("未提供 embeddings 且未配置 embedding 模型")
        values = self.embedding_model.encode(list(texts), batch_size=batch_size)
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [list(map(float, value)) for value in values]

    @staticmethod
    def _chunk_id(metadata: Dict[str, Any], index: Any) -> str:
        source = str(metadata.get("source") or metadata.get("filepath") or "unknown")
        page = int(metadata.get("page") or 1)
        return hashlib.md5(f"{source}|{page}|{index}".encode("utf-8")).hexdigest()

    def _rows_for_chunks(self, chunks: Sequence[Any], embeddings=None, batch_size: int = 128):
        items = list(chunks or [])
        if not items:
            return [], []
        if embeddings is None:
            vecs = self._embed(
                [str(getattr(item, "text", "") or "") for item in items],
                batch_size=batch_size,
            )
        else:
            vecs = [list(map(float, value)) for value in embeddings]
        rows = []
        sources = set()
        for item, vector in zip(items, vecs):
            metadata = dict(getattr(item, "metadata", None) or {})
            source = str(metadata.get("source") or metadata.get("filepath") or "unknown")
            sources.add(source)
            chunk_id = self._chunk_id(metadata, getattr(item, "index", 0))
            rows.append(
                (
                    chunk_id,
                    str(getattr(item, "text", "") or ""),
                    vector,
                    json.dumps(metadata, ensure_ascii=False),
                    source,
                )
            )
        return rows, sorted(sources)

    def _insert_rows(self, cur, rows) -> int:
        if not rows:
            return 0
        cur.executemany(
            f"INSERT INTO {self.table} (id, text, embedding, metadata) "
            "VALUES (%s,%s,%s::vector,%s::jsonb) "
            "ON CONFLICT (id) DO UPDATE SET "
            "text=EXCLUDED.text, embedding=EXCLUDED.embedding, metadata=EXCLUDED.metadata",
            [row[:4] for row in rows],
        )
        return len(rows)

    def add_chunks(self, chunks: Sequence[Any], embeddings=None, batch_size: int = 128) -> int:
        rows, _sources = self._rows_for_chunks(chunks, embeddings, batch_size)
        with self._connect() as conn, conn.cursor() as cur:
            count = self._insert_rows(cur, rows)
            conn.commit()
        return count

    def replace_chunks_for_sources(
        self,
        chunks: Sequence[Any],
        sources: Sequence[str],
        on_commit: Optional[Any] = None,
    ) -> int:
        source_set = sorted({str(source) for source in sources if source})
        if not source_set:
            raise ValueError("replace_chunks_for_sources 需要至少一个 source")
        rows, _ = self._rows_for_chunks(chunks)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table} WHERE metadata->>'source' = ANY(%s)",
                (source_set,),
            )
            count = self._insert_rows(cur, rows)
            conn.commit()
        if on_commit is not None:
            on_commit()
        return count

    def _where_sql(
        self,
        where: Optional[Dict[str, Any]],
        params: List[Any],
    ) -> str:
        if not where:
            return ""
        clauses: List[str] = []
        for key, cond in where.items():
            if key == "$and":
                nested = [self._where_sql(item, params) for item in (cond or [])]
                nested = [item for item in nested if item]
                if nested:
                    clauses.append("(" + " AND ".join(nested) + ")")
                continue
            if key == "$or":
                nested = [self._where_sql(item, params) for item in (cond or [])]
                nested = [item for item in nested if item]
                if nested:
                    clauses.append("(" + " OR ".join(nested) + ")")
                else:
                    clauses.append("FALSE")
                continue
            if key.startswith("$"):
                clauses.append("FALSE")
                continue
            field = f"metadata->>'{key.replace(chr(39), chr(39) * 2)}'"
            if isinstance(cond, dict):
                if len(cond) != 1:
                    clauses.append("FALSE")
                    continue
                op, operand = next(iter(cond.items()))
                if op == "$eq":
                    clauses.append(f"{field} = %s")
                    params.append(str(operand))
                elif op == "$ne":
                    clauses.append(f"({field} IS NULL OR {field} <> %s)")
                    params.append(str(operand))
                elif op == "$in":
                    values = [str(value) for value in (operand or [])]
                    if values:
                        clauses.append(f"{field} = ANY(%s)")
                        params.append(values)
                    else:
                        clauses.append("FALSE")
                elif op == "$nin":
                    values = [str(value) for value in (operand or [])]
                    clauses.append(f"({field} IS NULL OR NOT ({field} = ANY(%s)))")
                    params.append(values)
                else:
                    clauses.append("FALSE")
            else:
                clauses.append(f"{field} = %s")
                params.append(str(cond))
        return " AND ".join(clauses) if clauses else "FALSE"

    @staticmethod
    def _hit(row) -> Hit:
        chunk_id, text, metadata, score = row
        try:
            meta = json.loads(metadata) if isinstance(metadata, str) else dict(metadata or {})
        except Exception:
            meta = {}
        return Hit(id=str(chunk_id), text=str(text or ""), score=float(score), metadata=meta)

    def _vector_query(self, query_vector: List[float], limit: int, where) -> List[Hit]:
        params: List[Any] = []
        where_sql = self._where_sql(where, params)
        sql = (
            f"SELECT id, text, metadata, 1.0 - (embedding <=> %s::vector) AS score "
            f"FROM {self.table}"
        )
        if where_sql:
            sql += " WHERE " + where_sql
        sql += " ORDER BY embedding <=> %s::vector ASC LIMIT %s"
        vector_param = list(map(float, query_vector))
        query_params = [vector_param, *params, vector_param, max(1, int(limit))]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, query_params)
            return [self._hit(row) for row in cur.fetchall()]

    @staticmethod
    def _query_terms(query: str) -> List[str]:
        text = str(query or "").strip()
        if not text:
            return []
        terms: List[str] = []
        try:
            import jieba  # type: ignore

            candidates = jieba.lcut_for_search(text)
        except Exception:
            candidates = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text)
        for term in candidates:
            value = str(term or "").strip()
            if len(value) >= 2 and value not in terms:
                terms.append(value)
            if len(terms) >= 8:
                break
        return terms

    def _keyword_query(self, query: str, limit: int, where) -> List[Hit]:
        terms = self._query_terms(query)
        if not terms:
            return []
        params: List[Any] = []
        where_sql = self._where_sql(where, params)
        pattern_clauses = []
        for term in terms:
            escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern_clauses.append("text ILIKE %s")
            params.append(f"%{escaped}%")
        sql = f"SELECT id, text, metadata, 0.0 AS score FROM {self.table} WHERE ("
        sql += " OR ".join(pattern_clauses) + ")"
        if where_sql:
            sql += " AND (" + where_sql + ")"
        sql += " LIMIT %s"
        params.append(max(1, int(limit)))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return [self._hit(row) for row in cur.fetchall()]

    @staticmethod
    def _rrf(vector_hits: List[Hit], keyword_hits: List[Hit], keyword_weight: float, limit: int) -> List[Hit]:
        if not keyword_hits:
            return vector_hits[:limit]
        merged: Dict[str, Dict[str, Any]] = {}
        for rank, hit in enumerate(vector_hits, start=1):
            merged[hit.id] = {"hit": hit, "score": 1.0 / (60 + rank)}
        for rank, hit in enumerate(keyword_hits, start=1):
            entry = merged.setdefault(hit.id, {"hit": hit, "score": 0.0})
            entry["score"] += max(0.0, keyword_weight) / (60 + rank)
            if len(hit.text) > len(entry["hit"].text):
                entry["hit"] = hit
        ceiling = (1.0 + max(0.0, keyword_weight)) / 61.0
        hits = []
        for entry in merged.values():
            hit = entry["hit"]
            hits.append(
                Hit(
                    id=hit.id,
                    text=hit.text,
                    score=max(0.0, min(1.0, float(entry["score"]) / ceiling)),
                    metadata=hit.metadata,
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Hit]:
        vector = self._embed([query_text], batch_size=1)[0]
        fetch_k = max(int(top_k), self.hybrid_fetch_k if self.hybrid_enabled else int(top_k))
        vector_hits = self._vector_query(vector, fetch_k, where)
        if self.hybrid_enabled:
            keyword_hits = self._keyword_query(query_text, fetch_k, where)
            hits = self._rrf(vector_hits, keyword_hits, self.hybrid_bm25_weight, int(top_k))
        else:
            hits = vector_hits[: int(top_k)]
        return [hit for hit in hits if hit.score >= float(score_threshold or 0.0)]

    def query_batch(self, query_texts, top_k: int = 5, where=None, score_threshold: float = 0.0):
        return [self.query(text, top_k, where, score_threshold) for text in query_texts]

    def get_by_source(self, source: str, where=None) -> List[Hit]:
        filters: List[Dict[str, Any]] = [{"source": {"$eq": str(source)}}]
        if where:
            filters.append(where)
        combined = {"$and": filters}
        params: List[Any] = []
        where_sql = self._where_sql(combined, params)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text, metadata, 0.0 AS score FROM {self.table} "
                f"WHERE {where_sql}",
                params,
            )
            hits = [self._hit(row) for row in cur.fetchall()]
        hits.sort(key=lambda item: int((item.metadata or {}).get("chunk_index") or 0))
        return hits

    def get_by_ids(self, ids: Sequence[str]) -> List[Hit]:
        values = [str(item) for item in (ids or []) if item]
        if not values:
            return []
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text, metadata, 0.0 AS score FROM {self.table} WHERE id = ANY(%s)",
                (values,),
            )
            return [self._hit(row) for row in cur.fetchall()]

    def update_chunk(self, chunk_id: str, text: str, embedding, metadata) -> bool:
        vector = list(map(float, embedding))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE {self.table} SET text=%s, embedding=%s::vector, metadata=%s::jsonb WHERE id=%s",
                (text, vector, json.dumps(metadata or {}, ensure_ascii=False), chunk_id),
            )
            updated = int(cur.rowcount or 0)
            conn.commit()
        return updated > 0

    def delete_by_ids(self, ids: Sequence[str]) -> int:
        values = [str(item) for item in (ids or []) if item]
        if not values:
            return 0
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE id = ANY(%s)", (values,))
            deleted = int(cur.rowcount or 0)
            conn.commit()
        return deleted

    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        if not where:
            raise ValueError("where 条件不能为空，避免误删")
        params: List[Any] = []
        where_sql = self._where_sql(where, params)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE {where_sql}", params)
            deleted = int(cur.rowcount or 0)
            conn.commit()
        return deleted

    def list_sources(self, where=None) -> List[Dict[str, Any]]:
        params: List[Any] = []
        where_sql = self._where_sql(where, params)
        sql = (
            f"SELECT COALESCE(metadata->>'source', metadata->>'filepath', 'unknown') AS source, "
            f"COUNT(*) FROM {self.table}"
        )
        if where_sql:
            sql += " WHERE " + where_sql
        sql += " GROUP BY source ORDER BY COUNT(*) DESC"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return [{"source": row[0], "chunks": int(row[1])} for row in cur.fetchall()]

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            row = cur.fetchone()
            return int(row[0] if row else 0)

    def reset(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.table}")
            conn.commit()

    def delete_collection(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.table}")
                conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("删除 pgvector 表 %s 失败: %s", self.table, exc)
            return False

    def health_check(self, query_probe: bool = True) -> Dict[str, Any]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            count = int(cur.fetchone()[0])
        if query_probe and count:
            cur_vector = self._embed(["health probe"], batch_size=1)[0]
            self._vector_query(cur_vector, 1, None)
        return {
            "backend": "pgvector",
            "collection": self.collection_name,
            "collections": 1,
            "chunks": count,
            "query_probe": bool(query_probe and count),
            "status": "healthy",
        }


__all__ = ["PgVectorStore"]
