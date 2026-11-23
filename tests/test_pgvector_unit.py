"""pgvector 后端的 SQL 安全过滤与 RRF 融合单元测试。"""
from __future__ import annotations

from src.vector_store import Hit
from src.vector_store_pgvector import PgVectorStore


def test_where_sql_applies_acl_and_status_to_all_paths():
    store = PgVectorStore.__new__(PgVectorStore)
    params = []
    sql = store._where_sql(
        {
            "$and": [
                {"doc_status": "active"},
                {"acl": {"$in": ["*", "研发部"]}},
            ]
        },
        params,
    )
    assert "metadata->>'doc_status'" in sql
    assert "metadata->>'acl'" in sql
    assert params == ["active", ["*", "研发部"]]


def test_unknown_filter_fails_closed():
    store = PgVectorStore.__new__(PgVectorStore)
    params = []
    sql = store._where_sql({"$unsupported": ["x"]}, params)
    assert sql == "FALSE"


def test_rrf_fuses_vector_and_keyword_hits():
    vector = [
        Hit("a", "向量命中", 0.9, {"source": "a.md"}),
        Hit("b", "向量次命中", 0.8, {"source": "b.md"}),
    ]
    keyword = [
        Hit("b", "关键词命中完整正文", 0.0, {"source": "b.md"}),
        Hit("c", "仅关键词命中", 0.0, {"source": "c.md"}),
    ]
    fused = PgVectorStore._rrf(vector, keyword, 0.5, 3)
    assert [hit.id for hit in fused][0] == "b"
    assert {hit.id for hit in fused} == {"a", "b", "c"}
    assert all(0.0 <= hit.score <= 1.0 for hit in fused)
