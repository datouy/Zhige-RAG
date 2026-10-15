"""知识图谱路由（KG / GraphRAG）。

端点：
- GET  /api/v1/kg/stats
- GET  /api/v1/kg/entities
- GET  /api/v1/kg/relations
- POST /api/v1/kg/search
- POST /api/v1/kg/query        （带 cypher 注入防护 — P2.1）
- POST /api/v1/kg/extract
- POST /api/v1/kg/build
- POST /api/v1/kg/graph_rag

P2.1 安全要点
-------------
``SQLiteGraphStore.query_cypher`` 仅识别 ``MATCH ... RETURN`` 的极小子集。
我们**显式**校验客户端传入的 cypher：

- 必须以 ``MATCH`` 开头（忽略大小写与前导空白）。
- 必须包含 ``RETURN``。
- 不允许出现 ``WRITE / DELETE / DETACH / MERGE / CREATE / DROP / SET``
  等写操作关键字（即便在大写 / 小写 / 注释中）。
- 长度上限 2 KiB。

不匹配以上规则直接 400，避免触发 ``SQLiteGraphStore`` 把未识别查询
当作空结果默默吞掉的语义陷阱。
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api.deps import get_kg_extractor, get_runtime_config, get_runtime_pipeline
from src.db.models import User
from src.factories import TenantAwareFactory
from src.middleware.auth import get_current_user
from src.kg import GraphRAG, GraphRetriever
from src.kg.cypher_guard import (
    CypherValidationError,
    ValidatedCypher,
    validate_readonly_cypher,
)
from src.utils import get_logger, resolve_path

logger = get_logger("api.routes.kg")


def _ensure_within(base: Path, target: Path) -> None:
    """确保 target 落在 base 目录内，防止任意目录被读取/索引。

    与 chat.py 中的同名 helper 语义一致：用 ``Path.is_relative_to`` 做
    路径分隔符级比较，而不是字符串前缀（``data/raw2`` 不能冒充 ``data/raw``）。
    """
    base_resolved = base.resolve()
    target_resolved = target.resolve()
    if not target_resolved.is_relative_to(base_resolved):
        raise HTTPException(
            status_code=403,
            detail=f"仅允许访问数据目录内的路径（{base_resolved}）",
        )

router = APIRouter()


# ----------------------------------------------------------------------
# P2.1 兼容垫片：``_validate_cypher`` 保留以兼容既有测试
# ----------------------------------------------------------------------
def _validate_cypher(cypher: str) -> Optional[str]:
    """Cypher 校验的兼容包装 — 返回 ``None`` 表示通过，否则返回错误信息。

    真正的校验逻辑位于 :mod:`src.kg.cypher_guard`，这里只做异常到字符串的
    转换，便于旧测试断言。
    """
    try:
        validate_readonly_cypher(cypher or "", max_limit=200)
        return None
    except CypherValidationError as exc:
        return str(exc)


def _get_user_kg_store(user_id: str):
    cfg = get_runtime_config()
    from src.factories import TenantAwareFactory

    return TenantAwareFactory.get_kg_store(user_id, cfg)


def _get_user_kg_retriever(user_id: str):
    return GraphRetriever(_get_user_kg_store(user_id))


@router.get("/api/v1/kg/stats", tags=["知识图谱"])
async def kg_stats(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    user: User = current_user
    try:
        store = _get_user_kg_store(user.id)
        return store.count()
    except Exception as exc:
        logger.error("kg_stats 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/v1/kg/entities", tags=["知识图谱"])
async def kg_list_entities(
    type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    user: User = current_user
    try:
        store = _get_user_kg_store(user.id)
        entities: List[Any] = []
        if search:
            entities = store.find_entities_by_name(search, fuzzy=True, limit=limit)
        else:
            count = store.count().get("entities", 0)
            conn = sqlite3.connect(store.db_path)
            cur = conn.cursor()
            query = "SELECT name, type, description, aliases, attributes FROM entities"
            params: List[Any] = []
            if type:
                query += " WHERE type = ?"
                params.append(type)
            query += " LIMIT ?"
            params.append(limit)
            cur.execute(query, params)
            from src.kg.schema import Entity

            for row in cur.fetchall():
                entities.append(
                    Entity(
                        name=row[0],
                        type=row[1] or "Other",
                        description=row[2] or "",
                        aliases=json.loads(row[3] or "[]"),
                        attributes=json.loads(row[4] or "{}"),
                    )
                )
            conn.close()
        return {
            "entities": [
                {"name": e.name, "type": e.type, "description": e.description, "aliases": e.aliases}
                for e in entities
            ],
            "total": len(entities),
        }
    except Exception as exc:
        logger.error("kg_list_entities 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/v1/kg/relations", tags=["知识图谱"])
async def kg_list_relations(
    source: Optional[str] = None,
    target: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    user: User = current_user
    try:
        store = _get_user_kg_store(user.id)
        conn = sqlite3.connect(store.db_path)
        cur = conn.cursor()
        query = (
            "SELECT source, target, type, description, weight, attributes "
            "FROM relations WHERE 1=1"
        )
        params: List[Any] = []
        if source:
            query += " AND source = ?"
            params.append(source)
        if target:
            query += " AND target = ?"
            params.append(target)
        if type:
            query += " AND type = ?"
            params.append(type)
        query += " LIMIT ?"
        params.append(limit)
        cur.execute(query, params)
        relations: List[Dict[str, Any]] = []
        for row in cur.fetchall():
            relations.append(
                {
                    "source": row[0],
                    "target": row[1],
                    "type": row[2],
                    "description": row[3],
                    "weight": float(row[4] or 1.0),
                    "attributes": json.loads(row[5] or "{}"),
                }
            )
        conn.close()
        return {"relations": relations, "total": len(relations)}
    except Exception as exc:
        logger.error("kg_list_relations 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class KGSearchBody(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    hops: int = Field(2, ge=0, le=5)


@router.post("/api/v1/kg/search", tags=["知识图谱"])
async def kg_search(
    body: KGSearchBody,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    user: User = current_user
    try:
        retriever = _get_user_kg_retriever(user.id)
        result = retriever.search(
            query=body.query, top_k_entities=body.top_k, hops=body.hops
        )
        return {
            "query": body.query,
            "entities": [
                {"name": e.name, "type": e.type, "description": e.description}
                for e in result.get("entities", [])
            ],
            "relations": [
                {"source": r.source, "target": r.target, "type": r.type, "description": r.description}
                for r in result.get("relations", [])
            ],
            "triples": [
                {"subject": t.subject, "predicate": t.predicate, "object": t.object}
                for t in result.get("triples", [])
            ],
            "subgraph": result.get("subgraph", {}),
        }
    except Exception as exc:
        logger.error("kg_search 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class KGQueryBody(BaseModel):
    cypher: str = Field(..., min_length=1)
    params: Optional[Dict[str, Any]] = None


@router.post("/api/v1/kg/query", tags=["知识图谱"])
async def kg_query(
    body: KGQueryBody,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """执行 Cypher 查询（仅允许 ``MATCH ... RETURN`` 形式的只读查询）。

    P2.1：委托 ``src.kg.cypher_guard`` 完成白名单 + 黑名单 + LIMIT 注入，
    校验失败返回 400。
    """
    user: User = current_user

    try:
        validated = validate_readonly_cypher(body.cypher, max_limit=200)
    except CypherValidationError as exc:
        raise HTTPException(status_code=400, detail=f"cypher rejected: {exc}")

    try:
        store = _get_user_kg_store(user.id)
        results = store.query_cypher(validated.sanitized, body.params)
        # 防御性：再截一刀（即便 SQLite 后端可能没 LIMIT）
        if isinstance(results, list) and len(results) > validated.max_limit:
            results = results[: validated.max_limit]
        return {
            "cypher": body.cypher,
            "results": results,
            "limit": validated.max_limit,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("kg_query 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class KGExtractBody(BaseModel):
    text: str = Field(..., min_length=1)
    source_doc: str = ""
    persist: bool = True


@router.post("/api/v1/kg/extract", tags=["知识图谱"])
async def kg_extract(
    body: KGExtractBody,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    user: User = current_user
    try:
        extractor = get_kg_extractor()
        if extractor is None:
            raise HTTPException(status_code=503, detail="LLM 未就绪，无法抽取")
        entities, relations = extractor.extract(text=body.text, source_doc=body.source_doc)
        n_e = n_r = 0
        if body.persist:
            store = _get_user_kg_store(user.id)
            if entities:
                n_e = store.upsert_entities(entities)
            if relations:
                n_r = store.upsert_relations(relations)
        return {
            "entities": [
                {"name": e.name, "type": e.type, "description": e.description}
                for e in entities
            ],
            "relations": [
                {"source": r.source, "target": r.target, "type": r.type, "description": r.description}
                for r in relations
            ],
            "persisted": {"entities": n_e, "relations": n_r},
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("kg_extract 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class KGBuildBody(BaseModel):
    dir_path: str
    limit: Optional[int] = None


@router.post("/api/v1/kg/build", tags=["知识图谱"])
async def kg_build(
    body: KGBuildBody,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    user: User = current_user
    try:
        from scripts.build_kg import build as build_fn

        target = resolve_path(body.dir_path)
        # 此前只调用 resolve_path，未做边界校验：任何登录用户都能让服务端
        # 去索引任意目录（/etc、共享盘、其他租户的 data/raw 等）。
        # 限定在项目 data 目录内。
        _ensure_within(resolve_path("data"), target)
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"目录不存在: {body.dir_path}")
        try:
            # P5 审计修复：复用全局单例 pipeline（严禁 from_config 重建——
            # 每请求重新加载 GB 级模型，并发几次即 OOM），LLM 供所有租户共享。
            pipeline = get_runtime_pipeline(user_id=user.id)
            pipeline.ensure_llm()
            llm = pipeline.llm
        except Exception:
            llm = None
        if llm is None:
            return {"status": "skipped", "reason": "LLM 未就绪，仅构建空图谱"}
        # 关键：显式传入用户专属 KG store。之前 build() 内部按配置创建
        # 全局 data/kg.db，租户抽取结果全部写进了全局库。
        kg = _get_user_kg_store(user.id)
        summary = await run_in_threadpool(
            build_fn, input_dir=str(target), limit=body.limit, llm=llm, store=kg
        )
        return summary
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("kg_build 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class GraphRAGBody(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(4, ge=1, le=20)
    graph_hops: int = Field(2, ge=0, le=5)


@router.post("/api/v1/kg/graph_rag", tags=["知识图谱"])
async def kg_graph_rag(
    body: GraphRAGBody,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    user: User = current_user
    try:
        cfg_full = get_runtime_config()
        vs = TenantAwareFactory.get_chroma_store(user.id, cfg_full)
        kg_store = _get_user_kg_store(user.id)
        kg_retriever = GraphRetriever(kg_store)
        try:
            # P5 审计修复：复用全局单例 pipeline，不再每请求 from_config 重建
            pipeline = get_runtime_pipeline(user_id=user.id)
            pipeline.ensure_llm()
            llm = pipeline.llm
        except Exception:
            llm = None
        # 查询路径只读图谱，不做 KG 抽取（抽取属入库/kg build 阶段）
        graph_rag = GraphRAG(
            vector_store=vs,
            kg_store=kg_store,
            extractor=None,
            retriever=kg_retriever,
            llm=llm,
        )
        if graph_rag.llm is None:
            raise HTTPException(status_code=503, detail="LLM 未就绪，无法 GraphRAG 问答")
        result = await run_in_threadpool(
            graph_rag.query,
            question=body.question,
            top_k=body.top_k,
            graph_hops=body.graph_hops,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("kg_graph_rag 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))