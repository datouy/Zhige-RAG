"""FastAPI Web 服务 - 中文知识库 RAG 系统。

提供 REST API + WebSocket 流式问答，逐步替代 Streamlit 交互界面。

启动命令：
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

API 端点：
    GET  /api/health              - 健康检查
    GET  /api/documents           - 列出已入库文档
    GET  /api/documents/{doc_id}/chunks - 获取文档分块
    POST /api/search              - 检索
    POST /api/chat                - 同步问答（非流式）
    WS   /ws/chat                 - WebSocket 流式问答
    GET  /api/agent/tools         - 列出 Agent 可用工具
    POST /api/agent/chat          - 同步 Agent 调用
    WS   /ws/agent                - WebSocket 流式 Agent
    POST /api/ingest              - 上传文件并入库
    POST /api/ingest/dir          - 批量入库目录
    POST /api/eval/run            - 运行评估
    GET  /api/config              - 获取当前配置（脱敏）
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 项目根目录加入 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse

from src.document_loader import load_document, load_directory
from src.embeddings import EmbeddingModel
from src.llm import LocalLLM
from src.rag_pipeline import RAGPipeline
from src.reranker import BgeReranker
from src.text_splitter import ChineseTextSplitter, RecursiveTextSplitter
from src.utils import (
    apply_env_overrides,
    ensure_dir,
    get_logger,
    load_config,
    merge_dict,
    resolve_path,
)
from src.vector_store import ChromaStore
from src.kg import KGExtractor, GraphRAG, GraphRetriever, create_kg_store

logger = get_logger("api")

# =========================== 应用实例 ===========================
app = FastAPI(
    title="ChineseRAGKB API",
    description="中文知识库 RAG 系统的 FastAPI 接口",
    version="0.2.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 Pipeline 单例（懒加载）
_pipeline: Optional[RAGPipeline] = None
_embedding: Optional[EmbeddingModel] = None
_vector_store: Optional[ChromaStore] = None
_agent: Optional[Any] = None
_agent_tools: Optional[Any] = None
_kg_store = None
_kg_retriever: Optional[GraphRetriever] = None
_kg_extractor: Optional[KGExtractor] = None
_graph_rag: Optional[GraphRAG] = None


def get_pipeline() -> RAGPipeline:
    """获取或创建全局 Pipeline 实例（懒加载）。"""
    global _pipeline
    if _pipeline is None:
        cfg = load_config("config/config.yaml")
        cfg = apply_env_overrides(cfg)
        _pipeline = RAGPipeline.from_config("config/config.yaml", overrides=cfg, lazy_llm=False)
    return _pipeline


def get_embedding() -> EmbeddingModel:
    """获取或创建全局 Embedding 实例。"""
    global _embedding
    if _embedding is None:
        cfg = load_config("config/config.yaml")
        cfg = apply_env_overrides(cfg)
        _embedding = EmbeddingModel(
            model_name=cfg["embedding"]["model_name"],
            device=cfg["embedding"].get("device", "auto"),
            batch_size=cfg["embedding"].get("batch_size", 32),
            max_seq_length=cfg["embedding"].get("max_seq_length", 512),
            normalize=cfg["embedding"].get("normalize_embeddings", True),
            cache_dir=cfg["embedding"].get("cache_dir"),
            local_files_only=cfg["embedding"].get("local_files_only", False),
        )
    return _embedding


def get_vector_store() -> ChromaStore:
    """获取或创建全局 VectorStore 实例。"""
    global _vector_store
    if _vector_store is None:
        cfg = load_config("config/config.yaml")
        cfg = apply_env_overrides(cfg)
        embed = get_embedding()
        _vector_store = ChromaStore(
            persist_directory=cfg["vector_store"]["persist_directory"],
            collection_name=cfg["vector_store"].get("collection_name", "chinese_rag_kb"),
            embedding_model=embed,
            distance_fn=cfg["vector_store"].get("distance_fn", "cosine"),
        )
    return _vector_store


def get_splitter(cfg: dict):
    """根据配置构建分块器。"""
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


# =========================== REST 端点 ===========================
@app.get("/")
async def root():
    """重定向到前端页面。"""
    return RedirectResponse(url="/index.html")


@app.get("/api/health")
async def health_check():
    """健康检查。"""
    try:
        store = get_vector_store()
        return {
            "status": "ok",
            "chunks": store.count(),
            "sources": len(store.list_sources()),
        }
    except Exception as exc:
        logger.error("健康检查失败: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(exc)},
        )


@app.get("/api/documents")
async def list_documents():
    """列出已入库文档（按 source 聚合）。"""
    try:
        store = get_vector_store()
        sources = store.list_sources()
        return {"documents": sources, "total": len(sources), "total_chunks": store.count()}
    except Exception as exc:
        logger.error("列出文档失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str, limit: int = 50):
    """获取指定文档的分块内容。"""
    try:
        store = get_vector_store()
        # doc_id 作为 source 过滤
        result = store.collection.get(
            where={"source": {"$eq": doc_id}},
            include=["documents", "metadatas"],
            limit=limit,
        )
        chunks = []
        for i, (doc, meta) in enumerate(zip(
            result.get("documents", []), result.get("metadatas", [])
        )):
            chunks.append({
                "id": result["ids"][i] if "ids" in result and i < len(result["ids"]) else f"chunk_{i}",
                "text": doc,
                "metadata": meta,
            })
        return {"doc_id": doc_id, "chunks": chunks, "total": len(chunks)}
    except Exception as exc:
        logger.error("获取分块失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class SearchRequest:
    """搜索请求模型。"""
    def __init__(self, query: str, top_k: int = 4):
        self.query = query
        self.top_k = top_k


from pydantic import BaseModel, Field


class SearchBody(BaseModel):
    """POST /api/search 请求体。"""
    query: str
    top_k: int = 4


@app.post("/api/search")
async def search_documents(body: SearchBody):
    """语义检索文档。"""
    try:
        store = get_vector_store()
        hits = store.query(query_text=body.query, top_k=body.top_k)
        return {
            "query": body.query,
            "hits": [
                {
                    "id": h.id,
                    "text": h.text[:500],  # 限制长度
                    "score": h.score,
                    "metadata": h.metadata,
                }
                for h in hits
            ],
            "total": len(hits),
        }
    except Exception as exc:
        logger.error("检索失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class ChatRequest(BaseModel):
    """POST /api/chat 请求体（非流式）。"""
    query: str = Field(..., min_length=1, description="查询文本，不能为空")
    top_k: int = Field(4, ge=1, le=50, description="检索 Top-K")


@app.post("/api/chat")
async def chat_sync(body: ChatRequest):
    """同步问答（非流式，适合轻量客户端）。"""
    try:
        pipeline = get_pipeline()
        result = pipeline.answer(body.query, top_k=body.top_k)
        return {
            "answer": result.answer,
            "sources": result.sources,
            "timings": result.timings,
        }
    except Exception as exc:
        logger.error("同步问答失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ingest")
async def ingest_file(
    file: UploadFile,
    version: bool = False,
):
    """上传文件并入库。"""
    try:
        cfg = load_config("config/config.yaml")
        cfg = apply_env_overrides(cfg)

        # 保存上传文件到临时目录
        raw_dir = resolve_path(cfg.get("paths", {}).get("raw_docs", "data/raw"))
        ensure_dir(raw_dir)
        temp_path = raw_dir / file.filename

        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # 解析并分块
        splitter = get_splitter(cfg)
        docs = load_document(
            temp_path,
            pdf_engine=cfg["document_loader"].get("pdf_engine", "pdfplumber"),
            encoding=cfg["document_loader"].get("encoding", "utf-8"),
        )

        all_chunks = []
        for doc in docs:
            all_chunks.extend(splitter.split_text(doc.content, metadata=doc.metadata))

        # 入库
        store = get_vector_store()
        n = store.add_chunks(all_chunks)

        return {
            "filename": file.filename,
            "docs": len(docs),
            "chunks": len(all_chunks),
            "ingested": n,
        }
    except Exception as exc:
        logger.error("文件入库失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ingest/dir")
async def ingest_directory(dir_path: str = Form(...), recursive: bool = Form(True)):
    """批量入库指定目录。"""
    try:
        cfg = load_config("config/config.yaml")
        cfg = apply_env_overrides(cfg)

        target = resolve_path(dir_path)
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"目录不存在: {dir_path}")

        splitter = get_splitter(cfg)
        docs = load_directory(
            target,
            pdf_engine=cfg["document_loader"].get("pdf_engine", "pdfplumber"),
            recursive=recursive,
            encoding=cfg["document_loader"].get("encoding", "utf-8"),
        )

        all_chunks = []
        for doc in docs:
            all_chunks.extend(splitter.split_text(doc.content, metadata=doc.metadata))

        store = get_vector_store()
        n = store.add_chunks(all_chunks)

        return {
            "dir": str(target),
            "docs": len(docs),
            "chunks": len(all_chunks),
            "ingested": n,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("目录入库失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/eval/run")
async def run_evaluation():
    """运行评估（使用已配置的评估集）。"""
    try:
        from scripts.evaluate import load_eval_dataset, evaluate as run_eval

        cfg = load_config("config/config.yaml")
        dataset_path = resolve_path(
            cfg.get("evaluation", {}).get("dataset_path", "data/eval/eval_set.jsonl")
        )

        if not dataset_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"评估集不存在: {dataset_path}",
            )

        pipeline = get_pipeline()
        items = load_eval_dataset(dataset_path)
        summary = run_eval(pipeline, items)

        return summary
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("评估失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/config")
async def get_config():
    """获取当前配置（脱敏：隐藏模型路径等敏感信息）。"""
    try:
        cfg = load_config("config/config.yaml")
        # 脱敏处理
        safe_cfg = dict(cfg)
        # 移除可能的敏感字段
        sensitive_keys = ["cache_dir", "model_path"]
        for key in sensitive_keys:
            if key in safe_cfg:
                safe_cfg[key] = "***"
        return safe_cfg
    except Exception as exc:
        logger.error("获取配置失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# =========================== Agent 端点 ===========================
def get_agent_tools():
    """获取（或创建）内置工具集。"""
    global _agent_tools
    if _agent_tools is None:
        from src.agent import BuiltinTools

        vs = get_vector_store()
        _agent_tools = BuiltinTools.create(vector_store=vs)
    return _agent_tools


def get_agent():
    """获取（或创建）ReActAgent 实例。"""
    global _agent
    if _agent is None:
        from src.agent import ReActAgent

        pipeline = get_pipeline()
        try:
            pipeline.ensure_llm()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ensure_llm 失败：%s", exc)
        llm = getattr(pipeline, "llm", None)
        if llm is None:
            raise RuntimeError("Pipeline 未加载 LLM，无法启动 Agent")
        cfg = load_config("config/config.yaml")
        cfg = apply_env_overrides(cfg)
        max_steps = int(cfg.get("agent", {}).get("max_steps", 5))
        tools = get_agent_tools()
        _agent = ReActAgent(llm=llm, tools=tools, max_steps=max_steps)
    return _agent


@app.get("/api/agent/tools")
async def list_agent_tools():
    """列出 Agent 可用工具（名称 + 描述 + 参数 schema）。"""
    try:
        tools = get_agent_tools()
        items = []
        for t in tools.list_tools():
            items.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            )
        return {"tools": items, "total": len(items)}
    except Exception as exc:
        logger.error("获取工具列表失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class AgentChatRequest(BaseModel):
    """POST /api/agent/chat 请求体。"""

    query: str = Field(..., min_length=1, description="用户问题")
    max_steps: Optional[int] = Field(None, ge=1, le=20, description="覆盖默认 max_steps")


@app.post("/api/agent/chat")
async def agent_chat_sync(body: AgentChatRequest):
    """同步 Agent 调用：收集完整事件流后一次性返回。"""
    try:
        agent = get_agent()
        if body.max_steps is not None and body.max_steps != agent.max_steps:
            # 允许按请求临时调整 max_steps（不污染全局实例）
            from src.agent.react_agent import ReActAgent

            agent = ReActAgent(llm=agent.llm, tools=agent.tools, max_steps=body.max_steps)
        events = []
        steps = 0
        tools_used: List[str] = []
        answer = ""
        truncated = False
        for ev in agent.run(body.query, stream=False):
            et = ev.get("event")
            data = ev.get("data")
            events.append({"event": et, "data": data})
            if et == "done":
                steps = int((data or {}).get("steps", 0))
                tools_used = list((data or {}).get("tools_used", []))
                answer = (data or {}).get("answer", "")
                truncated = bool((data or {}).get("truncated", False))
            elif et == "error":
                return JSONResponse(
                    status_code=500,
                    content={"error": str(data), "events": events},
                )
        return {
            "answer": answer,
            "steps": steps,
            "tools_used": tools_used,
            "truncated": truncated,
            "events": events,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Agent 调用失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket):
    """WebSocket 流式 Agent 调用。"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            max_steps = data.get("max_steps")
            if not query:
                await websocket.send_json({"event": "error", "data": "query is required"})
                continue
            try:
                agent = get_agent()
                if isinstance(max_steps, int) and max_steps > 0 and max_steps != agent.max_steps:
                    from src.agent.react_agent import ReActAgent

                    agent = ReActAgent(llm=agent.llm, tools=agent.tools, max_steps=max_steps)
                for ev in agent.run(query, stream=True):
                    await websocket.send_json(ev)
            except Exception as exc:
                logger.error("Agent WebSocket 处理失败: %s", exc)
                await websocket.send_json({"event": "error", "data": str(exc)})
    except WebSocketDisconnect:
        logger.info("Agent WebSocket 客户端断开")
    except Exception as exc:
        logger.error("Agent WebSocket 异常: %s", exc)


# =========================== WebSocket 流式问答 ===========================
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket 流式问答。"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            top_k = data.get("top_k", 4)

            if not query:
                await websocket.send_json({"event": "error", "data": "query is required"})
                continue

            try:
                pipeline = get_pipeline()
                async for event in pipeline.astream_answer(query, top_k=top_k):
                    await websocket.send_json(event)
            except Exception as exc:
                logger.error("WebSocket 流式生成失败: %s", exc)
                await websocket.send_json({"event": "error", "data": str(exc)})
    except WebSocketDisconnect:
        logger.info("WebSocket 客户端断开")
    except Exception as exc:
        logger.error("WebSocket 异常: %s", exc)


# =========================== 知识图谱端点 ===========================
def _load_kg_config() -> Dict[str, Any]:
    cfg = load_config("config/config.yaml")
    cfg = apply_env_overrides(cfg)
    return cfg.get("knowledge_graph", {}) or {}


def get_kg_store():
    """获取（或创建）KG 存储实例。"""
    global _kg_store
    if _kg_store is None:
        kg_cfg = _load_kg_config()
        if not kg_cfg.get("enabled", True):
            kg_cfg = {**kg_cfg, "enabled": True}
        _kg_store = create_kg_store(kg_cfg)
    return _kg_store


def get_kg_retriever():
    """获取（或创建）图谱检索器。"""
    global _kg_retriever
    if _kg_retriever is None:
        _kg_retriever = GraphRetriever(get_kg_store())
    return _kg_retriever


def get_kg_extractor():
    """获取（或创建）KG 抽取器（需要 LLM）。"""
    global _kg_extractor
    if _kg_extractor is None:
        try:
            pipeline = get_pipeline()
            pipeline.ensure_llm()
            llm = pipeline.llm
        except Exception as exc:
            logger.warning("LLM 不可用，KG 抽取器将不可用：%s", exc)
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


@app.get("/api/kg/stats")
async def kg_stats():
    """实体/关系数量。"""
    try:
        store = get_kg_store()
        return store.count()
    except Exception as exc:
        logger.error("kg_stats 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/kg/entities")
async def kg_list_entities(type: Optional[str] = None, search: Optional[str] = None, limit: int = 50):
    """列出实体。"""
    try:
        store = get_kg_store()
        entities = []
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
            import json as _json
            for row in cur.fetchall():
                entities.append(
                    Entity(
                        name=row[0],
                        type=row[1] or "Other",
                        description=row[2] or "",
                        aliases=_json.loads(row[3] or "[]"),
                        attributes=_json.loads(row[4] or "{}"),
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


@app.get("/api/kg/relations")
async def kg_list_relations(
    source: Optional[str] = None,
    target: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = 100,
):
    """列出关系。"""
    try:
        store = get_kg_store()
        relations = []
        conn = sqlite3.connect(store.db_path)
        cur = conn.cursor()
        query = "SELECT source, target, type, description, weight, attributes FROM relations WHERE 1=1"
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


@app.post("/api/kg/search")
async def kg_search(body: KGSearchBody):
    """图谱检索。"""
    try:
        retriever = get_kg_retriever()
        result = retriever.search(query=body.query, top_k_entities=body.top_k, hops=body.hops)
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


@app.post("/api/kg/query")
async def kg_query(body: KGQueryBody):
    """执行 Cypher 查询。"""
    try:
        store = get_kg_store()
        return {"cypher": body.cypher, "results": store.query_cypher(body.cypher, body.params)}
    except Exception as exc:
        logger.error("kg_query 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class KGExtractBody(BaseModel):
    text: str = Field(..., min_length=1)
    source_doc: str = ""
    persist: bool = True


@app.post("/api/kg/extract")
async def kg_extract(body: KGExtractBody):
    """抽取并入库。"""
    try:
        extractor = get_kg_extractor()
        if extractor is None:
            raise HTTPException(status_code=503, detail="LLM 未就绪，无法抽取")
        entities, relations = extractor.extract(text=body.text, source_doc=body.source_doc)
        n_e = n_r = 0
        if body.persist:
            store = get_kg_store()
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


@app.post("/api/kg/build")
async def kg_build(body: KGBuildBody):
    """从目录构建图谱。"""
    try:
        from scripts.build_kg import build as build_fn
        target = resolve_path(body.dir_path)
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"目录不存在: {body.dir_path}")
        cfg = _load_kg_config()
        kg_cfg = cfg.get("extractor", {}) or {}
        try:
            pipeline = get_pipeline()
            pipeline.ensure_llm()
            llm = pipeline.llm
        except Exception:
            llm = None
        if llm is None:
            return {"status": "skipped", "reason": "LLM 未就绪，仅构建空图谱"}
        summary = build_fn(
            input_dir=str(target),
            limit=body.limit,
            llm=llm,
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


@app.post("/api/kg/graph_rag")
async def kg_graph_rag(body: GraphRAGBody):
    """GraphRAG 问答。"""
    try:
        graph_rag = get_graph_rag()
        if graph_rag is None or graph_rag.llm is None:
            raise HTTPException(status_code=503, detail="LLM 未就绪，无法 GraphRAG 问答")
        result = graph_rag.query(question=body.question, top_k=body.top_k, graph_hops=body.graph_hops)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("kg_graph_rag 失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# =========================== 静态文件 ===========================
# 挂载前端静态文件
web_static_dir = ROOT / "ui" / "web"
if web_static_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_static_dir), html=True), name="static")


# =========================== 启动入口 ===========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
