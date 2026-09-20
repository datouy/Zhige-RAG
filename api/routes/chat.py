"""问答 / 检索 / 文档管理路由。

端点：
- GET  /api/v1/documents
- GET  /api/v1/documents/{doc_id}/chunks
- POST /api/v1/search
- POST /api/v1/chat          （同步 RAG 问答）
- POST /api/v1/ingest        （单文件入库）
- POST /api/v1/ingest/dir    （目录批量入库）
- WS   /ws/chat              （兼容旧前端）

WS 聊天处理器通过 ``AsyncLLMExecutor`` 进行并发门控（P1.3），通过单例
Pipeline 复用 LLM / Embedding（P1.4）。
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps import (
    get_llm_executor,
    get_runtime_config,
    get_runtime_pipeline,
    get_splitter,
)
from api.quota import (
    CHAT_ENDPOINT,
    check_chunk_quota,
    check_query_quota,
    record_usage,
)
from src.data_quality import run_data_pipeline
from src.db.database import SessionLocal, get_db
from src.db.models import DocumentRecord, User
from src.document_loader import load_document, load_directory
from src.document_meta import enrich_directory, enrich_file
from src.factories import TenantAwareFactory
from src.memory import SESSION_MEMORY
from src.middleware.auth import get_current_user
from src.rag_pipeline import RAGPipeline
from src.utils import SUPPORTED_EXTENSIONS, ensure_dir, get_logger, resolve_path

logger = get_logger("api.routes.chat")

router = APIRouter()

# 上传流式写盘的缓冲大小（1 MiB）
_UPLOAD_CHUNK_SIZE = 1024 * 1024


def _allowed_groups(user: User) -> Optional[List[str]]:
    """用户的可见分组：全员 ``*`` 恒可见，另加其所属部门。

    与数据层 acl 约定配套（文档 metadata.acl 为 ``*`` 或单分组名）；
    返回 None 表示该用户不做 ACL 过滤（仅当配置关闭 acl_filter 时生效）。
    """
    groups = {"*"}
    dept = str(getattr(user, "department", "") or "").strip()
    if dept:
        groups.add(dept)
    return sorted(groups)


def _safe_filename(filename: Optional[str]) -> str:
    """清洗上传文件名，拒绝路径遍历（``../x``、``a\\..\\b``、绝对路径等）。"""
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    name = Path(filename).name.strip()
    if not name or name in {".", ".."} or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail=f"非法文件名: {filename!r}")
    return name


def _ensure_within(base: Path, target: Path) -> None:
    """确保 target 落在 base 目录内，防止任意目录读取。

    P5 审计修复：改用 ``Path.is_relative_to``。之前
    ``str(target).startswith(str(base))`` 没有路径分隔符语义，
    ``data/raw2`` 可以绕过 ``data/raw`` 的校验。
    """
    base_resolved = base.resolve()
    target_resolved = target.resolve()
    if not target_resolved.is_relative_to(base_resolved):
        raise HTTPException(
            status_code=403,
            detail=f"仅允许访问上传目录内的路径（{base_resolved}）",
        )


def _get_upload_limit_mb(cfg: Dict[str, Any]) -> int:
    """上传大小上限（MB），读 ui.max_upload_mb 配置（后端强制执行）。"""
    return int(cfg.get("ui", {}).get("max_upload_mb", 50) or 50)


def _save_upload_stream(file: UploadFile, dest: Path, max_bytes: int) -> int:
    """流式写盘并累计校验大小，超限抛 413 并清理半成品文件（同步，供线程池调用）。

    P5 审计修复：之前 ``await file.read()`` 把整个上传一次性读进内存且无
    大小上限，一个 2GB 的 POST 即可打爆进程内存。
    """
    size = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = file.file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过大小上限（{max_bytes // (1024 * 1024)} MB）",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return size


def _check_extension(filename: str) -> str:
    """校验扩展名在白名单内（写盘之前拒绝，避免垃圾文件落盘）。"""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {ext}，仅支持：{sorted(SUPPORTED_EXTENSIONS)}",
        )
    return ext


def _delete_stale_chunks(store, sources: List[str]) -> None:
    """入库前按 source 清理同名文档的旧分块。

    P5 审计修复：分块 id = md5(source|page|chunk_index) 只做 upsert，
    重新上传内容变少的文件时，超出的旧分块会永久残留并继续参与检索。
    """
    for src in set(sources):
        try:
            deleted = store.delete_by_metadata({"source": {"$eq": src}})
            if deleted:
                logger.info("已清理 %s 的 %d 条旧分块", src, deleted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("清理旧分块失败（source=%s）: %s", src, exc)


def _parse_datetime_safe(raw: Any) -> Optional[Any]:
    """把 metadata 中的 updated_at 字符串转回 datetime（解析失败返回 None）。"""
    from datetime import datetime as _dt

    try:
        return _dt.strptime(str(raw).strip(), "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def _register_documents(db: Session, user_id: str, chunks: List[Any], report: Dict[str, Any]) -> None:
    """数据层登记：把本次入库结果写入 document_records 台账（按 source upsert）。

    台账是"来源明确"的可审计依据：来源文件名、标题、状态、权限、内容哈希、
    分块数、源文件更新时间。与向量库 metadata 一一对应。
    """
    if not chunks:
        return
    md = dict(chunks[0].metadata or {})
    source = str(md.get("source") or md.get("filepath") or "unknown")
    hashes = report.get("content_hashes") or []
    values = {
        "title": md.get("title") or source,
        "doc_status": str(md.get("doc_status") or "active"),
        "acl": str(md.get("acl") or "*"),
        "department": str(md.get("department") or "") or None,
        "chunk_count": len(chunks),
        "file_ext": md.get("ext"),
        "content_hash": hashes[0] if hashes else None,
        "updated_at_source": _parse_datetime_safe(md.get("updated_at")),
    }
    try:
        rec = (
            db.query(DocumentRecord)
            .filter(DocumentRecord.user_id == user_id, DocumentRecord.source == source)
            .first()
        )
        if rec is None:
            db.add(DocumentRecord(user_id=user_id, source=source, **values))
        else:
            for k, v in values.items():
                setattr(rec, k, v)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.warning("文档登记失败（source=%s）: %s", source, exc)


# =========================== Documents ===========================
@router.get("/api/v1/documents", tags=["文档"])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """列出已入库文档（按 source 聚合，合并数据层台账的状态/权限/更新时间）。需要认证。"""
    user: User = current_user
    try:
        cfg = get_runtime_config()
        store = TenantAwareFactory.get_chroma_store(user.id, cfg)
        sources = await run_in_threadpool(store.list_sources)
        # 数据层台账：补齐 title / doc_status / acl / updated_at（向量库 metadata 之外的可审计信息）
        records = {
            r.source: r
            for r in db.query(DocumentRecord).filter(DocumentRecord.user_id == user.id).all()
        }
        for entry in sources:
            rec = records.get(str(entry.get("source")))
            if rec is not None:
                entry.update(
                    {
                        "title": rec.title,
                        "doc_status": rec.doc_status,
                        "acl": rec.acl,
                        "updated_at_source": rec.updated_at_source.strftime("%Y-%m-%d %H:%M:%S")
                        if rec.updated_at_source
                        else None,
                        "content_hash": rec.content_hash,
                    }
                )
        return {"documents": sources, "total": len(sources), "total_chunks": store.count()}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("列出文档失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/v1/documents/{doc_id}/chunks", tags=["文档"])
async def get_document_chunks(
    doc_id: str,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """获取指定文档的分块内容。需要认证。"""
    user: User = current_user
    try:
        cfg = get_runtime_config()
        store = TenantAwareFactory.get_chroma_store(user.id, cfg)

        def _fetch() -> Dict[str, Any]:
            return store.collection.get(
                where={"source": {"$eq": doc_id}},
                include=["documents", "metadatas"],
                limit=limit,
            )

        result = await run_in_threadpool(_fetch)
        chunks: List[Dict[str, Any]] = []
        for i, (doc, meta) in enumerate(
            zip(result.get("documents", []), result.get("metadatas", []))
        ):
            chunks.append(
                {
                    "id": result["ids"][i]
                    if "ids" in result and i < len(result["ids"])
                    else f"chunk_{i}",
                    "text": doc,
                    "metadata": meta,
                }
            )
        return {"doc_id": doc_id, "chunks": chunks, "total": len(chunks)}
    except Exception as exc:
        logger.error("获取分块失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# =========================== Search ===========================
class SearchBody(BaseModel):
    """POST /api/v1/search 请求体。"""

    query: str
    top_k: int = 4


@router.post("/api/v1/search", tags=["文档"])
async def search_documents(
    body: SearchBody,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """语义检索文档（仅返回 active 且用户有权可见的分块）。需要认证。"""
    user: User = current_user
    try:
        cfg = get_runtime_config()
        store = TenantAwareFactory.get_chroma_store(user.id, cfg)
        retrieval_cfg = cfg.get("retrieval", {})

        def _query() -> List[Any]:
            from src.vector_store import build_security_where

            where = build_security_where(
                status_filter=bool(retrieval_cfg.get("status_filter", True)),
                acl_groups=_allowed_groups(user) if retrieval_cfg.get("acl_filter", True) else None,
            )
            hits = store.query(query_text=body.query, top_k=body.top_k, where=where)
            # 旧索引兼容回退（仅状态过滤；ACL 绝不回退）
            if not hits and where is not None and retrieval_cfg.get("status_filter_fallback", True):
                hits = store.query(query_text=body.query, top_k=body.top_k)
            return hits

        hits = await run_in_threadpool(_query)
        return {
            "query": body.query,
            "hits": [
                {
                    "id": h.id,
                    "text": h.text[:500],
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


# =========================== Sync Chat ===========================
class ChatRequest(BaseModel):
    """POST /api/v1/chat 请求体（非流式）。"""

    query: str = Field(..., min_length=1, description="查询文本，不能为空")
    top_k: int = Field(4, ge=1, le=50, description="检索 Top-K")
    session_id: str = Field("", description="会话 ID；同一会话的多轮问答共享即时记忆")


@router.post("/api/v1/chat", tags=["问答"])
def chat_sync(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """同步问答（非流式）。需要认证。复用全局 Pipeline + LLM 单例。

    说明：必须是 ``def``（而非 ``async def``）——``pipeline.answer`` 内含
    分钟级的阻塞 LLM 推理，放在 async 端点里会把整个事件循环卡死，
    FastAPI 会自动把 ``def`` 端点调度到线程池。

    上下文与记忆：
    - 即时记忆：``session_id`` 相同的请求共享最近几轮对话（SessionMemory）；
    - 长期记忆：用户消息中的"记住……"显式指令写入 long_term_memories；
    - 检索记忆：由 pipeline 实时召回；
    - ACL：检索仅召回用户可见分组的分块。
    """
    user: User = current_user
    check_query_quota(db, user)
    started = time.perf_counter()
    status_code = 200
    error_detail: Optional[str] = None
    try:
        pipeline = get_runtime_pipeline(user_id=user.id)
        session_id = (body.session_id or "").strip() or "default"

        # 长期记忆：显式"记住"指令先行落库（替身 pipeline / 存储缺失时静默跳过）
        memories_written: List[Dict[str, str]] = []
        lt_store = getattr(pipeline, "long_term_store", None)
        if lt_store is not None:
            for key, value in lt_store.maybe_remember_from_message(user.id, body.query):
                memories_written.append({"key": key, "value": value})

        # 即时记忆：取当前会话窗口
        history = SESSION_MEMORY.get_history(user.id, session_id)

        result = pipeline.answer(
            body.query,
            top_k=body.top_k,
            history=history,
            user_id=user.id,
            allowed_groups=_allowed_groups(user),
        )

        # 回写即时记忆（user + assistant 各一条）
        SESSION_MEMORY.append(user.id, session_id, "user", body.query)
        SESSION_MEMORY.append(user.id, session_id, "assistant", result.answer or "")

        return {
            "answer": result.answer,
            "sources": result.sources,
            "timings": result.timings,
            "verification": result.verification,
            "session_id": session_id,
            "memories_written": memories_written,
        }
    except HTTPException:
        raise
    except Exception as exc:
        status_code = 500
        error_detail = str(exc)
        logger.error("同步问答失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        record_usage(
            db,
            user.id,
            CHAT_ENDPOINT,
            status_code=status_code,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error_message=error_detail,
        )


# =========================== Ingest ===========================
@router.post("/api/v1/ingest", tags=["文档"])
async def ingest_file(
    file: UploadFile = File(...),
    version: bool = False,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """上传文件并入库。需要认证。

    安全（P5 审计修复）：
    - 扩展名白名单在写盘前校验；
    - 流式写盘并强制执行 ``ui.max_upload_mb`` 大小上限；
    - 入库前校验套餐 chunk 配额；
    - 入库前清理同名文档的旧分块，避免内容更新后残留旧块。
    """
    user: User = current_user
    try:
        cfg = get_runtime_config()

        # 保存上传文件到用户隔离目录（文件名清洗，防路径遍历）
        raw_dir = resolve_path(cfg.get("paths", {}).get("raw_docs", "data/raw"))
        user_raw_dir = raw_dir / user.id[:8]
        ensure_dir(user_raw_dir)
        safe_name = _safe_filename(file.filename)
        _check_extension(safe_name)
        temp_path = user_raw_dir / safe_name

        max_bytes = _get_upload_limit_mb(cfg) * 1024 * 1024
        await run_in_threadpool(_save_upload_stream, file, temp_path, max_bytes)

        splitter = get_splitter(cfg)

        def _parse_and_chunk() -> tuple:
            """数据层统一管线：元数据增强 → 深度清洗 → 去重 → 准入门 → 分块。

            标题/层级/更新时间/权限在 enrich_file 中抽取；过期/草稿/权限
            不清的内容被 DataGate 拒绝（拒绝原因返回给调用方便于整改）。
            """
            docs = enrich_file(
                temp_path,
                default_acl=cfg.get("data_quality", {}).get("default_acl", "*"),
            )
            if not docs:
                docs = load_document(
                    temp_path,
                    pdf_engine=cfg["document_loader"].get("pdf_engine", "pdfplumber"),
                    encoding=cfg["document_loader"].get("encoding", "utf-8"),
                )
            chunks, report = run_data_pipeline(docs, cfg, splitter)
            report["docs"] = len(docs)
            return len(docs), chunks, report

        docs_count, all_chunks, report = await run_in_threadpool(_parse_and_chunk)
        if not all_chunks:
            temp_path.unlink(missing_ok=True)
            gate = report.get("gate", {})
            if gate.get("rejected"):
                raise HTTPException(
                    status_code=422,
                    detail=f"内容未通过数据准入门：{gate.get('reasons', {})} {gate.get('details', [])[:5]}",
                )
            raise HTTPException(status_code=400, detail="文件解析后没有可用内容")

        store = TenantAwareFactory.get_chroma_store(user.id, cfg)

        def _replace_and_add() -> int:
            _delete_stale_chunks(store, [str(c.metadata.get("source") or c.metadata.get("filepath") or "unknown") for c in all_chunks])
            return store.add_chunks(all_chunks)

        # 配额校验：现有块数 + 新增块数 <= 套餐上限
        existing = await run_in_threadpool(store.count)
        check_chunk_quota(user, existing, len(all_chunks))

        n = await run_in_threadpool(_replace_and_add)

        # 写入成功后更新数据层台账（含准入/去重统计）
        db = SessionLocal()
        try:
            _register_documents(db, user.id, all_chunks, report)
        finally:
            db.close()

        return {
            "filename": file.filename,
            "docs": docs_count,
            "chunks": len(all_chunks),
            "ingested": n,
            "data_quality": report,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("文件入库失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/v1/ingest/dir", tags=["文档"])
async def ingest_directory(
    dir_path: str = Form(...),
    recursive: bool = Form(True),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """批量入库指定目录。需要认证。

    安全（P5 审计修复）：仅允许访问**当前用户自己的**上传子目录
    ``paths.raw_docs/<user.id[:8]>``。之前只校验到 ``data/raw`` 根，
    用户可以把其他用户上传的文件批量收进自己的知识库。
    """
    user: User = current_user
    try:
        cfg = get_runtime_config()

        raw_root = resolve_path(cfg.get("paths", {}).get("raw_docs", "data/raw"))
        user_scope = raw_root / user.id[:8]
        target = resolve_path(dir_path)
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"目录不存在: {dir_path}")
        _ensure_within(user_scope, target)

        splitter = get_splitter(cfg)

        def _parse_and_chunk() -> tuple:
            """与单文件入库相同的数据层管线（元数据增强 → 清洗 → 去重 → 准入 → 分块）。"""
            docs = enrich_directory(
                target,
                default_acl=cfg.get("data_quality", {}).get("default_acl", "*"),
                recursive=recursive,
            )
            if not docs:
                docs = load_directory(
                    target,
                    pdf_engine=cfg["document_loader"].get("pdf_engine", "pdfplumber"),
                    recursive=recursive,
                    encoding=cfg["document_loader"].get("encoding", "utf-8"),
                )
            chunks, report = run_data_pipeline(docs, cfg, splitter)
            report["docs"] = len(docs)
            return len(docs), chunks, report

        docs_count, all_chunks, report = await run_in_threadpool(_parse_and_chunk)
        if not all_chunks:
            gate = report.get("gate", {})
            if gate.get("rejected"):
                raise HTTPException(
                    status_code=422,
                    detail=f"内容未通过数据准入门：{gate.get('reasons', {})} {gate.get('details', [])[:5]}",
                )
            raise HTTPException(status_code=400, detail="目录中没有可入库内容")

        store = TenantAwareFactory.get_chroma_store(user.id, cfg)

        def _replace_and_add() -> int:
            _delete_stale_chunks(store, [str(c.metadata.get("source") or c.metadata.get("filepath") or "unknown") for c in all_chunks])
            return store.add_chunks(all_chunks)

        existing = await run_in_threadpool(store.count)
        check_chunk_quota(user, existing, len(all_chunks))

        n = await run_in_threadpool(_replace_and_add)

        db = SessionLocal()
        try:
            _register_documents(db, user.id, all_chunks, report)
        finally:
            db.close()

        return {
            "dir": str(target),
            "docs": docs_count,
            "chunks": len(all_chunks),
            "ingested": n,
            "data_quality": report,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("目录入库失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# =========================== WebSocket Chat ===========================
@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, token: Optional[str] = None):
    """WebSocket 流式问答（兼容旧前端路径）。

    安全：必须携带有效 JWT（``?token=``），否则拒绝连接——避免匿名会话
    落到 admin 全局知识库上。

    P1.3：通过 ``AsyncLLMExecutor.semaphore`` 限制并发 LLM 调用，避免
    单实例下被并发 WS 流量击穿。
    """
    from src.auth.jwt_handler import decode_token

    payload = decode_token(token) if token else None
    if not payload or not payload.get("sub"):
        await websocket.close(code=4401)  # 未认证
        return
    user_id = payload["sub"]

    await websocket.accept()
    # 加载一次用户并校验状态（配额校验需要 max_queries_per_day）
    db = SessionLocal()
    try:
        ws_user = db.query(User).filter(User.id == user_id).first()
        if ws_user is None or not ws_user.is_active:
            await websocket.close(code=4403)  # 用户不存在或已禁用
            return
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            top_k = data.get("top_k", 4)
            session_id = str(data.get("session_id") or "default")

            if not query:
                await websocket.send_json({"event": "error", "data": "query is required"})
                continue

            # P5 审计修复：WS 问答同样受每日配额约束
            try:
                check_query_quota(db, ws_user)
            except HTTPException as quota_exc:
                await websocket.send_json({"event": "error", "data": str(quota_exc.detail)})
                continue

            executor = get_llm_executor()
            started = time.perf_counter()

            # 长期记忆：显式"记住"指令先行落库（失败/替身缺失均不影响问答）
            try:
                pipeline_probe: RAGPipeline = get_runtime_pipeline(user_id=user_id)
                lt_probe = getattr(pipeline_probe, "long_term_store", None)
                if lt_probe is not None:
                    lt_probe.maybe_remember_from_message(user_id, query)
            except Exception as exc:  # noqa: BLE001
                logger.warning("WS 长期记忆写入失败（已忽略）: %s", exc)

            history = SESSION_MEMORY.get_history(user_id, session_id)
            answer_buf = {"text": ""}

            # P1.3：semaphore 保护，避免并发 WS 流量击穿 LLM
            async def _run():
                pipeline: RAGPipeline = get_runtime_pipeline(user_id=user_id)
                async for event in pipeline.astream_answer(
                    query,
                    top_k=top_k,
                    history=history,
                    user_id=user_id,
                    allowed_groups=_allowed_groups(ws_user),
                ):
                    if event.get("event") == "token":
                        answer_buf["text"] += str(event.get("data") or "")
                    await websocket.send_json(event)

            try:
                # asyncio.Semaphore 是 async 上下文管理器 — 直接 await 即可。
                # 当前 AsyncLLMExecutor 暴露 ``semaphore`` 属性。
                sem = getattr(executor, "semaphore", None)
                if sem is not None:
                    async with sem:
                        await _run()
                else:
                    await _run()
                # 回写即时记忆
                SESSION_MEMORY.append(user_id, session_id, "user", query)
                SESSION_MEMORY.append(user_id, session_id, "assistant", answer_buf["text"])
                record_usage(
                    db,
                    ws_user.id,
                    CHAT_ENDPOINT,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            except Exception as exc:
                logger.error("WebSocket 流式生成失败: %s", exc)
                record_usage(
                    db,
                    ws_user.id,
                    CHAT_ENDPOINT,
                    status_code=500,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error_message=str(exc),
                )
                await websocket.send_json({"event": "error", "data": str(exc)})
    except WebSocketDisconnect:
        logger.info("WebSocket 客户端断开")
    except Exception as exc:
        logger.error("WebSocket 异常: %s", exc)
    finally:
        db.close()