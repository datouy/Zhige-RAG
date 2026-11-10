"""问答 / 检索 / 文档管理路由。

端点：
- GET  /api/v1/documents
- GET  /api/v1/documents/{doc_id}/chunks
- POST /api/v1/search
- POST /api/v1/chat          （同步 RAG 问答）
- POST /api/v1/ingest        （单文件入库）
- POST /api/v1/ingest/dir    （目录批量入库）
- WS   /ws/chat              （兼容旧前端）

WS 聊天处理器通过 ``LLMConcurrencyGate`` 进行并发门控（P1.3），通过单例
Pipeline 复用 LLM / Embedding（P1.4）。
"""
from __future__ import annotations

import hashlib
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
    get_decision_provider,
    get_llm_gate,
    get_runtime_config,
    get_runtime_pipeline,
    get_splitter,
)
from src.data_quality import run_data_pipeline
from src.db.database import SessionLocal, get_db
from src.db.models import DocumentRecord, User
from src.document_loader import load_document, load_directory
from src.document_meta import enrich_directory, enrich_file
from src.factories import TenantAwareFactory
from src.memory import SESSION_MEMORY
from src.middleware.identity import get_current_user
from src.rag_pipeline import RAGPipeline
from src.utils import SUPPORTED_EXTENSIONS, ensure_dir, get_logger, resolve_path

logger = get_logger("api.routes.chat")

router = APIRouter()


def _ingest_guard_provider(cfg: Dict[str, Any]):
    """入库门禁用的 d1 决策 provider。

    仅在 ``data_quality.ingest_guard.enabled`` 打开时才去取 provider ——
    未启用时返回 ``None``，``run_data_pipeline`` 会整体跳过门禁（零开销）。
    """
    try:
        gcfg = (cfg.get("data_quality", {}) or {}).get("ingest_guard", {}) or {}
        if not gcfg.get("enabled"):
            return None
        return get_decision_provider()
    except Exception as exc:  # noqa: BLE001 - 取值失败就相当于门禁关闭
        logger.warning("获取入库门禁 provider 失败（已跳过门禁）: %s", exc)
        return None

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

    grouped: Dict[tuple[str, Optional[str]], List[Any]] = {}
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        source = str(metadata.get("source") or metadata.get("filepath") or "unknown")
        grouped.setdefault((source, metadata.get("kb_id")), []).append(chunk)

    report_hashes = report.get("content_hashes") or []
    for index, ((source, kb_id), source_chunks) in enumerate(grouped.items()):
        metadata = dict(source_chunks[0].metadata or {})
        content = "\n".join(str(chunk.text or "") for chunk in source_chunks)
        content_hash = (
            hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content
            else (report_hashes[index] if index < len(report_hashes) else None)
        )
        values = {
            "kb_id": kb_id,
            "title": metadata.get("title") or source,
            "doc_status": str(metadata.get("doc_status") or "active"),
            "acl": str(metadata.get("acl") or "*"),
            "department": str(metadata.get("department") or "") or None,
            "chunk_count": len(source_chunks),
            "file_ext": metadata.get("ext"),
            "content_hash": content_hash,
            "updated_at_source": _parse_datetime_safe(metadata.get("updated_at")),
        }
        query = db.query(DocumentRecord).filter(
            DocumentRecord.user_id == user_id,
            DocumentRecord.source == source,
        )
        query = (
            query.filter(DocumentRecord.kb_id == kb_id)
            if kb_id is not None
            else query.filter(DocumentRecord.kb_id.is_(None))
        )
        record = query.first()
        if record is None:
            db.add(DocumentRecord(user_id=user_id, source=source, **values))
        else:
            for key, value in values.items():
                setattr(record, key, value)
    db.flush()


def _commit_ingestion_metadata(
    user_id: str,
    chunks: List[Any],
    report: Dict[str, Any],
    kb_id: str,
) -> None:
    """在向量事务回调中一次性提交台账与知识库统计。"""
    from src.kb_service import refresh_kb_stats

    db = SessionLocal()
    try:
        _register_documents(db, user_id, chunks, report)
        refresh_kb_stats(db, user_id, kb_id, commit=False)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# =========================== Documents ===========================
@router.get("/api/v1/documents", tags=["文档"])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """列出已入库文档（按 source 聚合，合并数据层台账的状态/权限/更新时间）。需要认证。"""
    user: User = current_user
    try:
        records = (
            db.query(DocumentRecord)
            .filter(DocumentRecord.user_id == user.id)
            .order_by(DocumentRecord.source.asc(), DocumentRecord.kb_id.asc())
            .all()
        )
        documents = [
            {
                "source": record.source,
                "title": record.title,
                "kb_id": record.kb_id,
                "chunks": int(record.chunk_count or 0),
                "doc_status": record.doc_status,
                "acl": record.acl,
                "updated_at_source": record.updated_at_source.strftime("%Y-%m-%d %H:%M:%S")
                if record.updated_at_source
                else None,
                "content_hash": record.content_hash,
            }
            for record in records
        ]
        return {
            "documents": documents,
            "total": len(documents),
            "total_chunks": sum(item["chunks"] for item in documents),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("列出文档失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/v1/documents/{doc_id}/chunks", tags=["文档"])
async def get_document_chunks(
    doc_id: str,
    limit: int = 50,
    kb_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """获取指定文档的分块内容。需要认证。

    `doc_id` 为文档 source（文件名）。`kb_id` 指定知识库，缺省为默认库；
    之前这里固定用默认库的 collection，导致非默认库的文档分块读不到。
    """
    user: User = current_user
    try:
        cfg = get_runtime_config()
        kb = _resolve_kb(user, cfg, kb_id)
        store = TenantAwareFactory.get_chroma_store_by_name(kb["collection_name"], cfg)

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


class ChunkUpdateBody(BaseModel):
    """P1-1 分块编辑：更新分块文本并重嵌。"""

    text: str = Field(..., min_length=1, description="新的分块文本")


@router.patch("/api/v1/documents/chunks/{chunk_id}", tags=["文档"])
async def update_document_chunk(
    chunk_id: str,
    body: ChunkUpdateBody,
    kb_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """更新单个分块文本并重嵌向量 + 同步 BM25。需要认证。

    `kb_id` 指定知识库，缺省为默认库；之前固定用默认库 collection，
    非默认库的分块改不到。
    """
    user: User = current_user
    try:
        cfg = get_runtime_config()
        kb = _resolve_kb(user, cfg, kb_id)
        store = TenantAwareFactory.get_chroma_store_by_name(kb["collection_name"], cfg)

        def _patch() -> Dict[str, Any]:
            got = store.collection.get(ids=[chunk_id], include=["documents", "metadatas", "embeddings"])
            ids = got.get("ids") or []
            if not ids:
                raise HTTPException(status_code=404, detail=f"分块不存在: {chunk_id}")
            meta = (got.get("metadatas") or [{}])[0] or {}
            new_text = body.text.strip()
            vec = store.embedding_model.encode([new_text], batch_size=1).tolist()[0]
            store.collection.upsert(ids=[chunk_id], documents=[new_text], embeddings=[vec], metadatas=[meta])
            if store._bm25 is not None:
                store._bm25.upsert_documents([chunk_id], [new_text], [meta])
            return {"id": chunk_id, "text": new_text, "metadata": meta}

        return await run_in_threadpool(_patch)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("更新分块失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# =========================== Search ===========================
class SearchBody(BaseModel):
    """POST /api/v1/search 请求体。"""

    query: str
    top_k: int = 4
    kb_id: Optional[str] = Field(None, description="在指定知识库内检索，缺省为默认知识库")


@router.post("/api/v1/search", tags=["文档"])
async def search_documents(
    body: SearchBody,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """语义检索文档（仅返回 active 且用户有权可见的分块）。需要认证。"""
    user: User = current_user
    try:
        cfg = get_runtime_config()
        kb = _resolve_kb(user, cfg, body.kb_id)
        store = TenantAwareFactory.get_chroma_store_by_name(kb["collection_name"], cfg)
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
            "kb_id": kb["id"],
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
    kb_id: Optional[str] = Field(None, description="在指定知识库内检索，缺省为默认知识库")


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
    try:
        # 指定知识库时在该库内检索，否则用默认库（行为与之前一致）
        kb = _resolve_kb(user, get_runtime_config(), body.kb_id) if body.kb_id else None
        pipeline = get_runtime_pipeline(user_id=user.id, kb=kb)
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

        # 会话持久化：session_id 指向真实会话时把本轮问答落库（供会话列表与历史回看）。
        # 前端自由传入的 id 取不到会话记录 → 静默跳过，不影响问答本身。
        if body.session_id and session_id != "default":
            try:
                from src.kb_service import KBError, append_message

                append_message(db, user.id, session_id, "user", body.query)
                append_message(
                    db, user.id, session_id, "assistant", result.answer or "", result.sources
                )
            except KBError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning("会话消息落库失败（已忽略）: %s", exc)

        return {
            "answer": result.answer,
            "sources": result.sources,
            "timings": result.timings,
            "verification": result.verification,
            "session_id": session_id,
            "kb_id": (kb or {}).get("id"),
            "memories_written": memories_written,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("同步问答失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# =========================== Ingest ===========================
def _resolve_kb(user: User, cfg: Dict[str, Any], kb_id: Optional[str]) -> Dict[str, str]:
    """解析本次写入的目标知识库。

    返回**脱离 ORM 会话的普通 dict**（id / name / collection_name）——
    直接把 ORM 对象带出 ``SessionLocal()`` 会在属性访问时抛
    ``DetachedInstanceError``。

    ``kb_id`` 为空时落到默认知识库（保证老客户端行为不变）。
    """
    from src.db.database import SessionLocal
    from src.kb_service import KBError, ensure_default_kb, get_kb

    db = SessionLocal()
    try:
        if kb_id:
            try:
                kb = get_kb(db, user.id, kb_id)
            except KBError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        else:
            kb = ensure_default_kb(db, user.id, cfg)
        return {"id": kb.id, "name": kb.name, "collection_name": kb.collection_name}
    finally:
        db.close()


def _maybe_trigger_kg_build(user_id: str, cfg: Dict[str, Any]) -> None:
    """入库后按需异步构建知识图谱（P1-6）。

    仅在 knowledge_graph.enabled 且 auto_build_on_ingest 同时开启，
    并且存在可用的生成式 LLM 时提交后台任务。extractive（无 LLM）模式、
    LLM 未就绪、raw 目录不存在时静默跳过，绝不影响入库主链路。
    """
    kg_cfg = cfg.get("knowledge_graph") or {}
    if not kg_cfg.get("enabled") or not kg_cfg.get("auto_build_on_ingest"):
        return
    try:
        pipeline = get_runtime_pipeline(user_id=user_id)
        if pipeline._is_extractive_mode():
            logger.info("KG 自动构建跳过：extractive 模式无生成式 LLM")
            return
        pipeline.ensure_llm()
        llm = pipeline.llm
    except Exception as exc:  # noqa: BLE001 - LLM 未就绪不影响入库
        logger.warning("KG 自动构建跳过（LLM 未就绪）: %s", exc)
        return
    if llm is None:
        return
    raw_dir = resolve_path(cfg.get("paths", {}).get("raw_docs", "data/raw")) / user_id[:8]
    if not raw_dir.exists():
        return
    try:
        from src.ingest_jobs import create_job, submit as _submit_job
        from scripts.build_kg import build as _build_kg

        job_id = create_job("kg_build", "知识图谱自动构建", {"dir": str(raw_dir)})

        def _bg(reporter) -> Dict[str, Any]:
            reporter(0.05, "extract")
            kg = TenantAwareFactory.get_kg_store(user_id, cfg)
            summary = _build_kg(input_dir=str(raw_dir), llm=llm, store=kg)
            reporter(1.0, "done")
            return summary

        _submit_job(_bg, job_id)
        logger.info("已提交知识图谱后台构建任务 %s（%s）", job_id, raw_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("提交 KG 后台构建任务失败: %s", exc)



@router.post("/api/v1/ingest", tags=["文档"])
async def ingest_file(
    file: UploadFile = File(...),
    version: bool = False,
    kb_id: Optional[str] = Form(None),
    background: bool = Form(False, description="true=异步入库，立即返回job_id，后台解析+upsert"),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """上传文件并入库。需要认证。

    ``kb_id`` 指定写入哪个知识库，缺省写入**默认知识库**（行为与之前一致）。
    每个知识库有独立的向量 collection，互不干扰。

    安全（P5 审计修复）：
    - 扩展名白名单在写盘前校验；
    - 流式写盘并强制执行 ``ui.max_upload_mb`` 大小上限；
    - 入库前清理同名文档的旧分块，避免内容更新后残留旧块。

    P1 异步入库：``background=true`` 时保存文件后立即返回
    ``{"job_id": ...}``，解析→分块→upsert 在后台线程池执行，
    通过 ``GET /api/v1/ingest/jobs/{job_id}`` 轮询进度（50MB文件不再超时）。
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

        # P1 异步入库：background=true 时落盘后即返，后台线程做重活
        if background:
            from src.ingest_jobs import create_job, submit as _submit_job

            _kb_early = _resolve_kb(user, cfg, kb_id)
            _uid, _kbid, _cfg_snapshot = user.id, _kb_early["id"], dict(cfg)
            _orig_name = file.filename
            _tmp = str(temp_path)
            job_id = create_job("file", _orig_name or safe_name,
                               {"kb_id": _kbid, "path": _tmp})

            def _bg_file(reporter) -> Dict[str, Any]:
                from pathlib import Path as _P
                reporter(0.1, "parse")
                _split = get_splitter(_cfg_snapshot)
                ocr_c = (_cfg_snapshot.get("document_loader", {}).get("ocr")) or {}
                docs = enrich_file(
                    _P(_tmp),
                    default_acl=_cfg_snapshot.get("data_quality", {}).get("default_acl", "*"),
                    ocr_cfg=ocr_c,
                )
                if not docs:
                    docs = load_document(
                        _P(_tmp),
                        pdf_engine=_cfg_snapshot["document_loader"].get("pdf_engine", "pdfplumber"),
                        encoding=_cfg_snapshot["document_loader"].get("encoding", "utf-8"),
                        ocr_cfg=ocr_c,
                    )
                chunks, rep = run_data_pipeline(
                    docs, _cfg_snapshot, _split,
                    decision_provider=_ingest_guard_provider(_cfg_snapshot))
                rep["docs"] = len(docs)
                if not chunks:
                    raise RuntimeError(f"文件解析后没有可用内容: {rep.get('gate', {})}")
                reporter(0.55, "upsert")
                for c in chunks:
                    c.metadata["kb_id"] = _kbid
                _kb_info = {"id": _kbid,
                            "collection_name": _kb_early["collection_name"]}
                _store = TenantAwareFactory.get_chroma_store_by_name(
                    _kb_info["collection_name"], _cfg_snapshot)
                _srcs = sorted({str(c.metadata.get("source")
                                    or c.metadata.get("filepath") or "unknown")
                                for c in chunks})
                # 分批128由 add_chunks 内部处理；大文件不一次性爆内存
                n = _store.replace_chunks_for_sources(
                    chunks, _srcs,
                    on_commit=lambda: _commit_ingestion_metadata(
                        _uid, chunks, rep, _kbid))
                reporter(0.95, "invalidate")
                try:
                    from api.deps import get_runtime_pipeline as _grp
                    _grp(user_id=_uid).invalidate_answer_cache()
                except Exception:
                    pass
                _maybe_trigger_kg_build(_uid, _cfg_snapshot)
                return {"filename": _orig_name, "docs": len(docs),
                        "chunks": len(chunks), "ingested": n,
                        "kb_id": _kbid, "data_quality": rep}

            _submit_job(_bg_file, job_id)
            return {"job_id": job_id, "status": "queued",
                    "filename": file.filename, "kb_id": _kb_early["id"],
                    "poll": f"/api/v1/ingest/jobs/{job_id}"}

        splitter = get_splitter(cfg)

        def _parse_and_chunk() -> tuple:
            """数据层统一管线：元数据增强 → 深度清洗 → 去重 → 准入门 → 分块。

            标题/层级/更新时间/权限在 enrich_file 中抽取；过期/草稿/权限
            不清的内容被 DataGate 拒绝（拒绝原因返回给调用方便于整改）。
            """
            ocr_cfg = cfg.get("document_loader", {}).get("ocr") or {}
            docs = enrich_file(
                temp_path,
                default_acl=cfg.get("data_quality", {}).get("default_acl", "*"),
                ocr_cfg=ocr_cfg,
            )
            if not docs:
                docs = load_document(
                    temp_path,
                    pdf_engine=cfg["document_loader"].get("pdf_engine", "pdfplumber"),
                    encoding=cfg["document_loader"].get("encoding", "utf-8"),
                    ocr_cfg=ocr_cfg,
                )
            chunks, report = run_data_pipeline(
                docs, cfg, splitter, decision_provider=_ingest_guard_provider(cfg)
            )
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

        # 知识库隔离：写入目标知识库专属的 collection
        kb = _resolve_kb(user, cfg, kb_id)
        for c in all_chunks:
            c.metadata["kb_id"] = kb["id"]
        store = TenantAwareFactory.get_chroma_store_by_name(kb["collection_name"], cfg)

        sources = sorted(
            {
                str(c.metadata.get("source") or c.metadata.get("filepath") or "unknown")
                for c in all_chunks
            }
        )

        def _replace() -> int:
            return store.replace_chunks_for_sources(
                all_chunks,
                sources,
                on_commit=lambda: _commit_ingestion_metadata(
                    user.id, all_chunks, report, kb["id"]
                ),
            )

        n = await run_in_threadpool(_replace)

        # P1：入库成功后失效答案/检索缓存，避免旧答案残留
        try:
            from api.deps import get_runtime_pipeline as _grp_sync
            _grp_sync(user_id=user.id).invalidate_answer_cache()
        except Exception:
            pass

        _maybe_trigger_kg_build(user.id, cfg)

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


@router.get("/api/v1/ingest/jobs/{job_id}", tags=["文档"])
def ingest_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """查询异步入库任务状态（queued/running/done/failed + progress/stage/result）。

    P1 异步入库配套：``POST /api/v1/ingest?background=true`` 返回 job_id 后，
    前端轮询本接口拿进度，done 后 result 即同步入库同款返回体。
    """
    from src.ingest_jobs import get_job

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    return job


class IngestURLBody(BaseModel):
    """网页入库请求。"""
    url: str = Field(..., min_length=4, description="网页地址（可省略协议，默认 https）")
    title: Optional[str] = Field(None, description="自定义标题，默认取网页 <title>")
    kb_id: Optional[str] = Field(None, description="目标知识库，缺省为默认知识库")
    background: bool = Field(False, description="true=异步入库，立即返回job_id")
    timeout: float = Field(15.0, ge=1.0, le=60.0, description="抓取超时（秒）")


@router.post("/api/v1/ingest/url", tags=["文档"])
async def ingest_url(
    body: IngestURLBody,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """抓取网页正文并入库。需要认证。

    安全（SSRF 防护，实现见 ``src/web_loader.py``）：

    - 仅允许 http / https 协议；
    - **校验域名解析后的 IP**，拒绝内网 / 回环 / 链路本地 / 保留地址 ——
      只校验域名会被 DNS rebinding（``evil.com`` → ``127.0.0.1``）绕过；
    - 重定向**逐跳**复检，避免"公网地址 302 到内网"；
    - 限制响应体大小与 Content-Type，拒绝二进制。

    说明：以最终 URL 作为 ``source``，同一页面重复导入会覆盖旧分块而不是堆积。
    """
    user: User = current_user
    try:
        cfg = get_runtime_config()

        # P1 异步入库：URL抓取也可能慢，先返job_id，后台fetch→分块→upsert
        if body.background:
            from src.ingest_jobs import create_job, submit as _submit_job

            _kb_early = _resolve_kb(user, cfg, body.kb_id)
            _uid, _kbid = user.id, _kb_early["id"]
            _cfg_snapshot = dict(cfg)
            _body_url, _body_title = body.url, body.title
            _body_timeout = body.timeout
            job_id = create_job("url", _body_url, {"kb_id": _kbid})

            def _bg_url(reporter) -> Dict[str, Any]:
                from src.document_loader import Document as _Doc
                from src.web_loader import fetch_and_extract as _fetch2
                reporter(0.1, "fetch")
                _url_cfg = (_cfg_snapshot.get("document_loader") or {}).get("url") or {}
                page = _fetch2(
                    _body_url, timeout=_body_timeout,
                    max_bytes=int(_url_cfg.get("max_bytes", 5 * 1024 * 1024)),
                    allow_private=bool(_url_cfg.get("allow_private", False)),
                    max_chars=int(_url_cfg.get("max_chars", 200000)))
                title = (_body_title or page["title"] or page["url"]).strip()
                source = page["final_url"] or page["url"]
                docs = [_Doc(page=1, content=page["text"], metadata={
                    "source": source, "filepath": source, "filename": title,
                    "title": title, "ext": ".url", "url": page["url"],
                    "final_url": source,
                    "acl": _cfg_snapshot.get("data_quality", {}).get("default_acl", "*")})]
                reporter(0.4, "chunk")
                _split = get_splitter(_cfg_snapshot)
                chunks, rep = run_data_pipeline(
                    docs, _cfg_snapshot, _split,
                    decision_provider=_ingest_guard_provider(_cfg_snapshot))
                rep["docs"] = len(docs)
                if not chunks:
                    raise RuntimeError("网页正文为空，无法入库")
                for c in chunks:
                    c.metadata["kb_id"] = _kbid
                reporter(0.7, "upsert")
                _store = TenantAwareFactory.get_chroma_store_by_name(
                    _kb_early["collection_name"], _cfg_snapshot)
                n = _store.replace_chunks_for_sources(
                    chunks, [source],
                    on_commit=lambda: _commit_ingestion_metadata(
                        _uid, chunks, rep, _kbid))
                try:
                    from api.deps import get_runtime_pipeline as _grp
                    _grp(user_id=_uid).invalidate_answer_cache()
                except Exception:
                    pass
                _maybe_trigger_kg_build(_uid, _cfg_snapshot)
                return {"url": page["url"], "final_url": source, "title": title,
                        "chars": int(page["chars"]), "docs": len(docs),
                        "chunks": len(chunks), "ingested": n, "kb_id": _kbid}

            _submit_job(_bg_url, job_id)
            return {"job_id": job_id, "status": "queued", "url": body.url,
                    "kb_id": _kb_early["id"],
                    "poll": f"/api/v1/ingest/jobs/{job_id}"}

        url_cfg = (cfg.get("document_loader") or {}).get("url") or {}
        allow_private = bool(url_cfg.get("allow_private", False))

        from src.document_loader import Document
        from src.web_loader import WebFetchError, fetch_and_extract

        def _fetch() -> Dict[str, Any]:
            return fetch_and_extract(
                body.url,
                timeout=body.timeout,
                max_bytes=int(url_cfg.get("max_bytes", 5 * 1024 * 1024)),
                allow_private=allow_private,
                max_chars=int(url_cfg.get("max_chars", 200000)),
            )

        try:
            page = await run_in_threadpool(_fetch)
        except WebFetchError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        title = (body.title or page["title"] or page["url"]).strip()
        source = page["final_url"] or page["url"]
        docs = [
            Document(
                page=1,
                content=page["text"],
                metadata={
                    "source": source,
                    "filepath": source,
                    "filename": title,
                    "title": title,
                    "ext": ".url",
                    "url": page["url"],
                    "final_url": source,
                    "acl": cfg.get("data_quality", {}).get("default_acl", "*"),
                },
            )
        ]

        splitter = get_splitter(cfg)
        chunks, report = await run_in_threadpool(
            run_data_pipeline, docs, cfg, splitter, _ingest_guard_provider(cfg)
        )
        report["docs"] = len(docs)
        if not chunks:
            gate = report.get("gate", {})
            if gate.get("rejected"):
                raise HTTPException(
                    status_code=422,
                    detail=f"内容未通过数据准入门：{gate.get('reasons', {})}",
                )
            raise HTTPException(status_code=400, detail="网页正文为空，无法入库")

        # 知识库隔离：写入目标知识库专属的 collection
        kb = _resolve_kb(user, cfg, body.kb_id)
        for c in chunks:
            c.metadata["kb_id"] = kb["id"]
        store = TenantAwareFactory.get_chroma_store_by_name(kb["collection_name"], cfg)

        def _replace() -> int:
            return store.replace_chunks_for_sources(
                chunks,
                [source],
                on_commit=lambda: _commit_ingestion_metadata(
                    user.id, chunks, report, kb["id"]
                ),
            )

        n = await run_in_threadpool(_replace)

        # P1：入库成功后失效答案/检索缓存，避免旧答案残留
        try:
            from api.deps import get_runtime_pipeline as _grp_url
            _grp_url(user_id=user.id).invalidate_answer_cache()
        except Exception:
            pass

        _maybe_trigger_kg_build(user.id, cfg)

        return {
            "url": page["url"],
            "final_url": source,
            "title": title,
            "chars": int(page["chars"]),
            "docs": len(docs),
            "chunks": len(chunks),
            "ingested": n,
            "data_quality": report,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("网页入库失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/v1/ingest/dir", tags=["文档"])
async def ingest_directory(
    dir_path: str = Form(...),
    recursive: bool = Form(True),
    kb_id: Optional[str] = Form(None),
    background: bool = Form(False, description="true=异步入库，立即返回job_id"),
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
        kb = _resolve_kb(user, cfg, kb_id)

        raw_root = resolve_path(cfg.get("paths", {}).get("raw_docs", "data/raw"))
        user_scope = raw_root / user.id[:8]
        target = resolve_path(dir_path)
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"目录不存在: {dir_path}")
        _ensure_within(user_scope, target)

        # P1 异步入库：目录批量同样支持 background=true
        if background:
            from src.ingest_jobs import create_job, submit as _submit_job

            _kb_e, _uid, _cfg_s = kb, user.id, dict(cfg)
            _tgt, _rec = str(target), bool(recursive)
            job_id = create_job("dir", _tgt, {"kb_id": _kb_e["id"]})

            def _bg_dir(reporter) -> Dict[str, Any]:
                reporter(0.1, "parse")
                _split = get_splitter(_cfg_s)
                docs = enrich_directory(
                    _tgt,
                    default_acl=_cfg_s.get("data_quality", {}).get("default_acl", "*"),
                    recursive=_rec)
                if not docs:
                    docs = load_directory(
                        _tgt,
                        pdf_engine=_cfg_s["document_loader"].get("pdf_engine", "pdfplumber"),
                        recursive=_rec,
                        ocr_cfg=_cfg_s.get("document_loader", {}).get("ocr") or {},
                        encoding=_cfg_s["document_loader"].get("encoding", "utf-8"))
                chunks, rep = run_data_pipeline(
                    docs, _cfg_s, _split,
                    decision_provider=_ingest_guard_provider(_cfg_s))
                for ch in chunks:
                    ch.metadata["kb_id"] = _kb_e["id"]
                rep["docs"] = len(docs)
                if not chunks:
                    raise RuntimeError("目录中没有可入库内容")
                reporter(0.6, "upsert")
                _store = TenantAwareFactory.get_chroma_store_by_name(
                    _kb_e["collection_name"], _cfg_s)
                _srcs = sorted({str(c.metadata.get("source")
                                    or c.metadata.get("filepath") or "unknown")
                                for c in chunks})
                n = _store.replace_chunks_for_sources(
                    chunks, _srcs,
                    on_commit=lambda: _commit_ingestion_metadata(
                        _uid, chunks, rep, _kb_e["id"]))
                try:
                    from api.deps import get_runtime_pipeline as _grp2
                    _grp2(user_id=_uid).invalidate_answer_cache()
                except Exception:
                    pass
                _maybe_trigger_kg_build(_uid, _cfg_s)
                return {"dir": _tgt, "docs": len(docs), "chunks": len(chunks),
                        "ingested": n, "kb_id": _kb_e["id"]}

            _submit_job(_bg_dir, job_id)
            return {"job_id": job_id, "status": "queued", "dir": str(target),
                    "kb_id": kb["id"], "poll": f"/api/v1/ingest/jobs/{job_id}"}

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
                    ocr_cfg=cfg.get("document_loader", {}).get("ocr") or {},
                    encoding=cfg["document_loader"].get("encoding", "utf-8"),
                )
            chunks, report = run_data_pipeline(
                docs, cfg, splitter, decision_provider=_ingest_guard_provider(cfg)
            )
            for chunk in chunks:
                chunk.metadata["kb_id"] = kb["id"]
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

        store = TenantAwareFactory.get_chroma_store_by_name(kb["collection_name"], cfg)
        sources = sorted(
            {
                str(c.metadata.get("source") or c.metadata.get("filepath") or "unknown")
                for c in all_chunks
            }
        )

        def _replace() -> int:
            return store.replace_chunks_for_sources(
                all_chunks,
                sources,
                on_commit=lambda: _commit_ingestion_metadata(
                    user.id, all_chunks, report, kb["id"]
                ),
            )

        n = await run_in_threadpool(_replace)

        # P1：入库成功后失效答案/检索缓存，避免旧答案残留
        try:
            from api.deps import get_runtime_pipeline as _grp_dir
            _grp_dir(user_id=user.id).invalidate_answer_cache()
        except Exception:
            pass

        _maybe_trigger_kg_build(user.id, cfg)

        return {
            "dir": str(target),
            "docs": docs_count,
            "chunks": len(all_chunks),
            "ingested": n,
            "kb_id": kb["id"],
            "data_quality": report,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("目录入库失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# =========================== WebSocket Chat ===========================
@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket 流式问答。

    服务不需要登录：连接建立后直接绑定本地用户（见 :mod:`src.middleware.identity`）。

    P1.3：通过 ``LLMConcurrencyGate.semaphore`` 限制并发 LLM 调用，避免
    单实例下被并发 WS 流量击穿。
    """
    from src.middleware.identity import get_local_user

    db = SessionLocal()
    try:
        ws_user = get_local_user(db)
        user_id = ws_user.id
        await websocket.accept()
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            top_k = data.get("top_k", 4)
            session_id = str(data.get("session_id") or "default")
            kb_id = data.get("kb_id")

            if not query:
                await websocket.send_json({"event": "error", "data": "query is required"})
                continue

            # P0-1 修复：使用 LLMConcurrencyGate 替代已删除的 AsyncLLMExecutor
            gate = get_llm_gate()
            started = time.perf_counter()

            # 指定知识库时在该库内检索（与 /api/v1/chat 行为一致）
            kb = None
            if kb_id:
                try:
                    kb = _resolve_kb(ws_user, get_runtime_config(), kb_id)
                except HTTPException as kb_exc:
                    await websocket.send_json(
                        {"event": "error", "data": str(kb_exc.detail)}
                    )
                    continue

            # 长期记忆：显式"记住"指令先行落库（失败/替身缺失均不影响问答）
            try:
                pipeline_probe: RAGPipeline = get_runtime_pipeline(user_id=user_id, kb=kb)
                lt_probe = getattr(pipeline_probe, "long_term_store", None)
                if lt_probe is not None:
                    lt_probe.maybe_remember_from_message(user_id, query)
            except Exception as exc:  # noqa: BLE001
                logger.warning("WS 长期记忆写入失败（已忽略）: %s", exc)

            history = SESSION_MEMORY.get_history(user_id, session_id)
            answer_buf = {"text": ""}
            metadata_buf = {"sources": [], "verification": None, "timings": {}}

            # P1-4：记录时机，用于流式响应的性能元数据
            retrieval_start = time.perf_counter()
            ttft_ms: Optional[float] = None
            first_token_time: Optional[float] = None

            # P1.3：semaphore 保护，避免并发 WS 流量击穿 LLM
            async def _run():
                nonlocal ttft_ms, first_token_time
                pipeline: RAGPipeline = get_runtime_pipeline(user_id=user_id, kb=kb)
                async for event in pipeline.astream_answer(
                    query,
                    top_k=top_k,
                    history=history,
                    user_id=user_id,
                    allowed_groups=_allowed_groups(ws_user),
                ):
                    event_type = event.get("event")
                    
                    # P1-4 修复：收集流式元数据
                    if event_type == "hits":
                        metadata_buf["sources"] = event.get("data", [])
                        metadata_buf["timings"]["retrieval_ms"] = round(
                            (time.perf_counter() - retrieval_start) * 1000, 2
                        )
                    elif event_type == "token":
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                            ttft_ms = round((first_token_time - started) * 1000, 2)
                            metadata_buf["timings"]["ttft_ms"] = ttft_ms
                        answer_buf["text"] += str(event.get("data") or "")
                    elif event_type == "done":
                        metadata_buf["sources"] = event.get("data", [])
                        metadata_buf["verification"] = event.get("verification")
                        # 计算生成总时长
                        if first_token_time:
                            metadata_buf["timings"]["generation_ms"] = round(
                                (time.perf_counter() - first_token_time) * 1000, 2
                            )
                    
                    await websocket.send_json(event)
                
                # P1-4：流式结束后发送汇总元数据事件
                total_ms = round((time.perf_counter() - started) * 1000, 2)
                metadata_buf["timings"]["total_ms"] = total_ms
                
                await websocket.send_json({
                    "event": "metadata",
                    "data": {
                        "sources": metadata_buf["sources"],
                        "verification": metadata_buf["verification"],
                        "timings": metadata_buf["timings"],
                    }
                })

            try:
                # P0-1 修复：使用 LLMConcurrencyGate 的 semaphore
                sem = getattr(gate, "semaphore", None)
                if sem is not None:
                    async with sem:
                        await _run()
                else:
                    await _run()
                # 回写即时记忆
                SESSION_MEMORY.append(user_id, session_id, "user", query)
                SESSION_MEMORY.append(user_id, session_id, "assistant", answer_buf["text"])

                # 会话持久化：session_id 指向真实会话时落库（与 /api/v1/chat 对齐）
                if session_id and session_id != "default" and answer_buf["text"]:
                    try:
                        from src.kb_service import KBError, append_message

                        append_message(db, user_id, session_id, "user", query)
                        append_message(
                            db, user_id, session_id, "assistant", answer_buf["text"]
                        )
                    except KBError:
                        pass
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("WS 会话消息落库失败（已忽略）: %s", exc)
            except Exception as exc:
                logger.error("WebSocket 流式生成失败: %s", exc)
                await websocket.send_json({"event": "error", "data": str(exc)})
    except WebSocketDisconnect:
        logger.info("WebSocket 客户端断开")
    except Exception as exc:
        logger.error("WebSocket 异常: %s", exc)
    finally:
        db.close()
