"""Isolated durable ingestion worker.

The API persists jobs and never parses, OCRs, embeds or writes vectors when
``INGEST_EXECUTION_MODE=worker``. A separately managed process claims jobs
under the shared file lock, so a large PDF cannot starve health checks or
request handling in the API workers.
"""
from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.ingest_jobs import (
    claim_next_job,
    execution_mode,
    update_job,
    write_worker_heartbeat,
)
from src.utils import get_logger

logger = get_logger("ingest_worker")

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _reporter(job_id: str) -> Callable[[float, str], None]:
    def report(progress: float, stage: str) -> None:
        update_job(
            job_id,
            status="running",
            progress=max(0.0, min(1.0, float(progress))),
            stage=str(stage),
            owner=WORKER_ID,
        )

    return report


def _runtime(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if cfg is not None:
        return cfg
    from api.deps import get_runtime_config

    return dict(get_runtime_config())


def _kb_from_job(job: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(job.get("meta") or {})
    kb_id = str(meta.get("kb_id") or "")
    collection_name = str(meta.get("collection_name") or "")
    owner_id = str(meta.get("owner_id") or meta.get("user_id") or "")
    if not kb_id or not collection_name or not owner_id:
        raise ValueError("任务缺少 kb_id、collection_name 或 owner_id")
    return {
        "id": kb_id,
        "name": str(meta.get("kb_name") or ""),
        "collection_name": collection_name,
        "owner_id": owner_id,
        "access_role": "editor",
    }


def _store_for(job: Dict[str, Any], cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Any]:
    from src.factories import TenantAwareFactory

    kb = _kb_from_job(job)
    return kb, TenantAwareFactory.get_chroma_store_by_name(
        kb["collection_name"], cfg
    )


def _ingest_file(job: Dict[str, Any], report: Callable[[float, str], None]) -> Dict[str, Any]:
    from api.routes.chat import (
        _ingest_guard_provider,
        _maybe_trigger_kg_build,
        _replace_changed_sources,
    )
    from api.deps import get_splitter
    from src.data_quality import run_data_pipeline
    from src.document_loader import load_document
    from src.document_meta import enrich_file

    cfg = _runtime()
    kb, store = _store_for(job, cfg)
    meta = dict(job.get("meta") or {})
    path = Path(str(meta.get("path") or ""))
    if not path.is_file():
        raise FileNotFoundError("待入库文件不存在或已被清理")

    report(0.1, "parse")
    loader = cfg.get("document_loader") or {}
    ocr_cfg = loader.get("ocr") or {}
    xiaomi_cfg = loader.get("xiaomi") or {}
    docs = enrich_file(
        path,
        default_acl=(cfg.get("data_quality") or {}).get("default_acl", "*"),
        ocr_cfg=ocr_cfg,
        xiaomi_cfg=xiaomi_cfg,
    )
    if not docs:
        docs = load_document(
            path,
            pdf_engine=loader.get("pdf_engine", "pdfplumber"),
            encoding=loader.get("encoding", "utf-8"),
            ocr_cfg=ocr_cfg,
            xiaomi_cfg=xiaomi_cfg,
        )
    chunks, data_report = run_data_pipeline(
        docs,
        cfg,
        get_splitter(cfg),
        decision_provider=_ingest_guard_provider(cfg),
    )
    data_report["docs"] = len(docs)
    if not chunks:
        raise RuntimeError(f"文件解析后没有可用内容: {data_report.get('gate', {})}")
    for chunk in chunks:
        chunk.metadata["kb_id"] = kb["id"]
    report(0.55, "upsert")
    inserted, incremental = _replace_changed_sources(
        store=store,
        user_id=kb["owner_id"],
        kb_id=kb["id"],
        chunks=chunks,
        report=data_report,
    )
    report(0.95, "invalidate")
    try:
        from api.deps import get_runtime_pipeline

        get_runtime_pipeline(user_id=kb["owner_id"], kb=kb).invalidate_answer_cache()
    except Exception:
        logger.debug("worker cache invalidation failed", exc_info=True)
    _maybe_trigger_kg_build(kb["owner_id"], cfg)
    return {
        "filename": str(meta.get("original_name") or path.name),
        "docs": len(docs),
        "chunks": len(chunks),
        "ingested": inserted,
        "changed": incremental["changed"],
        "unchanged": incremental["unchanged"],
        "kb_id": kb["id"],
        "data_quality": data_report,
    }


def _ingest_url(job: Dict[str, Any], report: Callable[[float, str], None]) -> Dict[str, Any]:
    from api.routes.chat import (
        _ingest_guard_provider,
        _maybe_trigger_kg_build,
        _replace_changed_sources,
    )
    from api.deps import get_splitter
    from src.data_quality import run_data_pipeline
    from src.document_loader import Document
    from src.web_loader import fetch_and_extract

    cfg = _runtime()
    kb, store = _store_for(job, cfg)
    meta = dict(job.get("meta") or {})
    url = str(meta.get("url") or "")
    if not url:
        raise ValueError("网页任务缺少 URL")
    url_cfg = (cfg.get("document_loader") or {}).get("url") or {}
    report(0.1, "fetch")
    page = fetch_and_extract(
        url,
        timeout=float(meta.get("timeout") or 15.0),
        max_bytes=int(url_cfg.get("max_bytes", 5 * 1024 * 1024)),
        allow_private=bool(url_cfg.get("allow_private", False)),
        max_chars=int(url_cfg.get("max_chars", 200000)),
    )
    title = str(meta.get("title") or page.get("title") or page["url"]).strip()
    source = page.get("final_url") or page["url"]
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
                "acl": (cfg.get("data_quality") or {}).get("default_acl", "*"),
            },
        )
    ]
    report(0.4, "chunk")
    chunks, data_report = run_data_pipeline(
        docs,
        cfg,
        get_splitter(cfg),
        decision_provider=_ingest_guard_provider(cfg),
    )
    data_report["docs"] = 1
    if not chunks:
        raise RuntimeError("网页正文为空，无法入库")
    for chunk in chunks:
        chunk.metadata["kb_id"] = kb["id"]
    report(0.7, "upsert")
    inserted, incremental = _replace_changed_sources(
        store=store,
        user_id=kb["owner_id"],
        kb_id=kb["id"],
        chunks=chunks,
        report=data_report,
    )
    try:
        from api.deps import get_runtime_pipeline

        get_runtime_pipeline(user_id=kb["owner_id"], kb=kb).invalidate_answer_cache()
    except Exception:
        logger.debug("worker cache invalidation failed", exc_info=True)
    _maybe_trigger_kg_build(kb["owner_id"], cfg)
    return {
        "url": page["url"],
        "final_url": source,
        "title": title,
        "chars": int(page["chars"]),
        "docs": 1,
        "chunks": len(chunks),
        "ingested": inserted,
        "changed": incremental["changed"],
        "unchanged": incremental["unchanged"],
        "kb_id": kb["id"],
    }


def _ingest_dir(job: Dict[str, Any], report: Callable[[float, str], None]) -> Dict[str, Any]:
    from api.routes.chat import (
        _ingest_guard_provider,
        _maybe_trigger_kg_build,
        _replace_changed_sources,
    )
    from api.deps import get_splitter
    from src.data_quality import run_data_pipeline
    from src.document_loader import load_directory
    from src.document_meta import enrich_directory

    cfg = _runtime()
    kb, store = _store_for(job, cfg)
    meta = dict(job.get("meta") or {})
    target = Path(str(meta.get("dir") or ""))
    if not target.is_dir():
        raise FileNotFoundError("批量入库目录不存在")
    loader = cfg.get("document_loader") or {}
    report(0.1, "parse")
    docs = enrich_directory(
        str(target),
        default_acl=(cfg.get("data_quality") or {}).get("default_acl", "*"),
        recursive=bool(meta.get("recursive", True)),
        ocr_cfg=loader.get("ocr") or {},
        xiaomi_cfg=loader.get("xiaomi") or {},
    )
    if not docs:
        docs = load_directory(
            str(target),
            pdf_engine=loader.get("pdf_engine", "pdfplumber"),
            recursive=bool(meta.get("recursive", True)),
            ocr_cfg=loader.get("ocr") or {},
            xiaomi_cfg=loader.get("xiaomi") or {},
            encoding=loader.get("encoding", "utf-8"),
        )
    chunks, data_report = run_data_pipeline(
        docs,
        cfg,
        get_splitter(cfg),
        decision_provider=_ingest_guard_provider(cfg),
    )
    data_report["docs"] = len(docs)
    if not chunks:
        raise RuntimeError("目录中没有可入库内容")
    for chunk in chunks:
        chunk.metadata["kb_id"] = kb["id"]
    report(0.6, "upsert")
    inserted, incremental = _replace_changed_sources(
        store=store,
        user_id=kb["owner_id"],
        kb_id=kb["id"],
        chunks=chunks,
        report=data_report,
    )
    try:
        from api.deps import get_runtime_pipeline

        get_runtime_pipeline(user_id=kb["owner_id"], kb=kb).invalidate_answer_cache()
    except Exception:
        logger.debug("worker cache invalidation failed", exc_info=True)
    _maybe_trigger_kg_build(kb["owner_id"], cfg)
    return {
        "dir": str(target),
        "docs": len(docs),
        "chunks": len(chunks),
        "ingested": inserted,
        "changed": incremental["changed"],
        "unchanged": incremental["unchanged"],
        "kb_id": kb["id"],
        "data_quality": data_report,
    }


def _build_kg(job: Dict[str, Any], report: Callable[[float, str], None]) -> Dict[str, Any]:
    from api.deps import get_runtime_pipeline
    from src.factories import TenantAwareFactory
    from scripts.build_kg import build

    cfg = _runtime()
    meta = dict(job.get("meta") or {})
    owner_id = str(meta.get("owner_id") or meta.get("user_id") or "")
    raw_dir = str(meta.get("dir") or "")
    if not owner_id or not raw_dir:
        raise ValueError("知识图谱任务缺少 owner_id 或目录")
    pipeline = get_runtime_pipeline(user_id=owner_id)
    pipeline.ensure_llm()
    report(0.05, "extract")
    summary = build(
        input_dir=raw_dir,
        llm=pipeline.llm,
        store=TenantAwareFactory.get_kg_store(owner_id, cfg),
    )
    report(1.0, "done")
    return summary


_EXECUTORS = {
    "file": _ingest_file,
    "url": _ingest_url,
    "dir": _ingest_dir,
    "kg_build": _build_kg,
}


def execute_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one claimed job and persist its terminal state."""
    job_id = str(job.get("job_id") or "")
    kind = str(job.get("kind") or "")
    runner = _EXECUTORS.get(kind)
    if not job_id or runner is None:
        raise ValueError(f"不支持的入库任务类型: {kind}")
    report = _reporter(job_id)
    update_job(job_id, status="running", stage="started", progress=0.02, owner=WORKER_ID)
    try:
        result = runner(job, report)
        update_job(
            job_id,
            status="done",
            stage="done",
            progress=1.0,
            result=result,
            error=None,
            owner=WORKER_ID,
        )
        logger.info("隔离入库任务完成 job_id=%s kind=%s", job_id, kind)
        return result
    except Exception as exc:
        logger.exception("隔离入库任务失败 job_id=%s kind=%s", job_id, kind)
        update_job(
            job_id,
            status="failed",
            stage="failed",
            error=f"{type(exc).__name__}: {str(exc)[:1800]}",
            owner=WORKER_ID,
        )
        raise


def run_once() -> bool:
    if execution_mode() != "worker":
        raise RuntimeError("INGEST_EXECUTION_MODE 必须设为 worker 才能启动隔离入库进程")
    job = claim_next_job(WORKER_ID)
    if job is None:
        return False
    execute_job(job)
    return True


def run_forever(poll_seconds: float = 2.0) -> None:
    logger.info("隔离入库 worker 启动 id=%s", WORKER_ID)
    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        while not stop_heartbeat.wait(10.0):
            try:
                write_worker_heartbeat(WORKER_ID)
            except Exception:
                logger.warning("入库 worker 心跳写入失败", exc_info=True)

    write_worker_heartbeat(WORKER_ID)
    heartbeat_thread = threading.Thread(
        target=_heartbeat,
        name="ingest-worker-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    while True:
        try:
            if not run_once():
                time.sleep(max(0.2, float(poll_seconds)))
        except KeyboardInterrupt:
            stop_heartbeat.set()
            logger.info("隔离入库 worker 收到停止信号")
            return
        except Exception:
            logger.exception("隔离入库 worker 本轮失败，稍后继续")
            time.sleep(max(1.0, float(poll_seconds)))


__all__ = ["WORKER_ID", "execute_job", "run_forever", "run_once"]
