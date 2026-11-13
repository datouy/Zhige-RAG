"""异步入库任务队列（P1）：上传即返回 job_id，后台线程做解析→分块→upsert。

为什么需要它：之前 /api/v1/ingest* 全在请求线程里同步跑
「解析→encode→upsert」，50MB 文件直接超时。RAGFlow/MaxKB 都是
「upload→任务ID→后台解析→WS/轮询进度」。

实现约束（零新依赖）：
- 纯 stdlib：uuid + threading + concurrent.futures，不引入 celery/redis；
- 单进程内存队列即可（本项目单机 Chroma，k8s 注明单副本）；
- 进度通过 GET /api/v1/ingest/jobs/{id} 轮询 + 可选 WS 推送；
- 写入路径复用 replace_chunks_for_sources(batch 128)，分批 upsert 防爆炸。

稳定性增强：
- 任务状态持久化到 data/ingest_jobs.json，重启后 queued/running 自动标记 interrupted；
- 线程池懒加载 + shutdown 可重入：应用关闭时取消未开始任务，之后仍可重新调度；
- 排队上限（INGEST_MAX_PENDING）做背压，避免无界堆积打爆内存；
- 淘汰只删已完成任务，绝不静默丢弃运行中/排队中的任务。
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.utils import get_logger, resolve_path

logger = get_logger("ingest_jobs")

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.RLock()
_FUTURES: Dict[str, Future] = {}
_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()
_MAX_JOBS = 200
_MAX_PENDING = int(os.getenv("INGEST_MAX_PENDING", "100") or 100)

_TERMINAL = {"done", "failed", "interrupted"}


def _data_dir() -> Path:
    configured = os.getenv("RAG_DATA_DIR", "").strip()
    if configured:
        return Path(configured).resolve()
    return resolve_path("data")


def _jobs_file() -> Path:
    return _data_dir() / "ingest_jobs.json"


class IngestQueueFullError(RuntimeError):
    """排队任务数达到上限，拒绝新入库任务（背压）。"""


def _get_executor() -> ThreadPoolExecutor:
    """懒加载线程池；shutdown 后再次调用会重建，避免不可逆关闭。"""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")
        return _EXECUTOR


def _persist() -> None:
    """把当前任务状态原子落盘（临时文件 + os.replace）。"""
    try:
        path = _jobs_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _JOBS_LOCK:
            snapshot = {k: dict(v) for k, v in _JOBS.items()}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(snapshot, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("入库任务状态落盘失败（忽略）: %s", exc)


def _pending_count() -> int:
    with _JOBS_LOCK:
        return sum(1 for j in _JOBS.values() if j.get("status") not in _TERMINAL)


def _evict_completed_if_needed(need: int) -> None:
    """只淘汰已完成任务；调用前需持有 _JOBS_LOCK。"""
    completed = sorted(
        [j for j in _JOBS.values() if j.get("status") in _TERMINAL],
        key=lambda j: j.get("updated_at", 0),
    )
    for job in completed[:need]:
        jid = job["job_id"]
        _JOBS.pop(jid, None)
        _FUTURES.pop(jid, None)


def create_job(kind: str, label: str, meta: Optional[Dict[str, Any]] = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    with _JOBS_LOCK:
        pending = _pending_count()
        if pending >= _MAX_PENDING:
            raise IngestQueueFullError(
                "入库队列已满（%d/%d），请稍后再试" % (pending, _MAX_PENDING)
            )
        if len(_JOBS) >= _MAX_JOBS:
            _evict_completed_if_needed(len(_JOBS) - _MAX_JOBS + 1)
            if len(_JOBS) >= _MAX_JOBS:
                raise IngestQueueFullError("任务数已达上限，且无可淘汰的已完成任务")
        _JOBS[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "label": label,
            "meta": meta or {},
            "status": "queued",
            "progress": 0.0,
            "stage": "queued",
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
    _persist()
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _update(job_id: str, **fields: Any) -> None:
    terminal = "status" in fields and fields.get("status") in _TERMINAL
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["updated_at"] = time.time()
    if terminal:
        _persist()


def submit(fn: Callable[[Callable[[float, str], None]], Dict[str, Any]],
           job_id: str) -> None:
    """把耗时入库函数丢进后台线程池。fn(reporter) -> result dict。"""

    def _report(progress: float, stage: str) -> None:
        try:
            _update(
                job_id,
                progress=max(0.0, min(1.0, float(progress))),
                stage=str(stage),
                status="running",
            )
        except Exception:
            pass

    def _run() -> None:
        _update(job_id, status="running", stage="started", progress=0.05)
        try:
            result = fn(_report)
            _update(job_id, status="done", stage="done",
                    progress=1.0, result=result, error=None)
            logger.info("入库任务完成 %s: %s", job_id, result)
        except Exception as exc:  # noqa: BLE001
            logger.error("入库任务失败 %s: %s", job_id, exc)
            _update(job_id, status="failed", stage="failed",
                    error=str(exc)[:2000])
        finally:
            with _JOBS_LOCK:
                _FUTURES.pop(job_id, None)

    _update(job_id, status="queued")
    with _JOBS_LOCK:
        fut = _get_executor().submit(_run)
        _FUTURES[job_id] = fut


def shutdown(wait: bool = False) -> None:
    """关闭线程池：取消尚未开始的任务，落盘最终状态。可重入：之后仍可重新调度。"""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        executor = _EXECUTOR
        _EXECUTOR = None
    with _JOBS_LOCK:
        futures = list(_FUTURES.values())
        _FUTURES.clear()
    for fut in futures:
        if not fut.done():
            fut.cancel()
    if executor is not None:
        try:
            if wait:
                executor.shutdown(wait=True)
            else:
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("入库线程池关闭失败: %s", exc)
    _persist()


def stats() -> Dict[str, Any]:
    with _JOBS_LOCK:
        statuses: Dict[str, int] = {}
        for j in _JOBS.values():
            key = str(j.get("status") or "unknown")
            statuses[key] = statuses.get(key, 0) + 1
        return {
            "total": len(_JOBS),
            "pending": _pending_count(),
            "by_status": statuses,
        }


def _recover() -> None:
    """启动时载入持久化任务；queued/running 标记为 interrupted。"""
    try:
        path = _jobs_file()
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        recovered = 0
        with _JOBS_LOCK:
            for jid, job in data.items():
                if not isinstance(job, dict):
                    continue
                status = job.get("status")
                if status in ("queued", "running"):
                    job["status"] = "interrupted"
                    job["stage"] = "interrupted"
                    job["error"] = "服务重启，任务中断"
                    job["updated_at"] = time.time()
                _JOBS[str(jid)] = job
                recovered += 1
        if recovered:
            logger.info("已恢复 %d 条入库任务（queued/running 已标记 interrupted）", recovered)
            _persist()
    except Exception as exc:  # noqa: BLE001
        logger.warning("入库任务恢复失败（忽略）: %s", exc)


_recover()

