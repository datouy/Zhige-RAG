"""异步入库任务队列：上传即返回 job_id，后台线程执行解析与入库。

任务状态保存在工作区 JSON 文件中，并通过 ``filelock`` 串行化读取、更新与
清理。这样多 worker 实例可以查询同一个 job_id、查看相同进度，也不会再用
各自的本地快照互相覆盖。耗时任务仍由创建任务的进程执行；进程异常退出时，
状态会在超过 ``INGEST_JOB_STALE_SECONDS`` 后标记为 interrupted。

实现约束：
- 纯标准库 + 项目已有的 ``filelock``，不引入 Celery；
- 有界线程池与排队上限，避免无界任务耗尽内存；
- 终态和每个阶段进度都原子落盘，所有 worker 读取同一份状态；
- 服务正常关闭时，只把本进程拥有的未完成任务标记为 interrupted。
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from filelock import FileLock

from src.utils import get_logger, resolve_path

logger = get_logger("ingest_jobs")

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.RLock()
_FUTURES: Dict[str, Future] = {}
_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()

_STATE_LOCK: Optional[FileLock] = None
_STATE_LOCK_GUARD = threading.Lock()
_OWNER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

_MAX_JOBS = 200
_MAX_PENDING = max(1, int(os.getenv("INGEST_MAX_PENDING", "100") or 100))
_STALE_SECONDS = max(60, int(os.getenv("INGEST_JOB_STALE_SECONDS", "3600") or 3600))
_STATE_LOCK_TIMEOUT = max(
    1.0, float(os.getenv("INGEST_STATE_LOCK_TIMEOUT", "30") or 30)
)
_TERMINAL = {"done", "failed", "interrupted"}


def execution_mode() -> str:
    """Return ``thread`` or ``worker`` for durable ingestion execution.

    ``worker`` is the production path: the API only persists a job, while a
    separately managed process performs parsing, OCR, embedding and upserts.
    This keeps large files from blocking an API worker's event loop.
    """
    mode = os.getenv("INGEST_EXECUTION_MODE", "thread").strip().lower()
    return mode if mode in {"thread", "worker"} else "thread"


def _data_dir() -> Path:
    configured = os.getenv("RAG_DATA_DIR", "").strip()
    if configured:
        return Path(configured).resolve()
    return resolve_path("data")


def _jobs_file() -> Path:
    return _data_dir() / "ingest_jobs.json"


def _heartbeat_file() -> Path:
    return _data_dir() / "ingest_worker_heartbeat.json"


def write_worker_heartbeat(worker_id: str) -> None:
    """Publish a lease for readiness checks of the isolated worker."""
    path = _heartbeat_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"worker_id": worker_id, "updated_at": time.time()}
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def worker_health(max_age_seconds: float = 30.0) -> Dict[str, Any]:
    """Return worker lease state without exposing filesystem paths."""
    mode = execution_mode()
    if mode != "worker":
        return {"status": "not_required", "mode": mode}
    path = _heartbeat_file()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        age = max(0.0, time.time() - float(payload.get("updated_at") or 0))
        healthy = age <= max(1.0, float(max_age_seconds))
        return {
            "status": "healthy" if healthy else "stale",
            "mode": mode,
            "age_seconds": round(age, 1),
        }
    except Exception:
        return {"status": "missing", "mode": mode}


def _state_lock() -> FileLock:
    """同一工作区的所有 worker 共用一个跨进程文件锁。"""
    global _STATE_LOCK
    path = _jobs_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(path.with_name(path.name + ".lock").resolve())
    with _STATE_LOCK_GUARD:
        current = getattr(_STATE_LOCK, "lock_file", None)
        if _STATE_LOCK is None or str(current) != lock_path:
            _STATE_LOCK = FileLock(lock_path, timeout=_STATE_LOCK_TIMEOUT)
        return _STATE_LOCK


def _read_state_unlocked() -> Dict[str, Dict[str, Any]]:
    path = _jobs_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {
            str(job_id): dict(job)
            for job_id, job in data.items()
            if isinstance(job, dict)
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("入库任务状态读取失败（将使用进程内快照）: %s", exc)
        return {}


def _write_state_unlocked(state: Dict[str, Dict[str, Any]]) -> None:
    path = _jobs_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    tmp.write_text(
        json.dumps(state, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _normalize_state(state: Dict[str, Dict[str, Any]]) -> bool:
    """标记超时任务并裁剪终态历史；返回状态是否发生变化。"""
    changed = False
    now = time.time()
    for job in state.values():
        status = str(job.get("status") or "")
        updated = float(job.get("updated_at") or 0)
        if status not in _TERMINAL and now - updated > _STALE_SECONDS:
            job.update(
                status="interrupted",
                stage="interrupted",
                error=f"任务超过 {_STALE_SECONDS} 秒无进度更新，执行进程可能已退出",
                updated_at=now,
            )
            changed = True

    if len(state) > _MAX_JOBS:
        completed = sorted(
            (
                (job_id, job)
                for job_id, job in state.items()
                if job.get("status") in _TERMINAL
            ),
            key=lambda item: float(item[1].get("updated_at") or 0),
        )
        for job_id, _ in completed[: len(state) - _MAX_JOBS]:
            state.pop(job_id, None)
            changed = True
    return changed


def _read_state() -> Dict[str, Dict[str, Any]]:
    with _state_lock():
        state = _read_state_unlocked()
        if _normalize_state(state):
            _write_state_unlocked(state)
        return state


def _persist() -> None:
    """把本进程的较新状态合并到共享文件，绝不覆盖其它 worker 的任务。"""
    with _state_lock():
        state = _read_state_unlocked()
        with _JOBS_LOCK:
            local = {job_id: dict(job) for job_id, job in _JOBS.items()}
        for job_id, job in local.items():
            disk = state.get(job_id)
            if disk is None or float(job.get("updated_at") or 0) >= float(
                disk.get("updated_at") or 0
            ):
                state[job_id] = job
        _normalize_state(state)
        _write_state_unlocked(state)


def _pending_count(state: Dict[str, Dict[str, Any]]) -> int:
    return sum(1 for job in state.values() if job.get("status") not in _TERMINAL)


class IngestQueueFullError(RuntimeError):
    """排队任务数达到上限，拒绝新入库任务（背压）。"""


def _get_executor() -> ThreadPoolExecutor:
    """懒加载线程池；shutdown 后再次调用会重建，避免不可逆关闭。"""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            workers = max(1, int(os.getenv("INGEST_WORKERS", "2") or 2))
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="ingest"
            )
        return _EXECUTOR


def create_job(kind: str, label: str, meta: Optional[Dict[str, Any]] = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    with _state_lock():
        state = _read_state_unlocked()
        _normalize_state(state)
        pending = _pending_count(state)
        if pending >= _MAX_PENDING:
            raise IngestQueueFullError(
                "入库队列已满（%d/%d），请稍后再试" % (pending, _MAX_PENDING)
            )
        if len(state) >= _MAX_JOBS:
            completed = sorted(
                (
                    (jid, job)
                    for jid, job in state.items()
                    if job.get("status") in _TERMINAL
                ),
                key=lambda item: float(item[1].get("updated_at") or 0),
            )
            for jid, _ in completed[: len(state) - _MAX_JOBS + 1]:
                state.pop(jid, None)
            if len(state) >= _MAX_JOBS:
                raise IngestQueueFullError("任务数已达上限，且无可淘汰的已完成任务")

        job = {
            "job_id": job_id,
            "kind": kind,
            "label": label,
            "meta": meta or {},
            "owner": _OWNER_ID,
            "status": "queued",
            "progress": 0.0,
            "stage": "queued",
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        state[job_id] = job
        _write_state_unlocked(state)
    with _JOBS_LOCK:
        _JOBS[job_id] = dict(job)
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    state = _read_state()
    job = state.get(job_id)
    if job is not None:
        return dict(job)
    with _JOBS_LOCK:
        local = _JOBS.get(job_id)
        return dict(local) if local else None


def update_job(job_id: str, **fields: Any) -> bool:
    """Update a durable job; exported for the isolated worker process."""
    _update(job_id, **fields)
    return get_job(job_id) is not None


def claim_next_job(worker_id: str) -> Optional[Dict[str, Any]]:
    """Atomically lease the oldest queued job to one worker.

    The file lock spans read, status transition and atomic replace, so two
    independently started worker processes cannot execute the same job.
    """
    now = time.time()
    with _state_lock():
        state = _read_state_unlocked()
        changed = _normalize_state(state)
        queued = sorted(
            (
                (job_id, job)
                for job_id, job in state.items()
                if str(job.get("status") or "") == "queued"
            ),
            key=lambda item: float(item[1].get("created_at") or 0),
        )
        if not queued:
            if changed:
                _write_state_unlocked(state)
            return None
        job_id, job = queued[0]
        job.update(
            status="running",
            stage="claimed",
            owner=str(worker_id),
            progress=max(0.01, float(job.get("progress") or 0.0)),
            updated_at=now,
        )
        _write_state_unlocked(state)
        claimed = dict(job)
    with _JOBS_LOCK:
        _JOBS[job_id] = dict(claimed)
    return claimed


def list_jobs(limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """返回所有 worker 的最近任务快照，时间倒序。"""
    capped = max(1, min(int(limit), 200))
    state = _read_state()
    jobs = [
        (index, job_id, job)
        for index, (job_id, job) in enumerate(state.items())
    ]
    if status:
        jobs = [item for item in jobs if item[2].get("status") == status]
    jobs.sort(
        key=lambda item: (
            float(item[2].get("created_at") or 0),
            float(item[2].get("updated_at") or 0),
            item[0],
        ),
        reverse=True,
    )
    return [dict(job) for _, _, job in jobs[:capped]]


def _update(job_id: str, **fields: Any) -> None:
    now = time.time()
    with _state_lock():
        state = _read_state_unlocked()
        job = state.get(job_id)
        if job is None:
            with _JOBS_LOCK:
                local = _JOBS.get(job_id)
            if local is None:
                return
            job = dict(local)
            state[job_id] = job
        job.update(fields)
        job["updated_at"] = now
        _normalize_state(state)
        _write_state_unlocked(state)
    with _JOBS_LOCK:
        local = _JOBS.get(job_id)
        if local is not None:
            local.update(fields)
            local["updated_at"] = now


def submit(
    fn: Callable[[Callable[[float, str], None]], Dict[str, Any]],
    job_id: str,
) -> None:
    """把耗时入库函数丢进后台线程池。fn(reporter) -> result dict。"""

    def _report(progress: float, stage: str) -> None:
        try:
            _update(
                job_id,
                progress=max(0.0, min(1.0, float(progress))),
                stage=str(stage),
                status="running",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("入库任务进度写入失败 %s: %s", job_id, exc)

    def _run() -> None:
        _update(job_id, status="running", stage="started", progress=0.05)
        try:
            result = fn(_report)
            _update(
                job_id,
                status="done",
                stage="done",
                progress=1.0,
                result=result,
                error=None,
            )
            logger.info("入库任务完成 %s: %s", job_id, result)
        except Exception as exc:  # noqa: BLE001
            logger.error("入库任务失败 %s: %s", job_id, exc)
            _update(
                job_id,
                status="failed",
                stage="failed",
                error=str(exc)[:2000],
            )
        finally:
            with _JOBS_LOCK:
                _FUTURES.pop(job_id, None)

    _update(job_id, status="queued", owner=_OWNER_ID)
    try:
        with _JOBS_LOCK:
            fut = _get_executor().submit(_run)
            _FUTURES[job_id] = fut
    except Exception as exc:  # noqa: BLE001
        _update(job_id, status="failed", stage="submit_failed", error=str(exc)[:2000])
        raise


def dispatch(
    fn: Callable[[Callable[[float, str], None]], Dict[str, Any]],
    job_id: str,
) -> bool:
    """Run in this process only in development ``thread`` mode.

    Returning ``False`` means the durable job remains queued for the isolated
    ingestion worker. Existing callers can keep a closure as a local fallback
    without making it part of the serialized job record.
    """
    if execution_mode() == "worker":
        return False
    submit(fn, job_id)
    return True


def shutdown(wait: bool = False) -> None:
    """关闭线程池，并只中断本进程拥有的未完成任务。"""
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

    try:
        with _state_lock():
            state = _read_state_unlocked()
            changed = False
            now = time.time()
            for job in state.values():
                if (
                    job.get("owner") == _OWNER_ID
                    and job.get("status") not in _TERMINAL
                ):
                    job.update(
                        status="interrupted",
                        stage="interrupted",
                        error="服务关闭，任务中断；请重新提交",
                        updated_at=now,
                    )
                    changed = True
            if changed:
                _write_state_unlocked(state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("入库任务关闭状态写入失败: %s", exc)

    if executor is not None:
        try:
            executor.shutdown(wait=wait, cancel_futures=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("入库线程池关闭失败: %s", exc)


def stats() -> Dict[str, Any]:
    state = _read_state()
    statuses: Dict[str, int] = {}
    for job in state.values():
        key = str(job.get("status") or "unknown")
        statuses[key] = statuses.get(key, 0) + 1
    result = {
        "total": len(state),
        "pending": _pending_count(state),
        "by_status": statuses,
    }
    try:
        from src.monitoring.metrics import metrics_collector

        metrics_collector.update_ingest_queue(statuses)
    except Exception:
        pass
    return result


def _recover() -> None:
    """启动时读取共享任务表；只中断超过租约时限的遗留任务。"""
    try:
        state = _read_state()
        with _JOBS_LOCK:
            _JOBS.update({job_id: dict(job) for job_id, job in state.items()})
        if state:
            logger.info("已恢复 %d 条入库任务", len(state))
    except Exception as exc:  # noqa: BLE001
        logger.warning("入库任务恢复失败（忽略）: %s", exc)


_recover()


__all__ = [
    "IngestQueueFullError",
    "claim_next_job",
    "create_job",
    "dispatch",
    "execution_mode",
    "get_job",
    "list_jobs",
    "shutdown",
    "stats",
    "submit",
    "update_job",
    "worker_health",
    "write_worker_heartbeat",
]
