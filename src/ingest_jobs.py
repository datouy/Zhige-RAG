"""异步入库任务队列（P1）：上传即返回 job_id，后台线程做解析→分块→upsert。

为什么需要它：之前 /api/v1/ingest* 全在请求线程里同步跑
``解析→encode→upsert``，50MB 文件直接超时。RAGFlow/MaxKB 都是
"upload→任务ID→后台解析→WS/轮询进度"。

实现约束（零新依赖）：
- 纯 stdlib：``uuid + threading + concurrent.futures``，不引入 celery/redis；
- 单进程内存队列即可（本项目单机 Chroma，k8s 注明单副本）；
- 进度通过 ``GET /api/v1/ingest/jobs/{id}`` 轮询 + 可选 WS 推送；
- 写入路径复用 ``replace_chunks_for_sources(batch 128)``，分批 upsert 防爆炸。
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from src.utils import get_logger

logger = get_logger("ingest_jobs")

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")
_MAX_JOBS = 200


def create_job(kind: str, label: str, meta: Optional[Dict[str, Any]] = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    with _JOBS_LOCK:
        if len(_JOBS) >= _MAX_JOBS:
            # 淘汰最老已完成任务
            oldest = sorted(_JOBS.items(), key=lambda kv: kv[1].get("updated_at", 0))
            for jid, _ in oldest[: len(_JOBS) - _MAX_JOBS + 1]:
                _JOBS.pop(jid, None)
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
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _update(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["updated_at"] = time.time()


def submit(fn: Callable[[Callable[[float, str], None]], Dict[str, Any]],
           job_id: str) -> None:
    """把耗时入库函数丢进后台线程池。fn(reporter) -> result dict。"""

    def _report(progress: float, stage: str) -> None:
        try:
            _update(job_id, progress=max(0.0, min(1.0, float(progress))),
                    stage=str(stage), status="running")
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

    _update(job_id, status="queued")
    _EXECUTOR.submit(_run)
