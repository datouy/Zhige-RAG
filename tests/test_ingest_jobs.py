"""异步入库任务队列：持久化、恢复、背压与线程池关闭回归测试。"""
from __future__ import annotations

import time

import pytest

import src.ingest_jobs as ij


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    # 重置模块级状态，隔离各测试
    with ij._JOBS_LOCK:
        ij._JOBS.clear()
        ij._FUTURES.clear()
    monkeypatch.setattr(ij, "_MAX_PENDING", 100)
    monkeypatch.setattr(ij, "_MAX_JOBS", 200)
    yield
    with ij._JOBS_LOCK:
        ij._JOBS.clear()
        ij._FUTURES.clear()


def _wait_done(job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ij.get_job(job_id)["status"] == "done":
            return
        time.sleep(0.02)
    raise AssertionError("任务未在 %ss 内完成" % timeout)


def test_create_and_get_job():
    jid = ij.create_job("file", "a.txt", {"kb_id": "k1"})
    job = ij.get_job(jid)
    assert job is not None
    assert job["status"] == "queued"
    assert job["kind"] == "file"
    assert job["meta"] == {"kb_id": "k1"}


def test_submit_completes_and_persists(tmp_path):
    jid = ij.create_job("file", "b.txt")
    progress = []

    def _work(reporter):
        reporter(0.5, "chunk")
        progress.append(1)
        return {"ingested": 3}

    ij.submit(_work, jid)
    _wait_done(jid)
    job = ij.get_job(jid)
    assert job["status"] == "done"
    assert job["result"] == {"ingested": 3}
    assert progress == [1]
    assert (tmp_path / "ingest_jobs.json").exists()


def test_shutdown_then_resubmit():
    """shutdown 后线程池应可重建，后续任务仍能调度。"""
    ij.shutdown()
    jid = ij.create_job("file", "g.txt")
    ij.submit(lambda reporter: {"ok": True}, jid)
    _wait_done(jid)
    assert ij.get_job(jid)["status"] == "done"


def test_backpressure_rejects_when_pending_full(monkeypatch):
    monkeypatch.setattr(ij, "_MAX_PENDING", 1)
    jid = ij.create_job("file", "c.txt")
    assert jid
    with pytest.raises(ij.IngestQueueFullError):
        ij.create_job("file", "d.txt")


def test_eviction_only_removes_completed(monkeypatch):
    monkeypatch.setattr(ij, "_MAX_JOBS", 3)
    a = ij.create_job("file", "a")
    b = ij.create_job("file", "b")
    c = ij.create_job("file", "c")
    ij._update(a, status="done")
    ij._update(b, status="failed", error="x")
    d = ij.create_job("file", "d")
    assert ij.get_job(c) is not None, "排队中的任务不能被静默淘汰"
    assert ij.get_job(d) is not None


def test_stats_reports_pending():
    ij.create_job("file", "e")
    ij.create_job("file", "f")
    s = ij.stats()
    assert s["total"] == 2
    assert s["pending"] == 2

