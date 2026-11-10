"""KG 入库自动建图（P1-6）单元测试。

覆盖 api.routes.chat._maybe_trigger_kg_build 的关键分支：
- 知识图谱关闭 / auto_build 关闭时静默跳过（不加载 pipeline）；
- extractive（无 LLM）模式跳过；
- LLM 未就绪跳过；
- LLM 就绪时提交 kg_build 后台任务。
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api.routes.chat as chat_module  # noqa: E402


class _FakePipeline:
    """仅覆盖 KG 自动构建探测所需的接口。"""

    def __init__(self, extractive: bool, llm=None):
        self._extractive = extractive
        self.llm = llm

    def _is_extractive_mode(self) -> bool:
        return self._extractive

    def ensure_llm(self) -> None:
        pass


def _cfg(enabled: bool, auto: bool) -> dict:
    return {
        "knowledge_graph": {"enabled": enabled, "auto_build_on_ingest": auto},
        "paths": {"raw_docs": "data/raw"},
    }


def _forbid_pipeline(*args, **kwargs):
    raise AssertionError("该分支不应加载 runtime pipeline")


def test_noop_when_kg_disabled(monkeypatch):
    monkeypatch.setattr(chat_module, "get_runtime_pipeline", _forbid_pipeline)
    assert chat_module._maybe_trigger_kg_build("12345678", _cfg(False, True)) is None


def test_noop_when_auto_build_disabled(monkeypatch):
    monkeypatch.setattr(chat_module, "get_runtime_pipeline", _forbid_pipeline)
    assert chat_module._maybe_trigger_kg_build("12345678", _cfg(True, False)) is None


def test_skip_in_extractive_mode(monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "get_runtime_pipeline",
        lambda user_id=None: _FakePipeline(extractive=True),
    )
    assert chat_module._maybe_trigger_kg_build("12345678", _cfg(True, True)) is None


def test_skip_when_llm_not_ready(monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "get_runtime_pipeline",
        lambda user_id=None: _FakePipeline(extractive=False, llm=None),
    )
    assert chat_module._maybe_trigger_kg_build("12345678", _cfg(True, True)) is None


def test_submits_job_when_llm_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(
        chat_module,
        "get_runtime_pipeline",
        lambda user_id=None: _FakePipeline(extractive=False, llm=object()),
    )
    raw_dir = tmp_path / "12345678"
    raw_dir.mkdir()
    monkeypatch.setattr(chat_module, "resolve_path", lambda p: tmp_path)

    import src.ingest_jobs as ingest_jobs

    submitted = {}

    def fake_create(kind, label, meta=None):
        submitted["kind"] = kind
        return "job-1"

    def fake_submit(fn, job_id):
        submitted["job_id"] = job_id

    monkeypatch.setattr(ingest_jobs, "create_job", fake_create)
    monkeypatch.setattr(ingest_jobs, "submit", fake_submit)

    chat_module._maybe_trigger_kg_build("12345678", _cfg(True, True))

    assert submitted == {"kind": "kg_build", "job_id": "job-1"}
