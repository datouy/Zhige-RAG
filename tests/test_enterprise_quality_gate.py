"""企业质量门禁必须真实失败，且固定语料规模足够。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_ci_quality_gate_is_hard_failure():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "RAG quality hard gate" in workflow
    assert "scripts/ci_eval.py" in workflow
    assert "continue-on-error" not in workflow


def test_eval_dataset_and_corpus_are_versioned():
    corpus = ROOT / "tests/fixtures/eval_corpus"
    assert len(list(corpus.glob("*.md"))) >= 8
    lines = [
        json.loads(line)
        for line in (ROOT / "tests/fixtures/eval_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(lines) >= 10
    assert all(item.get("question") for item in lines)
