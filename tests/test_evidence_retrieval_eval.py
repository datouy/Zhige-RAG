"""evidence/chunk 级检索评估回归测试。

测试只使用内存中的假 store/pipeline，不加载 embedding、Chroma 或 LLM。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate import (
    _build_arg_parser,
    build_evidence_chunk_map,
    check_thresholds,
    evaluate,
    evaluate_retrieval,
    evidence_metrics_for_hits,
    load_eval_dataset,
    normalize_eval_text,
)
from src.vector_store import Hit


def _hit(
    chunk_id: str,
    text: str,
    source: str,
    *,
    page: int = 1,
    chunk_index: int = 0,
) -> Hit:
    return Hit(
        id=chunk_id,
        text=text,
        score=1.0,
        metadata={
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
        },
    )


def _evidence(
    evidence_id: str,
    source: str,
    quote: str,
    *,
    page: int | None = None,
    relevance: int = 3,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "source": source,
        "page": page,
        "section": "测试章节",
        "quote": quote,
        "relevance": relevance,
    }


class _FakeStore:
    def __init__(self, grouped: dict[str, list[Hit]]) -> None:
        self.grouped = grouped

    def list_sources(self):
        return [{"source": source, "chunks": len(chunks)} for source, chunks in self.grouped.items()]

    def get_by_sources(self, sources, where=None):
        del where
        return {source: list(self.grouped.get(source, [])) for source in sources}


class _FakePipeline:
    def __init__(self, vector_store: _FakeStore, results: dict[str, list[Hit]]) -> None:
        self.vector_store = vector_store
        self.results = results

    def retrieve(self, question: str, top_k: int = 10):
        return list(self.results.get(question, []))[:top_k]


class _AnswerPipeline(_FakePipeline):
    def __init__(self, vector_store: _FakeStore, results: dict[str, list[Hit]]) -> None:
        super().__init__(vector_store, results)

    def answer_with_timing(self, question: str):
        hits = list(self.results.get(question, []))
        return SimpleNamespace(
            answer="500 元",
            raw_hits=hits,
            sources=[hit.to_dict() for hit in hits],
            timings={},
            verification=None,
        )


def _threshold_args(**overrides) -> Namespace:
    values = {
        "min_samples": 0,
        "min_hit_rate": None,
        "min_ndcg": None,
        "min_recall": None,
        "min_recall_at_5": None,
        "min_recall_at_10": None,
        "min_hit_rate_at_10": None,
        "min_mrr_at_10": None,
        "min_source_coverage": None,
        "min_evidence_coverage": None,
        "min_verified_rate": None,
        "max_p95_latency_ms": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_normalize_eval_text_handles_fullwidth_case_and_whitespace():
    assert normalize_eval_text("  ＧＢ／Ｔ\n 28181 ") == "gb/t 28181"


def test_build_evidence_map_uses_page_and_current_chunk_ids():
    source = "员工手册.md"
    items = [
        {
            "question": "住宿上限是多少？",
            "expected_evidence": [
                _evidence("e1", source, "住宿每晚不超过 500 元", page=3),
                _evidence("e2", source, "总部位于杭州", page=2),
            ],
        }
    ]
    chunks = {
        source: [
            _hit("page-2", "知阁科技总部位于杭州。", source, page=2),
            _hit("page-3", "一线城市住宿每晚不超过 500 元。", source, page=3),
        ]
    }

    mapping, coverage = build_evidence_chunk_map(items, chunks)

    assert mapping[normalize_eval_text(source), 3, "住宿每晚不超过 500 元"] == {"page-3"}
    assert mapping[normalize_eval_text(source), 2, "总部位于杭州"] == {"page-2"}
    assert coverage["coverage"] == 1.0
    assert coverage["unmapped_evidence"] == []


def test_evidence_metrics_recall_hit_rate_and_rank_three():
    metrics = evidence_metrics_for_hits(
        ["noise-1", "noise-2", "gold-a"],
        [{"gold-a"}],
        k=10,
    )
    assert metrics is not None
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1 / 3)


def test_evidence_metrics_respects_k_cutoff_and_unique_evidence():
    gold = [{"gold-a"}, {"gold-b"}]
    metrics = evidence_metrics_for_hits(
        ["gold-a", "gold-a", "gold-b"],
        gold,
        k=2,
    )
    assert metrics is not None
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)


def test_unmapped_quote_is_reported_instead_of_scored_as_model_miss():
    source = "制度.md"
    items = [
        {
            "question": "审批需要谁确认？",
            "expected_sources": [source],
            "expected_evidence": [_evidence("e1", source, "需要二级主管确认")],
        }
    ]
    store = _FakeStore({source: [_hit("c1", "这里没有目标引文。", source)]})
    pipeline = _FakePipeline(store, {items[0]["question"]: [_hit("c1", "这里没有目标引文。", source)]})

    summary = evaluate_retrieval(pipeline, items, top_k=10)

    assert summary["retrieval_hit_rate"] is None
    assert summary["retrieval_evidence"]["coverage"] == 0.0
    assert summary["retrieval_evidence"]["evaluable_samples"] == 0
    assert summary["retrieval_evidence"]["excluded_samples"] == 1
    assert summary["retrieval_evidence"]["unmapped_evidence"][0]["reason"] == "quote_not_found"


def test_invalid_relevance_is_reported_as_dataset_error():
    source = "制度.md"
    item = {
        "question": "测试非法 relevance",
        "expected_evidence": [
            _evidence("e1", source, "有效引文", relevance="high"),  # type: ignore[arg-type]
        ],
    }
    mapping, coverage = build_evidence_chunk_map(
        [item],
        {source: [_hit("c1", "有效引文", source)]},
    )

    assert mapping == {}
    assert coverage["coverage"] == 0.0
    assert coverage["unmapped_evidence"][0]["reason"] == "invalid_relevance"


def test_retrieval_only_uses_evidence_metrics_when_annotations_exist():
    source_a = "员工手册.md"
    source_b = "制度.md"
    items = [
        {
            "question": "住宿上限是多少？",
            "expected_evidence": [_evidence("e1", source_a, "住宿每晚不超过 500 元", page=3)],
        },
        {
            "question": "审批需要谁确认？",
            "expected_evidence": [_evidence("e2", source_b, "需要二级主管确认")],
        },
    ]
    store = _FakeStore(
        {
            source_a: [_hit("a-gold", "住宿每晚不超过 500 元。", source_a, page=3)],
            source_b: [_hit("b-gold", "报销需要二级主管确认。", source_b)],
        }
    )
    pipeline = _FakePipeline(
        store,
        {
            "住宿上限是多少？": [
                _hit("noise", "同文档其他段落。", source_a, page=2),
                _hit("a-gold", "住宿每晚不超过 500 元。", source_a, page=3),
            ],
            "审批需要谁确认？": [
                _hit("b-gold", "报销需要二级主管确认。", source_b),
            ],
        },
    )

    summary = evaluate_retrieval(pipeline, items, top_k=10)

    assert summary["retrieval_evidence"]["coverage"] == 1.0
    assert summary["retrieval_evidence"]["evaluable_samples"] == 2
    assert summary["hit_rate_at_10"] == pytest.approx(1.0)
    assert summary["recall_at_10"] == pytest.approx(1.0)
    assert summary["mrr_at_10"] == pytest.approx((0.5 + 1.0) / 2)


def test_legacy_expected_sources_dataset_remains_file_level():
    item = {
        "question": "旧格式问题",
        "expected_sources": ["legacy.md"],
    }
    store = _FakeStore({"legacy.md": []})
    pipeline = _FakePipeline(
        store,
        {item["question"]: [_hit("legacy", "旧格式答案", "legacy.md")]},
    )

    summary = evaluate_retrieval(pipeline, [item], top_k=10)

    assert summary["retrieval_evidence"]["mode"] == "file"
    assert summary["retrieval_hit_rate"] == 1.0
    assert summary["recall_at_10"] == 1.0
    assert summary["mrr"] == 1.0


def test_full_evaluate_uses_same_evidence_metric_alias():
    source = "员工手册.md"
    item = {
        "question": "住宿上限是多少？",
        "expected_evidence": [_evidence("e1", source, "住宿每晚不超过 500 元", page=3)],
    }
    gold = _hit("gold", "住宿每晚不超过 500 元。", source, page=3)
    pipeline = _AnswerPipeline(_FakeStore({source: [gold]}), {item["question"]: [gold]})

    summary = evaluate(pipeline, [item])

    assert summary["retrieval_evidence"]["mode"] == "evidence"
    assert summary["retrieval_hit_rate"] == 1.0
    assert summary["recall_at_10"] == 1.0
    assert summary["mrr_at_10"] == 1.0


def test_new_thresholds_and_evidence_coverage_fail_closed():
    summary = {
        "samples": 2,
        "retrieval_hit_rate": 0.5,
        "ndcg_at_5": 0.6,
        "recall_at_5": 0.5,
        "recall_at_10": 0.75,
        "hit_rate_at_10": 0.5,
        "mrr_at_10": 0.25,
        "retrieval_evidence": {"mode": "evidence", "coverage": 0.8},
    }
    failures = check_thresholds(
        summary,
        _threshold_args(
            min_recall_at_5=0.6,
            min_recall_at_10=0.8,
            min_hit_rate_at_10=0.9,
            min_mrr_at_10=0.5,
            min_evidence_coverage=0.95,
        ),
    )
    assert len(failures) == 5


def test_evidence_coverage_gate_rejects_file_only_dataset():
    failures = check_thresholds(
        {
            "samples": 1,
            "retrieval_hit_rate": 1.0,
            "retrieval_evidence": {"mode": "file", "coverage": None},
        },
        _threshold_args(min_evidence_coverage=1.0),
    )
    assert any("dataset has no expected_evidence" in failure for failure in failures)


def test_cli_exposes_evidence_gate_options():
    args = _build_arg_parser().parse_args(
        [
            "--min-recall-at-5",
            "0.9",
            "--min-recall-at-10",
            "0.95",
            "--min-hit-rate-at-10",
            "0.98",
            "--min-mrr-at-10",
            "0.8",
            "--min-evidence-coverage",
            "1",
        ]
    )
    assert args.min_recall_at_5 == 0.9
    assert args.min_recall_at_10 == 0.95
    assert args.min_hit_rate_at_10 == 0.98
    assert args.min_mrr_at_10 == 0.8
    assert args.min_evidence_coverage == 1.0


def test_checked_in_template_has_valid_unique_evidence_annotations():
    path = Path(__file__).resolve().parent.parent / "data/eval/templates/evidence_retrieval.jsonl"
    items = load_eval_dataset(path)

    assert len(items) == 11
    assert len({item["id"] for item in items}) == len(items)
    assert all(item.get("expected_evidence") for item in items)
    assert all(
        evidence.get("source")
        and evidence.get("quote")
        and evidence.get("relevance", 0) >= 2
        for item in items
        for evidence in item["expected_evidence"]
    )
