"""评估脚本新增指标的单元测试：NDCG@5、MRR、关键词覆盖率、历史存档。

不依赖真实模型或 LLM；只验证纯函数与本地 IO。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate import (
    INDEX_LIMIT,
    _now_ts,
    archive_report,
    dcg_at_k,
    keyword_coverage,
    mrr,
    ndcg_at_k,
)


# ======================================================================
#  NDCG@5 已知结果
# ======================================================================
def test_ndcg_at_5_all_hit_returns_one():
    """5 个问题全部命中 → NDCG@5 = 1.0。"""
    sources = ["a.md", "b.md", "c.md", "d.md", "e.md"]
    expected = ["a.md"]
    assert ndcg_at_k(sources, expected, k=5) == pytest.approx(1.0, abs=1e-6)


def test_ndcg_at_5_all_miss_returns_zero():
    """5 个问题全部未命中 → NDCG@5 = 0.0。"""
    sources = ["x.md", "y.md", "z.md", "p.md", "q.md"]
    expected = ["a.md"]
    assert ndcg_at_k(sources, expected, k=5) == pytest.approx(0.0, abs=1e-6)


def test_ndcg_at_5_alternating_is_between_zero_and_one():
    """交替命中 → NDCG@5 严格介于 0 和 1 之间。"""
    sources = ["a.md", "x.md", "a.md", "y.md", "a.md"]
    expected = ["a.md"]
    val = ndcg_at_k(sources, expected, k=5)
    assert val is not None
    assert 0.0 < val < 1.0


def test_ndcg_at_5_specific_value():
    """手动计算已知 NDCG@5。

    sources = ["a.md", "x.md", "a.md", "y.md", "a.md"]；expected = ["a.md"]
    命中位置：1, 3, 5 → rel = [1, 0, 1, 0, 1]，其中包含 3 个 1、2 个 0。
    ideal = sorted(rel, reverse=True)[:5] = [1, 1, 1, 0, 0]
    DCG  = 1/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5) + 1/log2(6)
         = 1.0 + 0 + 0.5 + 0 + 0.3869 ≈ 1.8869
    IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) + 0/log2(5) + 0/log2(6)
         = 1.0 + 0.6309 + 0.5 + 0 + 0 ≈ 2.1309
    NDCG ≈ 0.8855
    """
    sources = ["a.md", "x.md", "a.md", "y.md", "a.md"]
    expected = ["a.md"]
    val = ndcg_at_k(sources, expected, k=5)
    assert val == pytest.approx(0.8855, abs=1e-3)


def test_ndcg_at_5_no_expected_sources_returns_none():
    """expected_sources 缺失 → 返回 None，不报错。"""
    assert ndcg_at_k(["a.md"], [], k=5) is None
    assert ndcg_at_k(["a.md"], None, k=5) is None  # type: ignore[arg-type]


# ======================================================================
#  MRR 已知结果
# ======================================================================
def test_mrr_first_position_is_one():
    """第一个位置即命中 → MRR = 1.0。"""
    sources = ["a.md", "b.md", "c.md"]
    expected = ["a.md"]
    assert mrr(sources, expected) == pytest.approx(1.0, abs=1e-6)


def test_mrr_third_position():
    """第一个期望命中位于 rank 3 → MRR = 1/3。"""
    sources = ["x.md", "y.md", "a.md", "b.md"]
    expected = ["a.md"]
    assert mrr(sources, expected) == pytest.approx(1 / 3, abs=1e-6)


def test_mrr_no_match_returns_zero():
    """全部未命中 → MRR = 0.0。"""
    sources = ["x.md", "y.md"]
    expected = ["a.md"]
    assert mrr(sources, expected) == pytest.approx(0.0, abs=1e-6)


def test_mrr_no_expected_sources_returns_none():
    """expected_sources 缺失 → 返回 None。"""
    assert mrr(["a.md"], []) is None
    assert mrr(["a.md"], None) is None  # type: ignore[arg-type]


# ======================================================================
#  关键词覆盖率
# ======================================================================
def test_keyword_coverage_zero_match():
    assert keyword_coverage("回答里完全没有相关字", ["检索", "生成"]) == pytest.approx(0.0)


def test_keyword_coverage_half_match():
    assert keyword_coverage("这里提到了检索", ["检索", "生成"]) == pytest.approx(0.5)


def test_keyword_coverage_full_match():
    assert keyword_coverage("检索与生成的过程", ["检索", "生成"]) == pytest.approx(1.0)


def test_keyword_coverage_empty_keywords_returns_one():
    """空关键词列表 → 1.0（视为"无要求即满足"）。"""
    assert keyword_coverage("随便写点东西", []) == pytest.approx(1.0)


# ======================================================================
#  dcg_at_k 辅助函数
# ======================================================================
def test_dcg_empty_relevances_is_zero():
    assert dcg_at_k([], 5) == pytest.approx(0.0)


def test_dcg_single_rel_at_first():
    """rel=[1] → DCG = 1/log2(2) = 1.0。"""
    assert dcg_at_k([1], 5) == pytest.approx(1.0)


# ======================================================================
#  历史存档机制
# ======================================================================
def test_archive_report_creates_timestamped_file(tmp_path: Path):
    report_path = tmp_path / "report.json"
    summary = {
        "samples": 3,
        "retrieval_hit_rate": 0.66,
        "avg_keyword_coverage": 0.5,
        "avg_latency_ms": 1234.0,
        "ndcg_at_5": 0.7,
        "mrr": 0.8,
        "avg_ttft_ms": 200.0,
        "avg_tokens_per_sec": 50.0,
        "details": [
            {"index": 1, "question": "q1", "answer": "abcdef", "latency_ms": 100.0, "retrieval_hit": True},
        ],
    }
    entry = archive_report(summary, report_path)
    assert (report_path.parent / "reports" / f"{entry['ts']}.json").exists()
    assert entry["samples"] == 3
    assert entry["retrieval_hit_rate"] == 0.66


def test_archive_report_strips_answer_field(tmp_path: Path):
    """存档应去除 details[*].answer 以减小体积。"""
    report_path = tmp_path / "report.json"
    summary = {
        "samples": 1,
        "retrieval_hit_rate": 1.0,
        "details": [
            {"index": 1, "question": "q1", "answer": "很长的回答" * 50, "latency_ms": 10.0},
        ],
    }
    entry = archive_report(summary, report_path)
    archive_file = report_path.parent / "reports" / f"{entry['ts']}.json"
    raw = json.loads(archive_file.read_text(encoding="utf-8"))
    assert "answer" not in raw["details"][0]
    assert raw["details"][0]["question"] == "q1"


def test_archive_report_maintains_index_limit(tmp_path: Path):
    """index.json 中最多保留 INDEX_LIMIT (=50) 条记录。"""
    assert INDEX_LIMIT == 50
    report_path = tmp_path / "report.json"
    summary_base = {
        "samples": 1,
        "retrieval_hit_rate": 1.0,
        "details": [],
    }
    # 写入 55 次，期望只保留最后 50 条
    for i in range(55):
        summary_base["retrieval_hit_rate"] = i / 100.0
        archive_report(summary_base, report_path)

    idx = json.loads((report_path.parent / "reports" / "index.json").read_text(encoding="utf-8"))
    assert len(idx) == INDEX_LIMIT
    # 最后一条应该是第 54 次写入（命中率 0.54）
    assert abs(idx[-1]["retrieval_hit_rate"] - 0.54) < 1e-6


def test_archive_report_filenames_use_timestamp(tmp_path: Path):
    """存档文件名应符合 ``YYYYMMDD-HHMMSS.json`` 格式。"""
    report_path = tmp_path / "report.json"
    entry = archive_report({"samples": 0, "retrieval_hit_rate": 0.0, "details": []}, report_path)
    name = entry["ts"]
    assert len(name) == 15  # YYYYMMDD-HHMMSS
    assert name[8] == "-"
    int(name[:8])
    int(name[9:13])
    int(name[13:15]) or True  # seconds may legitimately be "00"