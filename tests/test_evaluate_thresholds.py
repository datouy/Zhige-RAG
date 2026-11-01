from argparse import Namespace

from scripts.evaluate import check_thresholds, percentile


def _args(**overrides):
    values = {
        "min_samples": 0,
        "min_hit_rate": None,
        "min_ndcg": None,
        "min_verified_rate": None,
        "max_p95_latency_ms": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 0.5) == 25
    assert percentile([], 0.95) is None


def test_check_thresholds_reports_each_failure():
    summary = {
        "samples": 3,
        "retrieval_hit_rate": 0.5,
        "ndcg_at_5": 0.4,
        "p95_latency_ms": 3000,
        "verifiability": {"verified_rate": 0.6},
    }
    failures = check_thresholds(
        summary,
        _args(
            min_samples=10,
            min_hit_rate=0.8,
            min_ndcg=0.7,
            min_verified_rate=0.9,
            max_p95_latency_ms=2000,
        ),
    )
    assert len(failures) == 5


def test_check_thresholds_passes():
    summary = {
        "samples": 10,
        "retrieval_hit_rate": 0.9,
        "ndcg_at_5": 0.8,
        "p95_latency_ms": 1000,
        "verifiability": {"verified_rate": 0.95},
    }
    assert check_thresholds(
        summary,
        _args(
            min_samples=10,
            min_hit_rate=0.8,
            min_ndcg=0.7,
            min_verified_rate=0.9,
            max_p95_latency_ms=2000,
        ),
    ) == []
