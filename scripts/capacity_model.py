#!/usr/bin/env python
"""知阁容量模型：把 DAU 推导为 HAU、CCU、QPS、生成并发和存储增长。"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# 9 个业务小时。权重和为 1；峰值小时 16%，用于避免把日均流量错误地
# 摊平到全天，也避免用 1 秒尖峰去代表整个容量规划。
HOURLY_WEIGHTS = [0.06, 0.08, 0.11, 0.14, 0.16, 0.14, 0.13, 0.11, 0.07]
WORK_HOURS = len(HOURLY_WEIGHTS)
PEAK_HOUR_WEIGHT = max(HOURLY_WEIGHTS)
MONTH_WORKDAYS = 22
NORMAL_DAYS_PER_MONTH = 18
PEAK_DAYS_PER_MONTH = MONTH_WORKDAYS - NORMAL_DAYS_PER_MONTH
# 即使低流量也保留 4 个模型槽：允许一个请求流式输出、一个超时占用、
# 一个管理/评估请求和一个突发缓冲。这是设计假设，不是供应商能力。
MIN_GENERATION_SLOTS = 4


@dataclass(frozen=True)
class BehaviorProfile:
    key: str
    label: str
    normal_dau: int
    peak_dau: int
    sessions_per_user_day: float
    session_minutes: float
    queries_per_session: float
    chat_ratio: float
    websocket_open_ratio: float
    uploads_per_user_day: float
    chunks_per_doc: int
    avg_doc_mb: float
    ocr_doc_ratio: float
    intra_hour_peak_factor: float
    concurrent_sessions_per_user: float
    generation_seconds: float
    retrieval_seconds: float
    prompt_tokens: int
    completion_tokens: int


PROFILES = {
    profile.key: profile
    for profile in (
        BehaviorProfile(
            key="low",
            label="低参与度",
            normal_dau=500,
            peak_dau=1500,
            sessions_per_user_day=1.4,
            session_minutes=18,
            queries_per_session=6,
            chat_ratio=0.55,
            websocket_open_ratio=0.25,
            uploads_per_user_day=0.02,
            chunks_per_doc=16,
            avg_doc_mb=1.5,
            ocr_doc_ratio=0.10,
            intra_hour_peak_factor=1.15,
            concurrent_sessions_per_user=1.05,
            generation_seconds=2.5,
            retrieval_seconds=0.25,
            prompt_tokens=2500,
            completion_tokens=350,
        ),
        BehaviorProfile(
            key="base",
            label="基准参与度",
            normal_dau=750,
            peak_dau=1750,
            sessions_per_user_day=1.8,
            session_minutes=24,
            queries_per_session=8,
            chat_ratio=0.70,
            websocket_open_ratio=0.40,
            uploads_per_user_day=0.04,
            chunks_per_doc=32,
            avg_doc_mb=3.0,
            ocr_doc_ratio=0.20,
            intra_hour_peak_factor=1.30,
            concurrent_sessions_per_user=1.10,
            generation_seconds=4.0,
            retrieval_seconds=0.35,
            prompt_tokens=3500,
            completion_tokens=500,
        ),
        BehaviorProfile(
            key="high",
            label="高参与度",
            normal_dau=1000,
            peak_dau=2000,
            sessions_per_user_day=2.4,
            session_minutes=32,
            queries_per_session=10,
            chat_ratio=0.80,
            websocket_open_ratio=0.60,
            uploads_per_user_day=0.08,
            chunks_per_doc=60,
            avg_doc_mb=6.0,
            ocr_doc_ratio=0.35,
            intra_hour_peak_factor=1.45,
            concurrent_sessions_per_user=1.15,
            generation_seconds=6.0,
            retrieval_seconds=0.60,
            prompt_tokens=5000,
            completion_tokens=700,
        ),
    )
}


def _ceil(value: float) -> int:
    return max(1, int(math.ceil(value)))


def _finite_mmc_queue(
    arrival_qps: float,
    service_seconds: float,
    servers: int,
    queue_slots: int,
) -> dict[str, Any]:
    """M/M/c/K 稳态指标。

    这里的服务台是生成并发槽位，队列容量对应应用内等待队列。该模型用于
    容量估算，不替代真实压测：LLM 服务时间并非严格指数分布，且 TTFT 与
    全文完成时间也不是同一 SLI。
    """
    c = max(1, int(servers))
    k = max(0, int(queue_slots))
    lam = max(0.0, float(arrival_qps))
    service = max(0.001, float(service_seconds))
    offered = lam * service
    if lam <= 0:
        return {
            "arrival_qps": 0.0,
            "service_seconds": round(service, 3),
            "servers": c,
            "queue_slots": k,
            "offered_load_erlang": 0.0,
            "utilization": 0.0,
            "queue_wait_probability": 0.0,
            "overflow_probability": 0.0,
            "mean_queue_wait_ms": 0.0,
            "p95_queue_wait_ms_approx": 0.0,
            "queue_stable_unbounded": True,
        }

    # 生灭链的未归一化概率。n<=c 时退出率为 n/服务时间，否则为 c/服务时间。
    pmf = [1.0]
    for n in range(1, c + k + 1):
        ratio = offered / n if n <= c else offered / c
        pmf.append(pmf[-1] * ratio)
    total = sum(pmf)
    pmf = [value / total for value in pmf]
    p_wait = sum(pmf[c:]) if c <= c + k else 0.0
    p_overflow = pmf[-1]
    mean_queue_length = sum((n - c) * pmf[n] for n in range(c + 1, c + k + 1))
    accepted_qps = lam * (1.0 - p_overflow)
    mean_wait_seconds = mean_queue_length / accepted_qps if accepted_qps > 0 else 0.0

    utilization = offered / c
    stable = utilization < 1.0
    p95_wait = 0.0
    if stable and p_wait > 0.05:
        # M/M/c 无限队列的等待分位近似；有限队列会丢弃超量请求，因此这里只
        # 作为被接纳请求的悲观参考值，不能当成生产 P95 结论。
        p95_wait = math.log(p_wait / 0.05) * service / (c - offered)

    return {
        "arrival_qps": round(lam, 4),
        "service_seconds": round(service, 3),
        "servers": c,
        "queue_slots": k,
        "offered_load_erlang": round(offered, 3),
        "utilization": round(utilization, 4),
        "queue_wait_probability": round(p_wait, 6),
        "overflow_probability": round(p_overflow, 6),
        "mean_queue_wait_ms": round(mean_wait_seconds * 1000, 2),
        "p95_queue_wait_ms_approx": round(p95_wait * 1000, 2),
        "queue_stable_unbounded": stable,
    }


def _queue_sensitivity(
    arrival_qps: float,
    service_seconds: float,
    design_concurrency: int,
    queue_slots: int,
) -> list[dict[str, Any]]:
    """展示并发槽位和短时突发对排队/丢弃的影响。"""
    candidates = sorted({4, 8, design_concurrency, max(design_concurrency * 2, 1)})
    rows: list[dict[str, Any]] = []
    for load_multiplier in (1.0, 2.0):
        for servers in candidates:
            metrics = _finite_mmc_queue(
                arrival_qps * load_multiplier,
                service_seconds,
                servers,
                queue_slots,
            )
            rows.append(
                {
                    "load_multiplier": load_multiplier,
                    "servers": servers,
                    **metrics,
                }
            )
    return rows


def _compute(profile: BehaviorProfile, dau: int, day_type: str) -> dict[str, Any]:
    sessions_day = dau * profile.sessions_per_user_day
    queries_day = sessions_day * profile.queries_per_session
    session_hours_day = sessions_day * profile.session_minutes / 60

    peak_hour_sessions = sessions_day * PEAK_HOUR_WEIGHT
    peak_hour_queries = queries_day * PEAK_HOUR_WEIGHT
    query_qps = peak_hour_queries / 3600
    chat_qps = query_qps * profile.chat_ratio
    search_qps = query_qps - chat_qps

    peak_hour_average_online = peak_hour_sessions * profile.session_minutes / 60
    ccu_max = peak_hour_average_online * profile.intra_hour_peak_factor
    session_concurrency = ccu_max * profile.concurrent_sessions_per_user
    generation_in_service = chat_qps * profile.generation_seconds
    # 目标并发加入 30% 排队头部空间；2 倍突发由压测场景单独验证。
    design_generation_concurrency = max(
        MIN_GENERATION_SLOTS, _ceil(generation_in_service * 1.30)
    )
    retrieval_in_service = query_qps * 0.6
    design_retrieval_concurrency = _ceil(retrieval_in_service * 1.30)
    generation_queue = _finite_mmc_queue(
        chat_qps,
        profile.generation_seconds,
        design_generation_concurrency,
        queue_slots=64,
    )
    retrieval_queue = _finite_mmc_queue(
        query_qps,
        profile.retrieval_seconds,
        design_retrieval_concurrency,
        queue_slots=128,
    )

    uploads_day = dau * profile.uploads_per_user_day
    uploads_peak_hour = uploads_day * PEAK_HOUR_WEIGHT
    chunks_day = uploads_day * profile.chunks_per_doc
    chunks_peak_hour = uploads_peak_hour * profile.chunks_per_doc
    raw_mb_day = uploads_day * profile.avg_doc_mb
    # float32 向量 + 文本、metadata、HNSW 边和副本的保守占用估算。
    vector_bytes_per_chunk = 768 * 4 + 1800
    vector_mb_day = chunks_day * vector_bytes_per_chunk / 1024 / 1024
    tokens_day = queries_day * profile.chat_ratio * (
        profile.prompt_tokens + profile.completion_tokens
    )
    tokens_peak_second = chat_qps * (
        profile.prompt_tokens + profile.completion_tokens
    )
    input_tokens_peak_second = chat_qps * profile.prompt_tokens
    output_tokens_peak_second = chat_qps * profile.completion_tokens
    monthly_raw_mb = raw_mb_day * (
        NORMAL_DAYS_PER_MONTH if day_type == "normal" else PEAK_DAYS_PER_MONTH
    )

    return {
        "day_type": day_type,
        "dau": dau,
        "sessions_per_day": round(sessions_day, 2),
        "queries_per_day": round(queries_day, 2),
        "hau_workday": round(session_hours_day / WORK_HOURS, 2),
        "peak_hour_average_online": round(peak_hour_average_online, 2),
        "ccu_max": round(ccu_max, 2),
        "session_concurrency_peak": round(session_concurrency, 2),
        "query_qps_peak_hour": round(query_qps, 4),
        "retrieval_qps_peak_hour": round(query_qps, 4),
        "search_qps_peak_hour": round(search_qps, 4),
        "chat_qps_peak_hour": round(chat_qps, 4),
        "generation_in_service": round(generation_in_service, 2),
        "design_generation_concurrency": design_generation_concurrency,
        "design_retrieval_concurrency": design_retrieval_concurrency,
        "generation_queue": generation_queue,
        "retrieval_queue": retrieval_queue,
        "websocket_peak_connections": _ceil(
            ccu_max * profile.websocket_open_ratio
        ),
        "uploads_per_day": round(uploads_day, 2),
        "uploads_peak_hour": round(uploads_peak_hour, 2),
        "ocr_docs_peak_hour": round(
            uploads_peak_hour * profile.ocr_doc_ratio, 2
        ),
        "chunks_peak_hour": round(chunks_peak_hour, 2),
        "raw_storage_mb_day": round(raw_mb_day, 2),
        "vector_storage_mb_day": round(vector_mb_day, 2),
        "model_tokens_per_day": _ceil(tokens_day),
        "model_tokens_peak_second": round(tokens_peak_second, 2),
        "input_tokens_peak_second": round(input_tokens_peak_second, 2),
        "output_tokens_peak_second": round(output_tokens_peak_second, 2),
        "tokens_per_user_day": round(
            tokens_day / dau if dau else 0.0, 2
        ),
        "raw_storage_month_contribution_mb": round(monthly_raw_mb, 2),
    }


def build_model(
    current_docs: int = 1000,
    input_price_per_million: float | None = None,
    output_price_per_million: float | None = None,
) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for key, profile in PROFILES.items():
        normal = _compute(profile, profile.normal_dau, "normal")
        peak = _compute(profile, profile.peak_dau, "peak")
        monthly_queries = (
            normal["queries_per_day"] * NORMAL_DAYS_PER_MONTH
            + peak["queries_per_day"] * PEAK_DAYS_PER_MONTH
        )
        monthly_tokens = (
            normal["model_tokens_per_day"] * NORMAL_DAYS_PER_MONTH
            + peak["model_tokens_per_day"] * PEAK_DAYS_PER_MONTH
        )
        monthly_uploaded_docs = (
            normal["uploads_per_day"] * NORMAL_DAYS_PER_MONTH
            + peak["uploads_per_day"] * PEAK_DAYS_PER_MONTH
        )
        monthly_vector_mb = (
            normal["vector_storage_mb_day"] * NORMAL_DAYS_PER_MONTH
            + peak["vector_storage_mb_day"] * PEAK_DAYS_PER_MONTH
        )
        monthly_raw_mb = (
            normal["raw_storage_month_contribution_mb"]
            + peak["raw_storage_month_contribution_mb"]
        )
        corpus_chunks = current_docs * profile.chunks_per_doc
        corpus_vector_gb = (
            corpus_chunks * (768 * 4 + 1800) / 1024**3
        )
        monthly_input_tokens = monthly_tokens * (
            profile.prompt_tokens / (profile.prompt_tokens + profile.completion_tokens)
        )
        monthly_output_tokens = monthly_tokens - monthly_input_tokens
        monthly_cost: float | None = None
        if (
            input_price_per_million is not None
            and output_price_per_million is not None
        ):
            monthly_cost = (
                monthly_input_tokens / 1_000_000 * input_price_per_million
                + monthly_output_tokens / 1_000_000 * output_price_per_million
            )
        scenarios[key] = {
            "label": profile.label,
            "assumptions": asdict(profile),
            "normal_day": normal,
            "peak_day": peak,
            "monthly": {
                "queries": round(monthly_queries, 0),
                "model_tokens": _ceil(monthly_tokens),
                "input_tokens": _ceil(monthly_input_tokens),
                "output_tokens": _ceil(monthly_output_tokens),
                "uploaded_docs": round(monthly_uploaded_docs, 0),
                "raw_storage_growth_gb": round(monthly_raw_mb / 1024, 3),
                "vector_storage_growth_gb": round(monthly_vector_mb / 1024, 3),
                "model_cost": (
                    round(monthly_cost, 2) if monthly_cost is not None else None
                ),
            },
            "corpus_at_current_docs": {
                "documents": current_docs,
                "chunks": corpus_chunks,
                "vector_estimate_gb": round(corpus_vector_gb, 3),
            },
            "generation_queue_sensitivity_peak_day": _queue_sensitivity(
                peak["chat_qps_peak_hour"],
                profile.generation_seconds,
                peak["design_generation_concurrency"],
                64,
            ),
        }
    return {
        "model_version": "2026-11-27.2",
        "formulas": {
            "HAU_workday": "DAU × sessions/user/day × session_minutes/60 ÷ 9",
            "peak_hour_online": "DAU × sessions/user/day × peak_hour_weight × session_minutes/60",
            "CCU_max": "peak_hour_online × intra_hour_peak_factor",
            "QPS": "DAU × sessions/user/day × queries/session × peak_hour_weight ÷ 3600",
            "generation_in_service": "chat_QPS × average_generation_seconds (Little's Law)",
            "session_concurrency_peak": "CCU_max × 每人同时会话数",
            "WebSocket_peak": "CCU_max × websocket_open_ratio",
            "upload_peak": "DAU × uploads/user/day × peak_hour_weight",
            "queue_wait_probability": "M/M/c/K 生灭链稳态：P(n≥c)",
            "queue_overflow_probability": "M/M/c/K：P(n=c+K)",
            "mean_queue_wait": "Lq ÷ λ_effective (Little's Law)",
            "model_tokens": "chat_queries × (prompt_tokens + completion_tokens)",
            "model_cost": "input_tokens/1e6 × input_price + output_tokens/1e6 × output_price",
            "vector_storage": "chunks × (768×4 bytes + ~1800 bytes metadata/text/index estimate)",
        },
        "assumptions": {
            "work_hours": WORK_HOURS,
            "hourly_weights": HOURLY_WEIGHTS,
            "peak_hour_weight": PEAK_HOUR_WEIGHT,
            "normal_days_per_month": NORMAL_DAYS_PER_MONTH,
            "peak_days_per_month": PEAK_DAYS_PER_MONTH,
            "current_docs": current_docs,
            "vector_dimension": 768,
            "generation_queue_slots": 64,
            "minimum_generation_slots": MIN_GENERATION_SLOTS,
            "retrieval_queue_slots": 128,
            "input_price_per_million": input_price_per_million,
            "output_price_per_million": output_price_per_million,
            "unprovided": [
                "生产文档总量、平均页数、最大单文件与真实日增数据量",
                "用户实际措辞分布、聊天/搜索占比与每次会话提问数",
                "公网带宽、LLM 供应商 RPM/TPM 与单 token 价格",
            ],
        },
        "scenarios": scenarios,
    }


def _markdown(model: dict[str, Any]) -> str:
    lines = [
        "# 知阁容量模型",
        "",
        "DAU 是每日去重用户，不等于 CCU 或 QPS。下表把 DAU 通过会话时长、"
        "请求频率和 9 小时非均匀流量分布换算为并发与吞吐。",
        "",
        "| 场景 | 日型 | DAU | HAU/业务小时均值 | 峰值小时平均在线 | CCU 上限 | 会话并发峰值 | 检索 QPS | Chat QPS | 生成在服务并发 | 设计生成并发 | 生成排队概率 | 设计队列溢出率 | WS 峰值 | 入库文档峰值/小时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("low", "base", "high"):
        scenario = model["scenarios"][key]
        for day in ("normal_day", "peak_day"):
            row = scenario[day]
            day_label = "正常日" if day == "normal_day" else "高峰日"
            lines.append(
                f"| {scenario['label']} | {day_label} | {row['dau']} | "
                f"{row['hau_workday']} | {row['peak_hour_average_online']} | "
                f"{row['ccu_max']} | {row['session_concurrency_peak']} | "
                f"{row['retrieval_qps_peak_hour']} | "
                f"{row['chat_qps_peak_hour']} | {row['generation_in_service']} | "
                f"{row['design_generation_concurrency']} | "
                f"{row['generation_queue']['queue_wait_probability']:.4f} | "
                f"{row['generation_queue']['overflow_probability']:.6f} | "
                f"{row['websocket_peak_connections']} | {row['uploads_peak_hour']} |"
            )
    lines.extend(
        [
            "",
            "## 生成并发排队敏感性",
            "",
            "M/M/c/K 中 c 为生成并发槽位，K=64；下表以各组高峰日流量为基线。2 倍流量用于识别人为配置能否在短突发中保持排队可控，结果仍需压测验证。",
            "",
            "| 场景 | 流量倍率 | 并发槽 c | 利用率 | 等待概率 | 溢出概率 | 平均排队 ms | 近似 P95 排队 ms | 无界队列稳定 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for key in ("low", "base", "high"):
        scenario = model["scenarios"][key]
        for row in scenario["generation_queue_sensitivity_peak_day"]:
            lines.append(
                f"| {scenario['label']} | {row['load_multiplier']:.1f}× | "
                f"{row['servers']} | {row['utilization']:.3f} | "
                f"{row['queue_wait_probability']:.4f} | "
                f"{row['overflow_probability']:.6f} | "
                f"{row['mean_queue_wait_ms']:.1f} | "
                f"{row['p95_queue_wait_ms_approx']:.1f} | "
                f"{'是' if row['queue_stable_unbounded'] else '否'} |"
            )
    lines.extend(
        [
            "",
            "## Token 与成本",
            "",
            "| 场景 | 正常日 token | 高峰日 token | 高峰输入 token/s | 高峰输出 token/s | 每活跃用户日 token | 月度增长（正常/高峰混合） |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("low", "base", "high"):
        scenario = model["scenarios"][key]
        normal = scenario["normal_day"]
        peak = scenario["peak_day"]
        lines.append(
            f"| {scenario['label']} | {normal['model_tokens_per_day']:,} | "
            f"{peak['model_tokens_per_day']:,} | {peak['input_tokens_peak_second']:,} | "
            f"{peak['output_tokens_peak_second']:,} | {peak['tokens_per_user_day']:,.1f} | "
            f"{scenario['monthly']['model_tokens']:,} |"
        )
    assumptions = model["assumptions"]
    if assumptions["input_price_per_million"] is None:
        cost_note = (
            "未提供模型单价，因此不虚构币种或金额。实际月成本按 "
            "`输入 token/1e6 × 输入单价 + 输出 token/1e6 × 输出单价` 代入；"
            "重排、Embedding、OCR、存储、带宽和人力成本另计。"
        )
    else:
        cost_note = (
            "模型成本使用运行时传入的单价计算，JSON 中逐场景给出 `model_cost`；"
            "重排、Embedding、OCR、存储、带宽和人力成本另计。"
        )
    lines.extend(
        [
            "",
            cost_note,
            "",
            "## 存储增长",
            "",
            "| 场景 | 现有文档数 | 估算 chunks | 现有向量数据 GB | 月新增原文 GB | 月新增向量数据 GB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("low", "base", "high"):
        scenario = model["scenarios"][key]
        corpus = scenario["corpus_at_current_docs"]
        monthly = scenario["monthly"]
        lines.append(
            f"| {scenario['label']} | {corpus['documents']:,} | {corpus['chunks']:,} | "
            f"{corpus['vector_estimate_gb']:.3f} | {monthly['raw_storage_growth_gb']:.3f} | "
            f"{monthly['vector_storage_growth_gb']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 公式",
            "",
            *[f"- **{name}**: `{formula}`" for name, formula in model["formulas"].items()],
            "",
            "## 重要边界",
            "",
            "- 这些是设计假设，不是用户提供的历史观测值；上线前必须用生产监控回归校准。",
            "- 9 小时权重为 `6%, 8%, 11%, 14%, 16%, 14%, 13%, 11%, 7%`，峰值小时占全天 16%。",
            "- 设计生成并发含 30% 排队余量；2 倍峰值突发需单独压测，不能由该值自动推导。",
            "- M/M/c/K 假设到达近似 Poisson、服务时间近似指数分布。LLM 的流式输出与长尾请求会偏离该假设；排队结果是设计近似，最终以实测 P50/P95/P99、429/超时和队列深度为准。",
            "- `HAU/业务小时均值` 是把总用户在线时长摊在 9 个业务小时，不能替代真实逐小时监控。",
            "- 模型不假设 DAU=CCU，也不假设 CCU=QPS。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-docs", type=int, default=1000)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()
    model = build_model(
        max(0, args.current_docs),
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
    )
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(model, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown_output:
        Path(args.markdown_output).write_text(_markdown(model), encoding="utf-8")
    print(json.dumps(model, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
