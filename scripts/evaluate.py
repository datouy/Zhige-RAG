"""评估脚本：检索命中率、关键词覆盖率、平均响应时间、NDCG@5、MRR、TTFT、tokens/s。

评估数据格式（jsonl，每行一个样本）::

    {
        "question": "什么是 RAG？",
        "reference_answer": "RAG 是检索增强生成……",
        "expected_sources": ["doc1.pdf", "doc2.pdf"],   // 可选
        "expected_keywords": ["检索", "生成"]            // 可选
    }

输出：
- ``data/eval/report.json``          最新一份完整评估（含 details）
- ``data/eval/reports/<TS>.json``    同上的时间戳存档（answer 字段被精简以减小体积）
- ``data/eval/reports/index.json``   最近 50 次评估的精简摘要（samples / hit_rate / kw / latency / ts）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag_pipeline import RAGPipeline
from src.utils import apply_env_overrides, ensure_dir, get_logger, load_config, resolve_path

logger = get_logger("evaluate")


# ======================================================================
#  数据加载与合成
# ======================================================================
def load_eval_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        logger.error("评估数据集不存在: %s", path)
        return []
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    logger.info("加载评估集：%d 条", len(items))
    return items


def synthesize_demo_dataset(md_path: Path, out_path: Path, n: int = 5) -> List[Dict[str, Any]]:
    """从一段 markdown 自动抽样生成 ``n`` 条简单问答对，写入 ``out_path``。

    启发式策略：按 ``## `` 切分章节，对每节用一句话作为 question，从首段抽取关键词作为
    ``expected_keywords``，把 ``Path(md_path).name`` 作为 ``expected_sources``。
    """
    if not md_path.exists():
        raise FileNotFoundError(f"用于合成评估集的 markdown 不存在：{md_path}")

    text = md_path.read_text(encoding="utf-8")
    # 章节切分
    sections: List[Tuple[str, str]] = []
    cur_title = "前言"
    cur_buf: List[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if cur_buf:
                sections.append((cur_title, "\n".join(cur_buf).strip()))
            cur_title = line[3:].strip() or "未命名章节"
            cur_buf = []
        else:
            cur_buf.append(line)
    if cur_buf:
        sections.append((cur_title, "\n".join(cur_buf).strip()))

    if not sections:
        sections = [("正文", text)]

    source_name = md_path.name
    items: List[Dict[str, Any]] = []
    for i, (title, body) in enumerate(sections[:n]):
        # 抽取若干关键词：取最长段中的实词
        first_para = body.split("\n\n", 1)[0] if body else title
        # 简单关键词：CJK 长度 >=2 的连续片段 + 数字/英文
        cjk_words = re.findall(r"[\u4e00-\u9fff]{2,6}", first_para)
        # 去重但保序
        seen, kws = set(), []
        for w in cjk_words:
            if w in seen:
                continue
            seen.add(w)
            kws.append(w)
            if len(kws) >= 5:
                break
        if not kws:
            kws = [title]
        question = f"请用一句话介绍「{title}」的内容。" if i > 0 else "请用一句话介绍本文的主题。"
        items.append(
            {
                "question": question,
                "reference_answer": first_para[:200],
                "expected_sources": [source_name],
                "expected_keywords": kws[:4],
            }
        )

    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    logger.info("已合成 %d 条评估样本 → %s", len(items), out_path)
    return items


# ======================================================================
#  指标实现（纯函数，便于单元测试）
# ======================================================================
def keyword_coverage(prediction: str, keywords: Sequence[str]) -> float:
    """关键词覆盖率 = 命中的关键词数 / 总关键词数；空 keywords 时返回 1.0。"""
    if not keywords:
        return 1.0
    hit = sum(1 for k in keywords if k and k in prediction)
    return hit / len(keywords)


def _char_bigrams(text: str) -> Dict[str, int]:
    """字符 bigram 词袋（无外部依赖的语义相似度代理，RAGAS answer_similarity 轻量版）。"""
    from collections import Counter as _C

    t = (text or "").strip()
    if len(t) < 2:
        return {t: 1} if t else {}
    return dict(_C(t[i:i + 2] for i in range(len(t) - 1)))


def semantic_similarity(a: str, b: str) -> Optional[float]:
    """RAGAS 风格 answer_similarity 轻量版：bigram 余弦 0~1；任一为空返回 None。"""
    if not (a or "").strip() or not (b or "").strip():
        return None
    ca, cb = _char_bigrams(a), _char_bigrams(b)
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def faithfulness_proxy(answer: str, contexts: Sequence[str]) -> Optional[float]:
    """RAGAS faithfulness 轻量版：答案句在上下文中的字面支撑比例。"""
    import re as _re

    sents = [s.strip() for s in _re.split(r"[。！？!?；;\\n]+", (answer or "")) if s.strip()]
    if not sents:
        return None
    ctx = "\n".join(contexts or [])
    if not ctx.strip():
        return None
    hit = 0
    for s in sents:
        key = s[:12] if len(s) > 12 else s
        if key and key in ctx:
            hit += 1
    return hit / len(sents)


def _hits_to_relevance(sources: Sequence[str], expected_sources: Sequence[str]) -> List[int]:
    """把命中来源映射为二元相关性：expected 命中 → 1，否则 → 0。"""
    rel: List[int] = []
    for s in sources:
        src = str(s or "")
        rel.append(1 if any(es in src for es in expected_sources) else 0)
    return rel


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    import math

    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        # 标准公式：rel_i / log2(i+2)
        denom = math.log2(i + 2)
        score += rel / denom
    return score


def ndcg_at_k(sources: Sequence[str], expected_sources: Sequence[str], k: int = 5) -> Optional[float]:
    """NDCG@k。二元相关性，期望序列全 1 作为 IDCG。"""
    if not expected_sources:
        return None
    rel = _hits_to_relevance(sources, expected_sources)
    ideal = sorted(rel, reverse=True)[:k]
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0:
        # 没有任何命中：得分为 0（不算 None）
        return 0.0
    return dcg_at_k(rel, k) / idcg


def mrr(sources: Sequence[str], expected_sources: Sequence[str]) -> Optional[float]:
    """MRR（Mean Reciprocal Rank），首个期望命中位置的倒数。"""
    if not expected_sources:
        return None
    rel = _hits_to_relevance(sources, expected_sources)
    for i, r in enumerate(rel, start=1):
        if r:
            return 1.0 / i
    return 0.0


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """线性插值百分位数；空序列返回 None。"""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * max(0.0, min(1.0, p))
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


# ======================================================================
#  评估主流程
# ======================================================================
@dataclass
class _PerItemStats:
    retrieval_hit: bool
    keyword_coverage: Optional[float]
    latency_ms: float
    ttft_ms: Optional[float]
    tokens_generated: Optional[int]
    tokens_per_sec: Optional[float]
    ndcg_at_5: Optional[float]
    mrr: Optional[float]


def evaluate(pipeline: RAGPipeline, items: List[Dict[str, Any]], llm_self_eval: bool = False) -> Dict[str, Any]:
    n = len(items)
    if n == 0:
        return {"samples": 0}

    retrieval_hits = 0
    keyword_scores: List[float] = []
    sem_sims: List[float] = []
    faiths: List[float] = []
    latencies: List[float] = []
    ttfts: List[float] = []
    tokens_per_sec_values: List[float] = []
    ndcgs: List[float] = []
    mrrs: List[float] = []
    # 可验证性聚合（核心指标：答案必须可回溯到上下文）
    verifications: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    for i, item in enumerate(items, 1):
        q = item["question"]
        expected_sources = item.get("expected_sources", []) or []
        keywords = item.get("expected_keywords", []) or []

        t0 = time.perf_counter()
        try:
            result = pipeline.answer_with_timing(q)
        except Exception as exc:
            logger.error("第 %d 条运行失败: %s", i, exc)
            details.append({"index": i, "question": q, "error": str(exc)})
            continue
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        # 检索命中 + NDCG/MRR
        retrieved_sources = [str(s.get("source") or "") for s in (result.sources or [])]
        hit = 0
        if expected_sources:
            for src in retrieved_sources:
                if any(es in src for es in expected_sources):
                    hit = 1
                    break
        else:
            hit = 1
        retrieval_hits += hit

        ndcg = ndcg_at_k(retrieved_sources, expected_sources, k=5)
        mrr_val = mrr(retrieved_sources, expected_sources)
        if ndcg is not None:
            ndcgs.append(ndcg)
        if mrr_val is not None:
            mrrs.append(mrr_val)

        # 关键词覆盖率 + RAGAS 轻量语义指标（无外部依赖）
        kw_cov = keyword_coverage(result.answer, keywords) if keywords else None
        if kw_cov is not None:
            keyword_scores.append(kw_cov)
        _ref = str(item.get("reference_answer") or "")
        _sim = semantic_similarity(result.answer or "", _ref) if _ref else None
        if _sim is not None:
            sem_sims.append(_sim)
        _ctx_texts = [str(s.get("content") or s.get("snippet") or "") for s in (result.sources or [])]
        _faith = faithfulness_proxy(result.answer or "", _ctx_texts)
        if _faith is not None:
            faiths.append(_faith)

        # TTFT & tokens
        timings = getattr(result, "timings", {}) or {}
        ttft_ms = timings.get("ttft_ms")
        if isinstance(ttft_ms, (int, float)):
            ttfts.append(float(ttft_ms))
        tps = timings.get("tokens_per_sec")
        if isinstance(tps, (int, float)) and tps > 0:
            tokens_per_sec_values.append(float(tps))

        # 可验证性（pipeline 侧规则校验结果）
        verification = getattr(result, "verification", None)
        if verification:
            verifications.append(verification)

        details.append(
            {
                "index": i,
                "question": q,
                "answer": result.answer,
                "latency_ms": round(latency, 2),
                "ttft_ms": round(ttft_ms, 2) if isinstance(ttft_ms, (int, float)) else None,
                "tokens_generated": timings.get("tokens_generated"),
                "tokens_per_sec": round(tps, 2) if isinstance(tps, (int, float)) else None,
                "retrieval_hit": bool(hit),
                "keyword_coverage": kw_cov,
                "answer_similarity": round(_sim, 4) if _sim is not None else None,
                "faithfulness": round(_faith, 4) if _faith is not None else None,
                "ndcg_at_5": ndcg,
                "mrr": mrr_val,
                "verification": verification,
            }
        )
        logger.info(
            "[%d/%d] 命中=%s 覆盖率=%s NDCG=%s MRR=%s 延迟=%.0fms TTFT=%sms",
            i,
            n,
            bool(hit),
            f"{kw_cov:.2f}" if kw_cov is not None else "—",
            f"{ndcg:.2f}" if ndcg is not None else "—",
            f"{mrr_val:.2f}" if mrr_val is not None else "—",
            latency,
            f"{ttft_ms:.0f}" if isinstance(ttft_ms, (int, float)) else "—",
        )

    # 可验证性汇总：verified 比例、平均引用覆盖率、编造事实计数
    answered = [v for v in verifications if not v.get("refused")]
    verified = [v for v in answered if v.get("status") == "verified"]
    summary: Dict[str, Any] = {
        "samples": n,
        "retrieval_hit_rate": retrieval_hits / n if n else 0.0,
        "avg_keyword_coverage": (
            sum(keyword_scores) / len(keyword_scores) if keyword_scores else None
        ),
        # P1-2 RAGAS 语义指标（无外部依赖轻量版；有 ragas 时可用其官方值替换）
        "avg_answer_similarity": (sum(sem_sims) / len(sem_sims) if sem_sims else None),
        "avg_faithfulness": (sum(faiths) / len(faiths) if faiths else None),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "ndcg_at_5": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else None,
        "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else None,
        "p95_ttft_ms": percentile(ttfts, 0.95),
        "avg_tokens_per_sec": (
            sum(tokens_per_sec_values) / len(tokens_per_sec_values) if tokens_per_sec_values else None
        ),
        "verifiability": {
            "checked": len(verifications),
            "verified_rate": (len(verified) / len(answered)) if answered else None,
            "avg_cited_sentence_ratio": (
                sum(v.get("cited_sentence_ratio", 0.0) for v in answered) / len(answered)
            ) if answered else None,
            "unsupported_fact_total": sum(len(v.get("unsupported_facts", [])) for v in verifications),
            "invalid_citation_total": sum(len(v.get("invalid_citations", [])) for v in verifications),
            "refusal_rate": (len(verifications) - len(answered)) / len(verifications)
            if verifications
            else None,
        },
        "details": details,
    }
    return summary


# ======================================================================
#  历史存档
# ======================================================================
INDEX_LIMIT = 50


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _strip_answers(summary: Dict[str, Any]) -> Dict[str, Any]:
    """移除 details[*].answer 以减小存档体积，保留其余字段。"""
    slim = dict(summary)
    details = []
    for d in summary.get("details", []) or []:
        d2 = {k: v for k, v in d.items() if k != "answer"}
        details.append(d2)
    slim["details"] = details
    return slim


def archive_report(summary: Dict[str, Any], report_path: Path) -> Dict[str, Any]:
    """把 ``summary`` 写入时间戳文件 + 更新 ``index.json``。

    Returns:
        精简后的 dict（含 ``ts`` / ``archive_path``），可直接作为索引条目。
    """
    ts = _now_ts()
    archive_dir = report_path.parent / "reports"
    ensure_dir(archive_dir)

    slim = _strip_answers(summary)
    slim["ts"] = ts

    # 1) 时间戳完整报告
    archive_path = archive_dir / f"{ts}.json"
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)

    # 2) 维护 index.json（最近 50 条精简摘要）
    index_path = archive_dir / "index.json"
    index_entries: List[Dict[str, Any]] = []
    if index_path.exists():
        try:
            index_entries = json.loads(index_path.read_text(encoding="utf-8") or "[]")
        except Exception:
            index_entries = []

    entry = {
        "ts": ts,
        "samples": summary.get("samples", 0),
        "retrieval_hit_rate": summary.get("retrieval_hit_rate", 0.0),
        "avg_keyword_coverage": summary.get("avg_keyword_coverage"),
        "avg_latency_ms": summary.get("avg_latency_ms"),
        "ndcg_at_5": summary.get("ndcg_at_5"),
        "mrr": summary.get("mrr"),
        "avg_ttft_ms": summary.get("avg_ttft_ms"),
        "avg_tokens_per_sec": summary.get("avg_tokens_per_sec"),
        "verified_rate": ((summary.get("verifiability") or {}).get("verified_rate")),
        "archive_path": str(archive_path.relative_to(report_path.parent)),
    }
    index_entries.append(entry)
    if len(index_entries) > INDEX_LIMIT:
        index_entries = index_entries[-INDEX_LIMIT:]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_entries, f, ensure_ascii=False, indent=2)

    return entry


# ======================================================================
#  CLI
# ======================================================================
def evaluate_retrieval(pipeline: RAGPipeline, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """检索回归评估：只跑"召回 + 重排"，不加载 LLM、不生成答案。

    用途：UI 占用 GPU 时也能做检索质量回归（CPU 即可），改动分词/分块/
    检索参数后快速看 hit_rate / NDCG / MRR 的涨跌。指标口径与全量评估
    的检索部分一致（expected_sources 子串匹配）。
    """
    n = len(items)
    if n == 0:
        return {"samples": 0}
    hits_total = 0
    ndcgs: List[float] = []
    mrrs: List[float] = []
    latencies: List[float] = []
    details: List[Dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        q = item["question"]
        expected_sources = item.get("expected_sources", []) or []
        t0 = time.perf_counter()
        try:
            hits = pipeline.retrieve(q)
        except Exception as exc:
            logger.error("第 %d 条检索失败: %s", i, exc)
            details.append({"index": i, "question": q, "error": str(exc)})
            continue
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        sources = [str((h.metadata or {}).get("source") or "") for h in hits]
        hit = 0
        if expected_sources:
            for src in sources:
                if any(es in src for es in expected_sources):
                    hit = 1
                    break
        else:
            hit = 1
        hits_total += hit
        ndcg = ndcg_at_k(sources, expected_sources, k=5)
        mrr_val = mrr(sources, expected_sources)
        if ndcg is not None:
            ndcgs.append(ndcg)
        if mrr_val is not None:
            mrrs.append(mrr_val)
        details.append(
            {
                "index": i,
                "question": q,
                "retrieval_hit": bool(hit),
                "ndcg_at_5": ndcg,
                "mrr": mrr_val,
                "latency_ms": round(latency, 2),
                "retrieved_sources": sources[:5],
            }
        )
        logger.info(
            "[%d/%d] 命中=%s NDCG=%s MRR=%s 检索延迟=%.0fms",
            i, n, bool(hit),
            f"{ndcg:.2f}" if ndcg is not None else "—",
            f"{mrr_val:.2f}" if mrr_val is not None else "—",
            latency,
        )
    return {
        "samples": n,
        "mode": "retrieval_only",
        "retrieval_hit_rate": hits_total / n if n else 0.0,
        "ndcg_at_5": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else None,
        "avg_retrieval_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p95_retrieval_latency_ms": percentile(latencies, 0.95),
        "details": details,
    }


def check_thresholds(summary: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    """返回未达标的阈值说明；空列表表示全部通过。"""
    failures: List[str] = []
    samples = int(summary.get("samples", 0) or 0)
    if args.min_samples and samples < args.min_samples:
        failures.append(f"samples {samples} < {args.min_samples}")

    hit_rate = float(summary.get("retrieval_hit_rate", 0.0) or 0.0)
    if args.min_hit_rate is not None and hit_rate < args.min_hit_rate:
        failures.append(f"retrieval_hit_rate {hit_rate:.3f} < {args.min_hit_rate:.3f}")

    ndcg_value = summary.get("ndcg_at_5")
    if args.min_ndcg is not None:
        actual = float(ndcg_value or 0.0)
        if actual < args.min_ndcg:
            failures.append(f"ndcg_at_5 {actual:.3f} < {args.min_ndcg:.3f}")

    verified_rate = (summary.get("verifiability") or {}).get("verified_rate")
    if args.min_verified_rate is not None:
        actual = float(verified_rate or 0.0)
        if actual < args.min_verified_rate:
            failures.append(
                f"verified_rate {actual:.3f} < {args.min_verified_rate:.3f}"
            )

    if args.max_p95_latency_ms is not None:
        latency_value = summary.get("p95_latency_ms")
        if latency_value is None:
            latency_value = summary.get("p95_retrieval_latency_ms")
        actual = float(latency_value or 0.0)
        if actual > args.max_p95_latency_ms:
            failures.append(
                f"p95_latency_ms {actual:.0f} > {args.max_p95_latency_ms:.0f}"
            )
    return failures


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG 评估脚本")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--dataset", default=None, help="覆盖评估集路径")
    p.add_argument("--output", default="data/eval/report.json", help="报告输出路径")
    p.add_argument("--llm-self-eval", action="store_true", help="启用 LLM 自评")
    p.add_argument("--retrieval-only", action="store_true", help="只评估检索（不加载 LLM/GPU，CPU 可跑）")
    p.add_argument("--top-k", type=int, default=None, help="覆盖 retrieval.top_k")
    p.add_argument(
        "--synthesize-demo",
        action="store_true",
        help="评估集缺失时，从 data/raw/<source> 合成 5 条样本",
    )
    p.add_argument("--synthesize-from", default="data/raw/rag-intro.md", help="合成使用的 markdown")
    p.add_argument("--no-archive", action="store_true", help="不写历史存档")
    p.add_argument("--min-samples", type=int, default=0, help="CI：最少样本数")
    p.add_argument("--min-hit-rate", type=float, default=None, help="CI：最低检索命中率")
    p.add_argument("--min-ndcg", type=float, default=None, help="CI：最低 NDCG@5")
    p.add_argument("--min-verified-rate", type=float, default=None, help="CI：最低可验证率")
    p.add_argument("--max-p95-latency-ms", type=float, default=None, help="CI：P95 延迟上限")
    return p


def main():
    args = _build_arg_parser().parse_args()

    cfg = load_config(args.config)
    cfg = apply_env_overrides(cfg)
    if args.top_k:
        cfg.setdefault("retrieval", {})["top_k"] = args.top_k
        cfg.setdefault("rag", {})["top_k"] = args.top_k

    dataset_path = Path(args.dataset or cfg.get("evaluation", {}).get("dataset_path", "data/eval/eval_set.jsonl"))
    dataset_path = resolve_path(dataset_path)

    if not dataset_path.exists() and args.synthesize_demo:
        md_path = resolve_path(args.synthesize_from)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        synthesize_demo_dataset(md_path, dataset_path, n=5)
        logger.info("评估集已生成：%s", dataset_path)

    items = load_eval_dataset(dataset_path)
    if not items:
        logger.error("评估集为空或不存在：%s", dataset_path)
        return 1

    if args.retrieval_only:
        # 检索回归模式：lazy_llm 不加载生成模型（CPU 可跑，可与 UI 并存）
        pipeline = RAGPipeline.from_config(args.config, overrides=cfg, lazy_llm=True)
        summary = evaluate_retrieval(pipeline, items)
    else:
        pipeline = RAGPipeline.from_config(args.config, overrides=cfg)
        summary = evaluate(pipeline, items, llm_self_eval=args.llm_self_eval)

    out = resolve_path(args.output)
    ensure_dir(out.parent)
    # 顶层 report.json：保留所有字段（含 details），并在末尾补一个 ts 方便追溯
    summary_top = dict(summary)
    summary_top["ts"] = _now_ts()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary_top, f, ensure_ascii=False, indent=2)

    if not args.no_archive:
        archive_entry = archive_report(summary, out)
        summary_top["archive_path"] = archive_entry["archive_path"]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary_top, f, ensure_ascii=False, indent=2)

    threshold_failures = check_thresholds(summary, args)

    print("\n========== 评估报告 ==========")
    print(f"样本数            : {summary['samples']}")
    print(f"检索命中率        : {summary['retrieval_hit_rate']:.2%}")
    if summary.get("mode") == "retrieval_only":
        print("(retrieval-only 模式：不含生成侧指标)")
        n5 = summary.get("ndcg_at_5")
        print(f"NDCG@5            : {n5:.3f}" if n5 is not None else "NDCG@5            : —")
        mr = summary.get("mrr")
        print(f"MRR               : {mr:.3f}" if mr is not None else "MRR               : —")
        rl = summary.get("avg_retrieval_latency_ms")
        print(f"平均检索延迟      : {rl:.0f}ms" if rl is not None else "平均检索延迟      : —")
        p95 = summary.get("p95_retrieval_latency_ms")
        print(f"P95 检索延迟      : {p95:.0f}ms" if p95 is not None else "P95 检索延迟      : —")
        print(f"报告已写入        : {out}")
        if threshold_failures:
            print("阈值未通过        : " + "; ".join(threshold_failures))
            return 1
        return 0
    cov = summary.get("avg_keyword_coverage")
    print(f"平均关键词覆盖率  : {cov:.2%}" if cov is not None else "平均关键词覆盖率  : —")
    v = summary.get("verifiability") or {}
    vr = v.get("verified_rate")
    print(f"可验证率          : {vr:.2%}" if vr is not None else "可验证率          : —")
    cr = v.get("avg_cited_sentence_ratio")
    print(f"平均引用覆盖率    : {cr:.2%}" if cr is not None else "平均引用覆盖率    : —")
    rr = v.get("refusal_rate")
    print(f"拒答率            : {rr:.2%}" if rr is not None else "拒答率            : —")
    print(f"编造事实/越界引用 : {v.get('unsupported_fact_total', 0)} / {v.get('invalid_citation_total', 0)}")
    lat = summary.get("avg_latency_ms")
    print(f"平均响应时间      : {lat:.0f} ms" if lat is not None else "平均响应时间      : —")
    p95 = summary.get("p95_latency_ms")
    print(f"P95 响应时间      : {p95:.0f} ms" if p95 is not None else "P95 响应时间      : —")
    ndcg = summary.get("ndcg_at_5")
    print(f"NDCG@5            : {ndcg:.4f}" if ndcg is not None else "NDCG@5            : —")
    mrr_val = summary.get("mrr")
    print(f"MRR               : {mrr_val:.4f}" if mrr_val is not None else "MRR               : —")
    ttft = summary.get("avg_ttft_ms")
    print(f"平均 TTFT         : {ttft:.0f} ms" if ttft is not None else "平均 TTFT         : —")
    tps = summary.get("avg_tokens_per_sec")
    print(f"平均 tokens/s     : {tps:.2f}" if tps is not None else "平均 tokens/s     : —")
    print(f"报告已写入        : {out}")
    if not args.no_archive:
        print(f"历史存档          : data/eval/reports/{summary_top['ts']}.json")
    if threshold_failures:
        print("阈值未通过        : " + "; ".join(threshold_failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
