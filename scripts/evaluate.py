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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    latencies: List[float] = []
    ttfts: List[float] = []
    tokens_per_sec_values: List[float] = []
    ndcgs: List[float] = []
    mrrs: List[float] = []
    details: List[Dict[str, Any]] = []

    for i, item in enumerate(items, 1):
        q = item["question"]
        ref = item.get("reference_answer", "")
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

        # 关键词覆盖率
        kw_cov = keyword_coverage(result.answer, keywords) if keywords else None
        if kw_cov is not None:
            keyword_scores.append(kw_cov)

        # TTFT & tokens
        timings = getattr(result, "timings", {}) or {}
        ttft_ms = timings.get("ttft_ms")
        if isinstance(ttft_ms, (int, float)):
            ttfts.append(float(ttft_ms))
        tps = timings.get("tokens_per_sec")
        if isinstance(tps, (int, float)) and tps > 0:
            tokens_per_sec_values.append(float(tps))

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
                "ndcg_at_5": ndcg,
                "mrr": mrr_val,
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

    summary: Dict[str, Any] = {
        "samples": n,
        "retrieval_hit_rate": retrieval_hits / n if n else 0.0,
        "avg_keyword_coverage": (
            sum(keyword_scores) / len(keyword_scores) if keyword_scores else None
        ),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "ndcg_at_5": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else None,
        "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else None,
        "avg_tokens_per_sec": (
            sum(tokens_per_sec_values) / len(tokens_per_sec_values) if tokens_per_sec_values else None
        ),
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
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG 评估脚本")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--dataset", default=None, help="覆盖评估集路径")
    p.add_argument("--output", default="data/eval/report.json", help="报告输出路径")
    p.add_argument("--llm-self-eval", action="store_true", help="启用 LLM 自评")
    p.add_argument("--top-k", type=int, default=None, help="覆盖 retrieval.top_k")
    p.add_argument(
        "--synthesize-demo",
        action="store_true",
        help="评估集缺失时，从 data/raw/<source> 合成 5 条样本",
    )
    p.add_argument("--synthesize-from", default="data/raw/rag-intro.md", help="合成使用的 markdown")
    p.add_argument("--no-archive", action="store_true", help="不写历史存档")
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

    print("\n========== 评估报告 ==========")
    print(f"样本数            : {summary['samples']}")
    print(f"检索命中率        : {summary['retrieval_hit_rate']:.2%}")
    cov = summary.get("avg_keyword_coverage")
    print(f"平均关键词覆盖率  : {cov:.2%}" if cov is not None else "平均关键词覆盖率  : —")
    lat = summary.get("avg_latency_ms")
    print(f"平均响应时间      : {lat:.0f} ms" if lat is not None else "平均响应时间      : —")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
