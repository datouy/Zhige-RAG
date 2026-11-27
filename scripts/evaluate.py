"""评估脚本：evidence 级 Recall/HitRate/MRR、关键词覆盖率、延迟与生成侧指标。

评估数据格式（jsonl，每行一个样本）::

    {
        "question": "什么是 RAG？",
        "reference_answer": "RAG 是检索增强生成……",
        "expected_sources": ["doc1.pdf", "doc2.pdf"],   // 可选
        "expected_evidence": [                           // 可选；推荐
            {
                "evidence_id": "e1",
                "source": "doc1.pdf",
                "page": 3,
                "section": "2. 权限模型",
                "quote": "管理员可以查看审计日志。",
                "relevance": 3
            }
        ],
        "expected_keywords": ["检索", "生成"]            // 可选
    }

有 ``expected_evidence`` 时按 evidence/chunk 级计算 Recall、Hit Rate 与 MRR；
没有时继续按文件级计算，保证旧评估集完全兼容。

输出：
- ``data/eval/report.json``          最新一份完整评估（含 details）
- ``data/eval/reports/<TS>.json``    同上的时间戳存档（answer 字段被精简以减小体积）
- ``data/eval/reports/index.json``   最近 50 次评估的精简摘要（samples / hit_rate / kw / latency / ts）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag_pipeline import RAGPipeline
from src.utils import ensure_dir, get_logger, resolve_path
from src.verifier import is_refusal_answer

logger = get_logger("evaluate")


def load_eval_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """按 API 相同优先级加载配置（base → local_overrides → 环境变量）。"""
    from api.deps import get_cached_config

    return get_cached_config(str(resolve_path(path)))


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


def item_expects_refusal(item: Dict[str, Any]) -> bool:
    """Whether a dataset item intentionally asks the system to decline."""
    return item.get("expect_refusal") is True


def _result_is_refusal(result: Any) -> bool:
    verification = getattr(result, "verification", None)
    if isinstance(verification, dict) and bool(verification.get("refused")):
        return True
    if verification is not None and bool(getattr(verification, "refused", False)):
        return True
    return is_refusal_answer(str(getattr(result, "answer", "") or ""))


def audit_eval_dataset(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize coverage and structural defects before trusting a score.

    A high HitRate on eight easy questions is not evidence of product quality.
    The report surfaces difficulty, query type, duplicate IDs/questions and
    answerable/refusal balance so CI can require a representative minimum.
    """
    difficulties = Counter(
        str(item.get("difficulty") or "unspecified").strip().lower()
        for item in items
    )
    query_types = Counter(
        str(item.get("query_type") or "unspecified").strip()
        for item in items
    )
    questions = [str(item.get("question") or "").strip() for item in items]
    nonempty_questions = [question for question in questions if question]
    ids = [str(item.get("id") or "").strip() for item in items]
    nonempty_ids = [item_id for item_id in ids if item_id]
    refusal_samples = sum(1 for item in items if item_expects_refusal(item))
    answerable_samples = len(items) - refusal_samples

    unscored: List[Dict[str, Any]] = []
    invalid_refusals: List[Dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        has_gold = bool(item.get("expected_sources")) or bool(
            item.get("expected_evidence")
        )
        if item_expects_refusal(item):
            if has_gold:
                invalid_refusals.append(
                    {
                        "index": index,
                        "question": item.get("question"),
                        "reason": "refusal_item_has_expected_gold",
                    }
                )
        elif not has_gold:
            unscored.append(
                {
                    "index": index,
                    "question": item.get("question"),
                    "reason": "answerable_item_has_no_expected_sources_or_evidence",
                }
            )

    duplicate_questions = sorted(
        question for question, count in Counter(nonempty_questions).items() if count > 1
    )
    duplicate_ids = sorted(
        item_id for item_id, count in Counter(nonempty_ids).items() if count > 1
    )
    missing_id = len(items) - len(nonempty_ids)
    return {
        "samples": len(items),
        "answerable_samples": answerable_samples,
        "refusal_samples": refusal_samples,
        "difficulty": dict(difficulties),
        "query_types": dict(query_types),
        "missing_difficulty": difficulties.get("unspecified", 0),
        "missing_question": sum(1 for question in questions if not question),
        "missing_id": missing_id,
        "duplicate_questions": duplicate_questions,
        "duplicate_ids": duplicate_ids,
        "unscored_answerable_samples": unscored,
        "invalid_refusal_samples": invalid_refusals,
        "valid": not (
            any(not question for question in questions)
            or missing_id
            or len(set(nonempty_ids)) != len(nonempty_ids)
            or duplicate_questions
            or unscored
            or invalid_refusals
        ),
    }


def normalize_eval_text(value: Any) -> str:
    """统一引文与 chunk 文本，避免全半角、英文大小写和空白造成假失配。"""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.casefold().split())


def _evidence_relevance(evidence: Dict[str, Any]) -> Optional[float]:
    """缺省 relevance 视为 2；非法值返回 None 并进入数据质量报告。"""
    try:
        return float(evidence.get("relevance", 2))
    except (TypeError, ValueError):
        return None


def _evidence_page(value: Any) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def evidence_anchor(
    evidence: Dict[str, Any],
) -> Optional[Tuple[str, Optional[int], str]]:
    """返回可稳定重定位的 evidence 锚点；无效标注返回 None。"""
    if not isinstance(evidence, dict):
        return None
    source = str(evidence.get("source") or "").strip().replace("\\", "/")
    quote = normalize_eval_text(evidence.get("quote"))
    if not source or not quote:
        return None
    try:
        page = _evidence_page(evidence.get("page"))
    except (TypeError, ValueError):
        return None
    return source, page, quote


def _relevant_evidence(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    relevant: List[Dict[str, Any]] = []
    for evidence in item.get("expected_evidence") or []:
        if not isinstance(evidence, dict):
            continue
        relevance = _evidence_relevance(evidence)
        if relevance is not None and relevance >= 2:
            relevant.append(evidence)
    return relevant


def dataset_uses_evidence(items: Sequence[Dict[str, Any]]) -> bool:
    return any(
        bool(item.get("expected_evidence"))
        for item in items
        if not item_expects_refusal(item)
    )


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


def recall_at_k(
    sources: Sequence[str], expected_sources: Sequence[str], k: int = 10
) -> Optional[float]:
    """Recall@k：前 k 个结果覆盖了多少个唯一期望来源。

    文件级评测无法判断同一文档内多块之间的增益，因此按唯一来源计数。
    重复命中同一文档不会虚增分母；``expected_sources`` 为空时返回 None。
    """
    expected = {str(item) for item in (expected_sources or []) if str(item)}
    if not expected:
        return None
    returned = {str(item or "") for item in list(sources or [])[: max(0, int(k))]}
    matched = sum(
        1
        for exp in expected
        if any(exp in source for source in returned)
    )
    return matched / len(expected)


def _chunk_id_text_metadata(chunk: Any) -> Tuple[str, str, Dict[str, Any]]:
    if isinstance(chunk, dict):
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
        text = str(chunk.get("text") or chunk.get("document") or chunk.get("content") or "")
        metadata = dict(chunk.get("metadata") or {})
        return chunk_id, text, metadata
    metadata = dict(getattr(chunk, "metadata", None) or {})
    return str(getattr(chunk, "id", "") or ""), str(getattr(chunk, "text", "") or ""), metadata


def _ranked_hit_ids(hits: Sequence[Any]) -> List[str]:
    ids: List[str] = []
    for hit in hits:
        chunk_id, _, _ = _chunk_id_text_metadata(hit)
        if chunk_id:
            ids.append(chunk_id)
    return ids


def evidence_metrics_for_hits(
    hit_ids: Sequence[str],
    gold_chunk_ids: Sequence[Sequence[str]],
    k: int,
) -> Optional[Dict[str, Any]]:
    """计算单个 query 的 evidence 级 Recall@K、HitRate@K 与 MRR@K。

    ``gold_chunk_ids`` 的每个元素对应一条相关 evidence；一条引文映射到多个
    当前 chunk 时，命中其中任意一个都算该 evidence 被召回。
    """
    groups = [set(group) for group in gold_chunk_ids if group]
    if not groups:
        return None
    cutoff = max(0, int(k))
    ranked = list(hit_ids or [])[:cutoff]
    retrieved = set(ranked)
    matched = sum(1 for group in groups if group & retrieved)
    first_rank: Optional[int] = None
    for rank, chunk_id in enumerate(ranked, 1):
        if any(chunk_id in group for group in groups):
            first_rank = rank
            break
    return {
        "recall": matched / len(groups),
        "hit_rate": 1.0 if matched else 0.0,
        "mrr": (1.0 / first_rank) if first_rank is not None else 0.0,
    }


def _resolve_evidence_source(requested: str, available: Sequence[str]) -> str:
    normalized = requested.strip().replace("\\", "/").strip("/")
    exact = [source for source in available if source.replace("\\", "/") == normalized]
    if exact:
        return exact[0]
    suffix_matches = [
        source
        for source in available
        if source.replace("\\", "/").strip("/").endswith("/" + normalized)
    ]
    return suffix_matches[0] if len(suffix_matches) == 1 else requested


def load_evidence_chunks(
    vector_store: Any,
    sources: Sequence[str],
) -> Tuple[Dict[str, List[Any]], Dict[str, str]]:
    """从当前 collection 批量读取 evidence 来源的所有可检索 chunk。"""
    requested = sorted({str(source).strip().replace("\\", "/") for source in sources if source})
    if not requested:
        return {}, {}

    available: List[str] = []
    try:
        rows = vector_store.list_sources()
        available = [str(row.get("source") or "") for row in rows if row.get("source")]
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法读取 collection 来源列表，将按原 source 查询：%s", exc)

    aliases = {source: _resolve_evidence_source(source, available) for source in requested}
    actual_sources = sorted(set(aliases.values()))
    if hasattr(vector_store, "get_by_sources"):
        grouped = vector_store.get_by_sources(actual_sources)
    elif hasattr(vector_store, "get_by_source"):
        grouped = {
            source: vector_store.get_by_source(source)
            for source in actual_sources
        }
    else:
        raise TypeError("vector_store 需要 get_by_sources 或 get_by_source")

    normalized_grouped: Dict[str, List[Any]] = {}
    for source, chunks in (grouped or {}).items():
        normalized_grouped[str(source).replace("\\", "/")] = list(chunks or [])
    return {
        requested_source: normalized_grouped.get(actual.replace("\\", "/"), [])
        for requested_source, actual in aliases.items()
    }, aliases


def build_evidence_chunk_map(
    items: Sequence[Dict[str, Any]],
    chunks_by_source: Dict[str, Sequence[Any]],
) -> Tuple[Dict[Tuple[str, Optional[int], str], set[str]], Dict[str, Any]]:
    """把标注引文映射到当前 collection 的 chunk ID。

    只使用实际可检索的 chunk.text，不使用 metadata.parent_text，避免把不可
    直接召回的父文本误算成正确答案。
    """
    rows: List[Tuple[int, Dict[str, Any], Tuple[str, Optional[int], str]]] = []
    invalid_rows: List[Dict[str, Any]] = []
    for item_index, item in enumerate(items, 1):
        for evidence_index, raw in enumerate(item.get("expected_evidence") or [], 1):
            if not isinstance(raw, dict):
                invalid_rows.append(
                    {
                        "query_index": item_index,
                        "question": item.get("question"),
                        "evidence_index": evidence_index,
                        "reason": "evidence_not_object",
                    }
                )
                continue
            relevance = _evidence_relevance(raw)
            if relevance is None:
                invalid_rows.append(
                    {
                        "query_index": item_index,
                        "question": item.get("question"),
                        "evidence_index": evidence_index,
                        "evidence_id": raw.get("evidence_id"),
                        "reason": "invalid_relevance",
                    }
                )
                continue
            if relevance < 2:
                continue
            anchor = evidence_anchor(raw)
            if anchor is None:
                invalid_rows.append(
                    {
                        "query_index": item_index,
                        "question": item.get("question"),
                        "evidence_index": evidence_index,
                        "evidence_id": raw.get("evidence_id"),
                        "source": raw.get("source"),
                        "reason": "invalid_source_page_or_quote",
                    }
                )
                continue
            rows.append((item_index, raw, anchor))

    anchors = {anchor for _, _, anchor in rows}
    normalized_chunks: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
    for source, chunks in (chunks_by_source or {}).items():
        normalized_source = str(source).strip().replace("\\", "/")
        normalized_chunks[normalized_source] = [
            _chunk_id_text_metadata(chunk) for chunk in (chunks or [])
        ]

    mapping: Dict[Tuple[str, Optional[int], str], set[str]] = {}
    source_not_found = set()
    quote_not_found = set()
    for anchor in anchors:
        source, page, quote = anchor
        chunks = normalized_chunks.get(source, [])
        if not chunks:
            source_not_found.add(source)
            continue
        matched_ids: set[str] = set()
        for chunk_id, text, metadata in chunks:
            if not chunk_id:
                continue
            if page is not None:
                try:
                    chunk_page = int(metadata.get("page"))
                except (TypeError, ValueError):
                    continue
                if chunk_page != page:
                    continue
            if quote in normalize_eval_text(text):
                matched_ids.add(chunk_id)
        if matched_ids:
            mapping[anchor] = matched_ids
        else:
            quote_not_found.add(source)

    unmapped_evidence: List[Dict[str, Any]] = list(invalid_rows)
    for item_index, evidence, anchor in rows:
        if anchor in mapping:
            continue
        unmapped_evidence.append(
            {
                "query_index": item_index,
                "question": items[item_index - 1].get("question"),
                "evidence_id": evidence.get("evidence_id"),
                "source": anchor[0],
                "page": anchor[1],
                "quote": evidence.get("quote"),
                "reason": "source_not_found" if anchor[0] in source_not_found else "quote_not_found",
            }
        )

    evidence_total = len(anchors) + len(invalid_rows)
    coverage = {
        "mode": "evidence" if evidence_total else "file",
        "evidence_total": evidence_total,
        "mapped_evidence": len(mapping),
        "coverage": (len(mapping) / evidence_total) if evidence_total else None,
        "missing_sources": sorted(source_not_found),
        "sources_without_quote": sorted(quote_not_found),
        "unmapped_evidence": unmapped_evidence,
    }
    return mapping, coverage


def prepare_evidence_evaluation(
    items: Sequence[Dict[str, Any]],
    vector_store: Any,
) -> Tuple[Dict[Tuple[str, Optional[int], str], set[str]], Dict[str, Any]]:
    """解析当前 collection，并返回 evidence 映射与可直接写入报告的覆盖率。"""
    if not dataset_uses_evidence(items):
        return {}, {
            "mode": "file",
            "evidence_total": 0,
            "mapped_evidence": 0,
            "coverage": None,
            "unmapped_evidence": [],
        }

    sources = sorted(
        {
            evidence_anchor(evidence)[0]
            for item in items
            for evidence in _relevant_evidence(item)
            if evidence_anchor(evidence) is not None
        }
    )
    try:
        chunks_by_source, aliases = load_evidence_chunks(vector_store, sources)
        mapping, coverage = build_evidence_chunk_map(items, chunks_by_source)
        coverage["source_aliases"] = aliases
        return mapping, coverage
    except Exception as exc:  # noqa: BLE001
        logger.error("evidence 到当前 chunk 的映射失败：%s", exc)
        mapping, coverage = build_evidence_chunk_map(
            items,
            {source: [] for source in sources},
        )
        coverage["error"] = str(exc)
        coverage["coverage"] = 0.0
        return mapping, coverage


def evidence_gold_groups(
    item: Dict[str, Any],
    mapping: Dict[Tuple[str, Optional[int], str], set[str]],
) -> Tuple[List[set[str]], List[Dict[str, Any]]]:
    """返回该 query 的 evidence 金标 chunk 组，以及未映射的 evidence。"""
    gold: List[set[str]] = []
    missing: List[Dict[str, Any]] = []
    seen: set[Tuple[str, Optional[int], str]] = set()
    for evidence in _relevant_evidence(item):
        anchor = evidence_anchor(evidence)
        if anchor is None or anchor in seen:
            continue
        seen.add(anchor)
        chunk_ids = mapping.get(anchor) or set()
        if chunk_ids:
            gold.append(set(chunk_ids))
        else:
            missing.append(
                {
                    "evidence_id": evidence.get("evidence_id"),
                    "source": anchor[0],
                    "page": anchor[1],
                    "quote": evidence.get("quote"),
                }
            )
    return gold, missing


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


def resolve_eval_collection(explicit: str = "", kb_id: str = "") -> str:
    """解析评测使用的真实知识库 collection。

    ``RAGPipeline.from_config`` 默认打开裸 collection（例如
    ``chinese_rag_kb``）；Web 写入的数据实际位于租户/知识库 collection。
    评测若继续使用裸名称，只会得到空库上的假零分。
    """
    if explicit:
        return explicit.strip()

    db_path = Path(os.getenv("DB_PATH") or "data/users.db")
    if not db_path.is_absolute():
        db_path = resolve_path(db_path)
    if not db_path.exists():
        return ""

    local_user_id = "00000000-0000-0000-0000-000000000001"
    try:
        con = sqlite3.connect(str(db_path))
        try:
            if kb_id:
                row = con.execute(
                    "SELECT collection_name FROM knowledge_bases WHERE id = ?",
                    (kb_id,),
                ).fetchone()
            else:
                row = con.execute(
                    """
                    SELECT collection_name
                    FROM knowledge_bases
                    WHERE is_default = 1
                    ORDER BY CASE WHEN user_id = ? THEN 0 ELSE 1 END, created_at ASC
                    LIMIT 1
                    """,
                    (local_user_id,),
                ).fetchone()
        finally:
            con.close()
        return str(row[0]) if row and row[0] else ""
    except (sqlite3.Error, OSError) as exc:
        logger.warning("无法从 %s 解析默认知识库：%s", db_path, exc)
        return ""


def expected_source_coverage(
    items: Sequence[Dict[str, Any]], available_sources: Sequence[str]
) -> Dict[str, Any]:
    """检查评估集期望文件是否真的存在于当前知识库。

    期望文件缺失时，低命中率只能说明数据集与索引不匹配，不能归因到
    分词、召回或重排。
    """
    expected = {
        str(source).replace("\\", "/")
        for item in items
        if not item_expects_refusal(item)
        for source in (item.get("expected_sources") or [])
        if source
    }
    expected.update(
        evidence_anchor(evidence)[0]
        for item in items
        if not item_expects_refusal(item)
        for evidence in _relevant_evidence(item)
        if evidence_anchor(evidence) is not None
    )
    available = {
        str(source).replace("\\", "/")
        for source in available_sources
        if source
    }
    present = sorted(
        source
        for source in expected
        if any(source in candidate for candidate in available)
    )
    missing = sorted(expected - set(present))
    return {
        "expected": len(expected),
        "present": len(present),
        "missing": len(missing),
        "coverage": (len(present) / len(expected)) if expected else None,
        "missing_sources": missing,
    }


def attach_source_coverage(
    pipeline: RAGPipeline,
    items: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """把当前 collection 的期望源覆盖率附加到报告。"""
    try:
        rows = pipeline.vector_store.list_sources()
        sources = [str(row.get("source") or "") for row in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法检查评估集来源覆盖：%s", exc)
        summary["dataset_source_coverage"] = {"error": str(exc)}
        return summary

    coverage = expected_source_coverage(items, sources)
    summary["dataset_source_coverage"] = coverage
    if coverage["missing_sources"]:
        logger.warning(
            "评估集有 %d/%d 个期望来源不在当前 collection：%s",
            coverage["missing"],
            coverage["expected"],
            ", ".join(coverage["missing_sources"][:10]),
        )
    return summary


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

    evidence_mode = dataset_uses_evidence(items)
    evidence_map, evidence_coverage = prepare_evidence_evaluation(
        items, getattr(pipeline, "vector_store", None)
    )
    retrieval_hits = 0
    keyword_scores: List[float] = []
    sem_sims: List[float] = []
    faiths: List[float] = []
    latencies: List[float] = []
    ttfts: List[float] = []
    tokens_per_sec_values: List[float] = []
    ndcgs: List[float] = []
    mrrs: List[float] = []
    recalls: List[float] = []
    evidence_hits: List[float] = []
    evidence_mrrs_at_5: List[float] = []
    evidence_mrrs_at_10: List[float] = []
    evidence_recalls_at_5: List[float] = []
    evidence_recalls_at_10: List[float] = []
    evidence_excluded = 0
    retrieval_evaluable_samples = 0
    expected_refusal_samples = 0
    correct_refusal_samples = 0
    answerable_samples = 0
    answerable_refusals = 0
    fact_scored_samples = 0
    fact_complete_samples = 0
    answer_contract_pass_samples = 0
    # 可验证性聚合（核心指标：答案必须可回溯到上下文）
    verifications: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    for i, item in enumerate(items, 1):
        q = item["question"]
        expected_sources = item.get("expected_sources", []) or []
        keywords = item.get("expected_keywords", []) or []
        expects_refusal = item_expects_refusal(item)

        t0 = time.perf_counter()
        try:
            result = pipeline.answer_with_timing(q)
        except Exception as exc:
            logger.error("第 %d 条运行失败: %s", i, exc)
            details.append({"index": i, "question": q, "error": str(exc)})
            continue
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        verification = getattr(result, "verification", None)
        actual_refusal = _result_is_refusal(result)
        refusal_correct: Optional[bool] = None
        if expects_refusal:
            expected_refusal_samples += 1
            refusal_correct = actual_refusal
            if actual_refusal:
                correct_refusal_samples += 1
        else:
            answerable_samples += 1
            if actual_refusal:
                answerable_refusals += 1

        # 检索命中 + NDCG/MRR
        retrieval_items = list(getattr(result, "raw_hits", None) or result.sources or [])
        retrieved_sources = []
        for retrieval_item in retrieval_items:
            if isinstance(retrieval_item, dict):
                retrieved_sources.append(str(retrieval_item.get("source") or ""))
            else:
                retrieved_sources.append(
                    str((getattr(retrieval_item, "metadata", None) or {}).get("source") or "")
                )
        source_hit: Optional[bool]
        if expects_refusal:
            source_hit = None
            ndcg = None
            source_mrr = None
            source_recall = None
        else:
            retrieval_evaluable_samples += 1
            source_hit = bool(
                expected_sources
                and any(
                    any(es in src for es in expected_sources)
                    for src in retrieved_sources
                )
            )
            if source_hit:
                retrieval_hits += 1
            ndcg = ndcg_at_k(retrieved_sources, expected_sources, k=5)
            source_mrr = mrr(retrieved_sources, expected_sources)
            source_recall = recall_at_k(retrieved_sources, expected_sources, k=10)
        if ndcg is not None:
            ndcgs.append(ndcg)
        if source_mrr is not None:
            mrrs.append(source_mrr)
        if source_recall is not None:
            recalls.append(source_recall)

        evidence_hit: Optional[float] = None
        evidence_mrr_5: Optional[float] = None
        evidence_mrr_10: Optional[float] = None
        evidence_recall_5: Optional[float] = None
        evidence_recall_10: Optional[float] = None
        gold_groups: List[set[str]] = []
        missing_evidence: List[Dict[str, Any]] = []
        if evidence_mode and not expects_refusal:
            gold_groups, missing_evidence = evidence_gold_groups(item, evidence_map)
            if gold_groups and not missing_evidence:
                hit_ids = _ranked_hit_ids(retrieval_items)
                metrics_5 = evidence_metrics_for_hits(hit_ids, gold_groups, k=5)
                metrics_10 = evidence_metrics_for_hits(hit_ids, gold_groups, k=10)
                if metrics_5 is not None and metrics_10 is not None:
                    evidence_hit = float(metrics_10["hit_rate"])
                    evidence_mrr_5 = float(metrics_5["mrr"])
                    evidence_mrr_10 = float(metrics_10["mrr"])
                    evidence_recall_5 = float(metrics_5["recall"])
                    evidence_recall_10 = float(metrics_10["recall"])
                    evidence_hits.append(evidence_hit)
                    evidence_mrrs_at_5.append(evidence_mrr_5)
                    evidence_mrrs_at_10.append(evidence_mrr_10)
                    evidence_recalls_at_5.append(evidence_recall_5)
                    evidence_recalls_at_10.append(evidence_recall_10)
            else:
                evidence_excluded += 1

        hit: Optional[float | bool]
        if expects_refusal:
            hit = None
        else:
            hit = evidence_hit if evidence_mode else source_hit
        mrr_val = evidence_mrr_10 if evidence_mode else source_mrr
        recall_val = evidence_recall_10 if evidence_mode else source_recall

        # 关键词覆盖率 + RAGAS 轻量语义指标（无外部依赖）
        kw_cov = keyword_coverage(result.answer, keywords) if keywords else None
        if kw_cov is not None:
            keyword_scores.append(kw_cov)
            if not expects_refusal:
                fact_scored_samples += 1
                if not actual_refusal and kw_cov >= 1.0:
                    fact_complete_samples += 1
        if not expects_refusal and not actual_refusal and kw_cov is not None:
            vf = verification if isinstance(verification, dict) else {}
            if (
                kw_cov >= 1.0
                and not vf.get("invalid_citations")
                and not vf.get("unsupported_facts")
            ):
                answer_contract_pass_samples += 1
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
                "expect_refusal": expects_refusal,
                "actual_refusal": actual_refusal,
                "refusal_correct": refusal_correct,
                "retrieval_hit": bool(hit) if hit is not None else None,
                "source_retrieval_hit": bool(source_hit) if source_hit is not None else None,
                "keyword_coverage": kw_cov,
                "answer_similarity": round(_sim, 4) if _sim is not None else None,
                "faithfulness": round(_faith, 4) if _faith is not None else None,
                "ndcg_at_5": ndcg,
                "mrr": mrr_val,
                "mrr_at_10": evidence_mrr_10,
                "recall_at_10": recall_val,
                "recall_at_5": evidence_recall_5,
                "hit_rate_at_10": evidence_hit,
                "evidence_eval_eligible": evidence_hit is not None if evidence_mode else None,
                "unmapped_evidence": missing_evidence,
                "verification": verification,
            }
        )
        logger.info(
            "[%d/%d] 命中=%s 覆盖率=%s NDCG=%s MRR=%s 延迟=%.0fms TTFT=%sms",
            i,
            n,
            hit if hit is not None else "—",
            f"{kw_cov:.2f}" if kw_cov is not None else "—",
            f"{ndcg:.2f}" if ndcg is not None else "—",
            f"{mrr_val:.2f}" if mrr_val is not None else "—",
            latency,
            f"{ttft_ms:.0f}" if isinstance(ttft_ms, (int, float)) else "—",
        )

    # 可验证性汇总：verified 比例、平均引用覆盖率、编造事实计数
    answered = [v for v in verifications if not v.get("refused")]
    verified = [v for v in answered if v.get("status") == "verified"]
    citation_total = sum(int(v.get("citation_total", 0) or 0) for v in verifications)
    valid_citation_total = sum(
        int(v.get("valid_citation_total", 0) or 0) for v in verifications
    )
    sentence_total = sum(int(v.get("sentence_count", 0) or 0) for v in answered)
    cited_sentence_total = sum(
        int(v.get("cited_sentence_count", 0) or 0) for v in answered
    )
    unsupported_sentence_total = sum(
        len(v.get("unsupported_sentences", []) or []) for v in answered
    )
    hard_fact_total = sum(int(v.get("hard_fact_total", 0) or 0) for v in answered)
    unsupported_fact_total = sum(
        len(v.get("unsupported_facts", []) or []) for v in answered
    )
    source_hit_rate = (
        retrieval_hits / retrieval_evaluable_samples
        if retrieval_evaluable_samples
        else None
    )
    source_mrr_mean = sum(mrrs) / len(mrrs) if mrrs else None
    source_recall_mean = sum(recalls) / len(recalls) if recalls else None
    evidence_summary = {
        **evidence_coverage,
        "evaluable_samples": len(evidence_hits),
        "excluded_samples": evidence_excluded,
        "recall_at_5": (
            sum(evidence_recalls_at_5) / len(evidence_recalls_at_5)
            if evidence_recalls_at_5
            else None
        ),
        "recall_at_10": (
            sum(evidence_recalls_at_10) / len(evidence_recalls_at_10)
            if evidence_recalls_at_10
            else None
        ),
        "hit_rate_at_10": (
            sum(evidence_hits) / len(evidence_hits) if evidence_hits else None
        ),
        "mrr_at_5": (
            sum(evidence_mrrs_at_5) / len(evidence_mrrs_at_5)
            if evidence_mrrs_at_5
            else None
        ),
        "mrr_at_10": (
            sum(evidence_mrrs_at_10) / len(evidence_mrrs_at_10)
            if evidence_mrrs_at_10
            else None
        ),
    }
    summary: Dict[str, Any] = {
        "samples": n,
        "retrieval_hit_rate": (
            evidence_summary["hit_rate_at_10"] if evidence_mode else source_hit_rate
        ),
        "avg_keyword_coverage": (
            sum(keyword_scores) / len(keyword_scores) if keyword_scores else None
        ),
        # P1-2 RAGAS 语义指标（无外部依赖轻量版；有 ragas 时可用其官方值替换）
        "avg_answer_similarity": (sum(sem_sims) / len(sem_sims) if sem_sims else None),
        "avg_faithfulness": (sum(faiths) / len(faiths) if faiths else None),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "ndcg_at_5": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        "mrr": evidence_summary["mrr_at_10"] if evidence_mode else source_mrr_mean,
        "mrr_at_10": evidence_summary["mrr_at_10"],
        "recall_at_5": evidence_summary["recall_at_5"],
        "recall_at_10": (
            evidence_summary["recall_at_10"] if evidence_mode else source_recall_mean
        ),
        "hit_rate_at_10": evidence_summary["hit_rate_at_10"],
        "retrieval_evidence": evidence_summary,
        "answer_behavior": {
            "answerable_samples": answerable_samples,
            "expected_refusal_samples": expected_refusal_samples,
            "correct_refusal_samples": correct_refusal_samples,
            "refusal_accuracy": (
                correct_refusal_samples / expected_refusal_samples
                if expected_refusal_samples
                else None
            ),
            "answerable_refusals": answerable_refusals,
            "answerable_refusal_rate": (
                answerable_refusals / answerable_samples
                if answerable_samples
                else None
            ),
            "fact_scored_samples": fact_scored_samples,
            "required_fact_complete_rate": (
                fact_complete_samples / fact_scored_samples
                if fact_scored_samples
                else None
            ),
            "answer_contract_pass_rate": (
                answer_contract_pass_samples / fact_scored_samples
                if fact_scored_samples
                else None
            ),
        },
        "retrieval_evaluable_samples": retrieval_evaluable_samples,
        "excluded_refusal_samples": expected_refusal_samples,
        "dataset_audit": audit_eval_dataset(items),
        "source_retrieval": {
            "hit_rate": source_hit_rate,
            "mrr": source_mrr_mean,
            "recall_at_10": source_recall_mean,
            "ndcg_at_5": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        },
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
            "citation_total": citation_total,
            "valid_citation_total": valid_citation_total,
            "citation_precision": (
                valid_citation_total / citation_total if citation_total else None
            ),
            "sentence_total": sentence_total,
            "cited_sentence_total": cited_sentence_total,
            "unsupported_sentence_total": unsupported_sentence_total,
            "ungrounded_cited_sentence_rate": (
                unsupported_sentence_total / cited_sentence_total
                if cited_sentence_total
                else None
            ),
            "hard_fact_total": hard_fact_total,
            "unsupported_fact_rate": (
                unsupported_fact_total / hard_fact_total
                if hard_fact_total
                else None
            ),
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
        "mrr_at_10": summary.get("mrr_at_10"),
        "recall_at_5": summary.get("recall_at_5"),
        "recall_at_10": summary.get("recall_at_10"),
        "hit_rate_at_10": summary.get("hit_rate_at_10"),
        "evidence_coverage": (
            summary.get("retrieval_evidence") or {}
        ).get("coverage"),
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
def evaluate_retrieval(
    pipeline: RAGPipeline,
    items: List[Dict[str, Any]],
    top_k: int = 10,
) -> Dict[str, Any]:
    """检索回归评估：只跑"召回 + 重排"，不加载 LLM、不生成答案。

    用途：UI 占用 GPU 时也能做检索质量回归（CPU 即可），改动分词/分块/
    检索参数后快速看 Recall@K / HitRate@K / MRR@K 的涨跌。带
    expected_evidence 的样本按 evidence/chunk 级计算；旧数据集继续按
    expected_sources 做文件级计算。
    """
    n = len(items)
    if n == 0:
        return {"samples": 0}

    evidence_mode = dataset_uses_evidence(items)
    evidence_map, evidence_coverage = prepare_evidence_evaluation(
        items, getattr(pipeline, "vector_store", None)
    )
    source_hits_total = 0
    ndcgs: List[float] = []
    source_mrrs: List[float] = []
    source_recalls: List[float] = []
    evidence_hits: List[float] = []
    evidence_mrrs_at_5: List[float] = []
    evidence_mrrs_at_10: List[float] = []
    evidence_recalls_at_5: List[float] = []
    evidence_recalls_at_10: List[float] = []
    evidence_excluded = 0
    retrieval_evaluable_samples = 0
    refusal_samples_skipped = 0
    latencies: List[float] = []
    details: List[Dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        q = item["question"]
        expected_sources = item.get("expected_sources", []) or []
        if item_expects_refusal(item):
            refusal_samples_skipped += 1
            details.append(
                {
                    "index": i,
                    "question": q,
                    "expect_refusal": True,
                    "skipped_reason": "retrieval_metrics_not_applicable_to_refusal_sample",
                    "retrieval_hit": None,
                    "source_retrieval_hit": None,
                }
            )
            continue
        retrieval_evaluable_samples += 1
        t0 = time.perf_counter()
        try:
            hits = pipeline.retrieve(q, top_k=top_k)
        except Exception as exc:
            logger.error("第 %d 条检索失败: %s", i, exc)
            details.append({"index": i, "question": q, "error": str(exc)})
            continue
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        sources = [str((h.metadata or {}).get("source") or "") for h in hits]
        source_hit = 0
        if expected_sources:
            for src in sources:
                if any(es in src for es in expected_sources):
                    source_hit = 1
                    break
        else:
            source_hit = 1
        source_hits_total += source_hit
        ndcg = ndcg_at_k(sources, expected_sources, k=5)
        source_mrr = mrr(sources, expected_sources)
        source_recall = recall_at_k(sources, expected_sources, k=10)
        if ndcg is not None:
            ndcgs.append(ndcg)
        if source_mrr is not None:
            source_mrrs.append(source_mrr)
        if source_recall is not None:
            source_recalls.append(source_recall)

        evidence_hit: Optional[float] = None
        evidence_mrr_5: Optional[float] = None
        evidence_mrr_10: Optional[float] = None
        evidence_recall_5: Optional[float] = None
        evidence_recall_10: Optional[float] = None
        gold_groups: List[set[str]] = []
        missing_evidence: List[Dict[str, Any]] = []
        if evidence_mode:
            gold_groups, missing_evidence = evidence_gold_groups(item, evidence_map)
            if gold_groups and not missing_evidence:
                hit_ids = _ranked_hit_ids(hits)
                metrics_5 = evidence_metrics_for_hits(hit_ids, gold_groups, k=5)
                metrics_10 = evidence_metrics_for_hits(hit_ids, gold_groups, k=10)
                if metrics_5 is not None and metrics_10 is not None:
                    evidence_hit = float(metrics_10["hit_rate"])
                    evidence_mrr_5 = float(metrics_5["mrr"])
                    evidence_mrr_10 = float(metrics_10["mrr"])
                    evidence_recall_5 = float(metrics_5["recall"])
                    evidence_recall_10 = float(metrics_10["recall"])
                    evidence_hits.append(evidence_hit)
                    evidence_mrrs_at_5.append(evidence_mrr_5)
                    evidence_mrrs_at_10.append(evidence_mrr_10)
                    evidence_recalls_at_5.append(evidence_recall_5)
                    evidence_recalls_at_10.append(evidence_recall_10)
            else:
                evidence_excluded += 1

        hit = evidence_hit if evidence_mode else source_hit
        reported_mrr = evidence_mrr_10 if evidence_mode else source_mrr
        reported_recall = evidence_recall_10 if evidence_mode else source_recall
        details.append(
            {
                "index": i,
                "question": q,
                "retrieval_hit": bool(hit),
                "source_retrieval_hit": bool(source_hit),
                "ndcg_at_5": ndcg,
                "mrr": reported_mrr,
                "mrr_at_10": evidence_mrr_10,
                "recall_at_10": reported_recall,
                "recall_at_5": evidence_recall_5,
                "hit_rate_at_10": evidence_hit,
                "evidence_eval_eligible": evidence_hit is not None if evidence_mode else None,
                "unmapped_evidence": missing_evidence,
                "latency_ms": round(latency, 2),
                "retrieved_sources": sources[:top_k],
            }
        )
        logger.info(
            "[%d/%d] 命中=%s NDCG=%s MRR@K=%s 检索延迟=%.0fms",
            i, n, bool(hit),
            f"{ndcg:.2f}" if ndcg is not None else "—",
            f"{reported_mrr:.2f}" if reported_mrr is not None else "—",
            latency,
        )

    source_hit_rate = (
        source_hits_total / retrieval_evaluable_samples
        if retrieval_evaluable_samples
        else None
    )
    source_mrr_mean = sum(source_mrrs) / len(source_mrrs) if source_mrrs else None
    source_recall_mean = (
        sum(source_recalls) / len(source_recalls) if source_recalls else None
    )
    evidence_summary = {
        **evidence_coverage,
        "evaluable_samples": len(evidence_hits),
        "excluded_samples": evidence_excluded,
        "unlabeled_samples": n - sum(bool(item.get("expected_evidence")) for item in items),
        "recall_at_5": (
            sum(evidence_recalls_at_5) / len(evidence_recalls_at_5)
            if evidence_recalls_at_5
            else None
        ),
        "recall_at_10": (
            sum(evidence_recalls_at_10) / len(evidence_recalls_at_10)
            if evidence_recalls_at_10
            else None
        ),
        "hit_rate_at_10": (
            sum(evidence_hits) / len(evidence_hits) if evidence_hits else None
        ),
        "mrr_at_5": (
            sum(evidence_mrrs_at_5) / len(evidence_mrrs_at_5)
            if evidence_mrrs_at_5
            else None
        ),
        "mrr_at_10": (
            sum(evidence_mrrs_at_10) / len(evidence_mrrs_at_10)
            if evidence_mrrs_at_10
            else None
        ),
    }
    return {
        "samples": n,
        "mode": "retrieval_only",
        "retrieval_hit_rate": (
            evidence_summary["hit_rate_at_10"] if evidence_mode else source_hit_rate
        ),
        "ndcg_at_5": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        "mrr": evidence_summary["mrr_at_10"] if evidence_mode else source_mrr_mean,
        "mrr_at_10": evidence_summary["mrr_at_10"],
        "recall_at_5": evidence_summary["recall_at_5"],
        "recall_at_10": (
            evidence_summary["recall_at_10"] if evidence_mode else source_recall_mean
        ),
        "hit_rate_at_10": evidence_summary["hit_rate_at_10"],
        "retrieval_evidence": evidence_summary,
        "dataset_audit": audit_eval_dataset(items),
        "retrieval_evaluable_samples": retrieval_evaluable_samples,
        "excluded_refusal_samples": refusal_samples_skipped,
        "source_retrieval": {
            "hit_rate": source_hit_rate,
            "mrr": source_mrr_mean,
            "recall_at_10": source_recall_mean,
            "ndcg_at_5": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        },
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

    audit = summary.get("dataset_audit") or {}
    min_answerable_samples = getattr(args, "min_answerable_samples", None)
    if min_answerable_samples is not None:
        actual = int(audit.get("answerable_samples", 0) or 0)
        if actual < min_answerable_samples:
            failures.append(
                f"answerable_samples {actual} < {min_answerable_samples}"
            )

    min_hard_samples = getattr(args, "min_hard_samples", None)
    if min_hard_samples is not None:
        actual = int((audit.get("difficulty") or {}).get("hard", 0) or 0)
        if actual < min_hard_samples:
            failures.append(f"hard_samples {actual} < {min_hard_samples}")

    min_refusal_samples = getattr(args, "min_refusal_samples", None)
    if min_refusal_samples is not None:
        actual = int(audit.get("refusal_samples", 0) or 0)
        if actual < min_refusal_samples:
            failures.append(
                f"refusal_samples {actual} < {min_refusal_samples}"
            )

    hit_rate = summary.get("retrieval_hit_rate")
    if args.min_hit_rate is not None and float(hit_rate or 0.0) < args.min_hit_rate:
        actual = float(hit_rate or 0.0)
        failures.append(f"retrieval_hit_rate {actual:.3f} < {args.min_hit_rate:.3f}")

    ndcg_value = summary.get("ndcg_at_5")
    if args.min_ndcg is not None:
        actual = float(ndcg_value or 0.0)
        if actual < args.min_ndcg:
            failures.append(f"ndcg_at_5 {actual:.3f} < {args.min_ndcg:.3f}")

    recall_value = summary.get("recall_at_10")
    min_recall = getattr(args, "min_recall", None)
    if min_recall is not None:
        actual = float(recall_value or 0.0)
        if actual < min_recall:
            failures.append(f"recall_at_10 {actual:.3f} < {min_recall:.3f}")

    recall_at_5 = summary.get("recall_at_5")
    min_recall_at_5 = getattr(args, "min_recall_at_5", None)
    if min_recall_at_5 is not None:
        actual = float(recall_at_5 or 0.0)
        if actual < min_recall_at_5:
            failures.append(
                f"recall_at_5 {actual:.3f} < {min_recall_at_5:.3f}"
            )

    min_recall_at_10 = getattr(args, "min_recall_at_10", None)
    if min_recall_at_10 is not None:
        actual = float(recall_value or 0.0)
        if actual < min_recall_at_10:
            failures.append(
                f"recall_at_10 {actual:.3f} < {min_recall_at_10:.3f}"
            )

    hit_rate_at_10 = summary.get("hit_rate_at_10")
    min_hit_rate_at_10 = getattr(args, "min_hit_rate_at_10", None)
    if min_hit_rate_at_10 is not None:
        actual = float(hit_rate_at_10 or 0.0)
        if actual < min_hit_rate_at_10:
            failures.append(
                f"hit_rate_at_10 {actual:.3f} < {min_hit_rate_at_10:.3f}"
            )

    mrr_at_10 = summary.get("mrr_at_10")
    min_mrr_at_10 = getattr(args, "min_mrr_at_10", None)
    if min_mrr_at_10 is not None:
        actual = float(mrr_at_10 or 0.0)
        if actual < min_mrr_at_10:
            failures.append(f"mrr_at_10 {actual:.3f} < {min_mrr_at_10:.3f}")

    source_coverage = summary.get("dataset_source_coverage") or {}
    source_coverage_value = source_coverage.get("coverage")
    min_source_coverage = getattr(args, "min_source_coverage", None)
    if min_source_coverage is not None:
        actual = float(source_coverage_value or 0.0)
        if actual < min_source_coverage:
            failures.append(
                f"dataset_source_coverage {actual:.3f} < {min_source_coverage:.3f}"
            )

    evidence_coverage = summary.get("retrieval_evidence") or {}
    min_evidence_coverage = getattr(args, "min_evidence_coverage", None)
    if min_evidence_coverage is not None:
        if evidence_coverage.get("mode") != "evidence":
            failures.append(
                "retrieval_evidence.coverage unavailable (dataset has no expected_evidence)"
            )
        else:
            actual = float(evidence_coverage.get("coverage") or 0.0)
            if actual < min_evidence_coverage:
                failures.append(
                    "retrieval_evidence.coverage "
                    f"{actual:.3f} < {min_evidence_coverage:.3f}"
                )

    verified_rate = (summary.get("verifiability") or {}).get("verified_rate")
    if args.min_verified_rate is not None:
        actual = float(verified_rate or 0.0)
        if actual < args.min_verified_rate:
            failures.append(
                f"verified_rate {actual:.3f} < {args.min_verified_rate:.3f}"
            )

    answer_behavior = summary.get("answer_behavior") or {}
    verifiability = summary.get("verifiability") or {}
    threshold_checks = [
        (
            "required_fact_complete_rate",
            answer_behavior.get("required_fact_complete_rate"),
            getattr(args, "min_required_fact_complete_rate", None),
            "min",
        ),
        (
            "answer_contract_pass_rate",
            answer_behavior.get("answer_contract_pass_rate"),
            getattr(args, "min_answer_contract_pass_rate", None),
            "min",
        ),
        (
            "refusal_accuracy",
            answer_behavior.get("refusal_accuracy"),
            getattr(args, "min_correct_refusal_rate", None),
            "min",
        ),
        (
            "answerable_refusal_rate",
            answer_behavior.get("answerable_refusal_rate"),
            getattr(args, "max_false_refusal_rate", None),
            "max",
        ),
        (
            "citation_precision",
            verifiability.get("citation_precision"),
            getattr(args, "min_citation_precision", None),
            "min",
        ),
        (
            "unsupported_fact_rate",
            verifiability.get("unsupported_fact_rate"),
            getattr(args, "max_unsupported_fact_rate", None),
            "max",
        ),
    ]
    for name, value, limit, direction in threshold_checks:
        if limit is None:
            continue
        if value is None:
            failures.append(f"{name} unavailable")
            continue
        actual = float(value)
        if direction == "min" and actual < limit:
            failures.append(f"{name} {actual:.3f} < {limit:.3f}")
        if direction == "max" and actual > limit:
            failures.append(f"{name} {actual:.3f} > {limit:.3f}")

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
    p.add_argument(
        "--disable-reranker",
        action="store_true",
        help="覆盖 config/local_overrides.yaml，关闭重排，用于同库前后对照",
    )
    p.add_argument("--top-k", type=int, default=None, help="覆盖 retrieval.top_k")
    p.add_argument("--collection", default="", help="评测 collection；默认取本地用户的默认知识库")
    p.add_argument("--kb-id", default="", help="按知识库 ID 选择评测 collection")
    p.add_argument(
        "--synthesize-demo",
        action="store_true",
        help="评估集缺失时，从 data/raw/<source> 合成 5 条样本",
    )
    p.add_argument("--synthesize-from", default="data/raw/rag-intro.md", help="合成使用的 markdown")
    p.add_argument("--no-archive", action="store_true", help="不写历史存档")
    p.add_argument(
        "--allow-empty-eval",
        action="store_true",
        help=(
            "评估集为空时返回 0（仅告警）而不是 1。"
            "空库冷启属预期，此时指标无从谈起，不应把 CI 卡死；"
            "但**指标不达标**仍然返回 1 —— 门禁只对'没有数据'宽容。"
        ),
    )
    p.add_argument("--min-samples", type=int, default=0, help="CI：最少样本数")
    p.add_argument(
        "--min-answerable-samples",
        type=int,
        default=None,
        help="CI：最少可回答问题数（不含 expect_refusal）",
    )
    p.add_argument(
        "--min-hard-samples",
        type=int,
        default=None,
        help="CI：最少 hard 难度样本数",
    )
    p.add_argument(
        "--min-refusal-samples",
        type=int,
        default=None,
        help="CI：最少 expect_refusal 无答案样本数",
    )
    p.add_argument("--min-hit-rate", type=float, default=None, help="CI：最低检索命中率")
    p.add_argument("--min-ndcg", type=float, default=None, help="CI：最低 NDCG@5")
    p.add_argument("--min-recall", type=float, default=None, help="CI：最低 Recall@10")
    p.add_argument(
        "--min-recall-at-5",
        type=float,
        default=None,
        help="CI：evidence 级最低 Recall@5",
    )
    p.add_argument(
        "--min-recall-at-10",
        type=float,
        default=None,
        help="CI：evidence 级最低 Recall@10",
    )
    p.add_argument(
        "--min-hit-rate-at-10",
        type=float,
        default=None,
        help="CI：evidence 级最低 HitRate@10",
    )
    p.add_argument(
        "--min-mrr-at-10",
        type=float,
        default=None,
        help="CI：evidence 级最低 MRR@10",
    )
    p.add_argument(
        "--min-evidence-coverage",
        type=float,
        default=None,
        help="CI：expected_evidence 能映射到当前 collection chunk 的最低比例",
    )
    p.add_argument(
        "--min-source-coverage",
        type=float,
        default=None,
        help="CI：评估集期望文件在当前知识库中的最低覆盖率",
    )
    p.add_argument("--min-verified-rate", type=float, default=None, help="CI：最低可验证率")
    p.add_argument(
        "--min-required-fact-complete-rate",
        type=float,
        default=None,
        help="CI：全部期望关键事实均出现的可回答样本最低比例",
    )
    p.add_argument(
        "--min-answer-contract-pass-rate",
        type=float,
        default=None,
        help="CI：关键事实完整、引用有效且无未接地硬事实的最低比例",
    )
    p.add_argument(
        "--min-correct-refusal-rate",
        type=float,
        default=None,
        help="CI：应拒答样本的实际拒答率",
    )
    p.add_argument(
        "--max-false-refusal-rate",
        type=float,
        default=None,
        help="CI：可回答样本被错误拒答的最高比例",
    )
    p.add_argument(
        "--min-citation-precision",
        type=float,
        default=None,
        help="CI：引用编号有效性最低比例",
    )
    p.add_argument(
        "--max-unsupported-fact-rate",
        type=float,
        default=None,
        help="CI：数字/日期/金额等硬事实未接地比例上限",
    )
    p.add_argument("--max-p95-latency-ms", type=float, default=None, help="CI：P95 延迟上限")
    return p


def main():
    args = _build_arg_parser().parse_args()

    cfg = load_eval_config(args.config)
    if args.disable_reranker:
        cfg.setdefault("reranker", {})["enabled"] = False
    if args.top_k:
        # 只写 retrieval.top_k —— top_k 的唯一真值来源。
        # 此前还会写 rag.top_k，而该字段既从未被读取、也已从 config.yaml 删除
        # （schema 是 extra="forbid"，多写这个键会让配置校验直接失败）。
        cfg.setdefault("retrieval", {})["top_k"] = args.top_k

    dataset_path = Path(args.dataset or cfg.get("evaluation", {}).get("dataset_path", "data/eval/eval_set.jsonl"))
    dataset_path = resolve_path(dataset_path)

    if not dataset_path.exists() and args.synthesize_demo:
        md_path = resolve_path(args.synthesize_from)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        synthesize_demo_dataset(md_path, dataset_path, n=5)
        logger.info("评估集已生成：%s", dataset_path)

    items = load_eval_dataset(dataset_path)
    if not items:
        if args.allow_empty_eval:
            # 空评估集：空库冷启属预期，指标无从谈起。
            # 明确告警并以 0 退出 —— 但注意这只针对"没有数据"，
            # 指标不达标仍然是硬失败。
            logger.warning(
                "评估集为空或不存在（%s），已跳过评估（--allow-empty-eval）。"
                "本次不产生任何质量结论。",
                dataset_path,
            )
            print("::warning::评估集为空，本次评估已跳过（无质量结论）")
            return 0
        logger.error("评估集为空或不存在：%s", dataset_path)
        return 1

    if args.retrieval_only:
        # 检索回归模式：lazy_llm 不加载生成模型（CPU 可跑，可与 UI 并存）
        pipeline = RAGPipeline.from_config(args.config, overrides=cfg, lazy_llm=True)
        collection = resolve_eval_collection(args.collection, args.kb_id)
        if collection:
            from src.factories import TenantAwareFactory

            pipeline.vector_store = TenantAwareFactory.get_chroma_store_by_name(collection, cfg)
            logger.info(
                "检索评测 collection=%s（%d chunks）",
                collection,
                pipeline.vector_store.count(),
            )
        else:
            logger.warning("未找到默认知识库或显式 collection，将使用配置中的基础 collection")
        summary = evaluate_retrieval(pipeline, items, top_k=args.top_k or 10)
        summary = attach_source_coverage(pipeline, items, summary)
        summary["effective_config"] = {
            "collection": collection or pipeline.vector_store.collection_name,
            "reranker_enabled": getattr(pipeline, "reranker", None) is not None,
            "llm_backend": (cfg.get("llm", {}) or {}).get("backend"),
            "llm_model": (
                (cfg.get("llm", {}) or {}).get("openai", {}) or {}
            ).get("model"),
            "embedding_backend": (cfg.get("embedding", {}) or {}).get("backend"),
            "hybrid_enabled": bool(
                (cfg.get("vector_store", {}).get("hybrid", {}) or {}).get(
                    "enabled", False
                )
            ),
            "top_k": args.top_k or 10,
        }
    else:
        # 全链路评估也保持懒加载：extractive 模式不应构造生成模型，
        # 生成式后端则在首次 answer() 时加载。此前默认 eager 会让
        # `llm.backend=extractive` 在评估入口直接报“未知后端”。
        pipeline = RAGPipeline.from_config(args.config, overrides=cfg, lazy_llm=True)
        collection = resolve_eval_collection(args.collection, args.kb_id)
        if collection:
            from src.factories import TenantAwareFactory

            pipeline.vector_store = TenantAwareFactory.get_chroma_store_by_name(collection, cfg)
        summary = evaluate(pipeline, items, llm_self_eval=args.llm_self_eval)
        summary = attach_source_coverage(pipeline, items, summary)
        summary["effective_config"] = {
            "collection": collection or pipeline.vector_store.collection_name,
            "reranker_enabled": getattr(pipeline, "reranker", None) is not None,
            "llm_backend": (cfg.get("llm", {}) or {}).get("backend"),
            "llm_model": (
                (cfg.get("llm", {}) or {}).get("openai", {}) or {}
            ).get("model"),
            "embedding_backend": (cfg.get("embedding", {}) or {}).get("backend"),
            "hybrid_enabled": bool((cfg.get("vector_store", {}).get("hybrid", {}) or {}).get("enabled", False)),
        }

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
    audit = summary.get("dataset_audit") or {}
    if audit:
        difficulty = audit.get("difficulty") or {}
        print(
            "难度分布          : "
            f"easy={difficulty.get('easy', 0)} "
            f"medium={difficulty.get('medium', 0)} "
            f"hard={difficulty.get('hard', 0)}"
        )
        print(
            "可回答 / 拒答     : "
            f"{audit.get('answerable_samples', 0)} / "
            f"{audit.get('refusal_samples', 0)}"
        )
        if not audit.get("valid", True):
            print("数据集质量        : 存在结构问题，请查看 report.dataset_audit")
    source_coverage = (summary.get("dataset_source_coverage") or {}).get("coverage")
    print(
        f"评测源覆盖率      : {source_coverage:.2%}"
        if source_coverage is not None
        else "评测源覆盖率      : —"
    )
    hit_rate_value = summary.get("retrieval_hit_rate")
    print(
        f"检索命中率        : {hit_rate_value:.2%}"
        if hit_rate_value is not None
        else "检索命中率        : —"
    )
    evidence_summary = summary.get("retrieval_evidence") or {}
    if evidence_summary.get("mode") == "evidence":
        evidence_cov = evidence_summary.get("coverage")
        print(
            f"Evidence 覆盖率   : {evidence_cov:.2%}"
            if evidence_cov is not None
            else "Evidence 覆盖率   : —"
        )
        print(
            "可评测 / 排除     : "
            f"{evidence_summary.get('evaluable_samples', 0)} / "
            f"{evidence_summary.get('excluded_samples', 0)}"
        )
    if summary.get("mode") == "retrieval_only":
        print("(retrieval-only 模式：不含生成侧指标)")
        n5 = summary.get("ndcg_at_5")
        print(f"NDCG@5            : {n5:.3f}" if n5 is not None else "NDCG@5            : —")
        mr = summary.get("mrr_at_10")
        if mr is None and evidence_summary.get("mode") != "evidence":
            mr = summary.get("mrr")
        print(f"MRR@10            : {mr:.3f}" if mr is not None else "MRR@10            : —")
        recall5 = summary.get("recall_at_5")
        print(
            f"Recall@5          : {recall5:.2%}"
            if recall5 is not None
            else "Recall@5          : —"
        )
        recall = summary.get("recall_at_10")
        print(f"Recall@10         : {recall:.2%}" if recall is not None else "Recall@10         : —")
        hit10 = summary.get("hit_rate_at_10")
        print(
            f"HitRate@10        : {hit10:.2%}"
            if hit10 is not None
            else "HitRate@10        : —"
        )
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
    behavior = summary.get("answer_behavior") or {}
    vr = v.get("verified_rate")
    print(f"可验证率          : {vr:.2%}" if vr is not None else "可验证率          : —")
    cr = v.get("avg_cited_sentence_ratio")
    print(f"平均引用覆盖率    : {cr:.2%}" if cr is not None else "平均引用覆盖率    : —")
    rr = v.get("refusal_rate")
    print(f"拒答率            : {rr:.2%}" if rr is not None else "拒答率            : —")
    ra = behavior.get("refusal_accuracy")
    print(f"无答案题正确拒答  : {ra:.2%}" if ra is not None else "无答案题正确拒答  : —")
    arr = behavior.get("answerable_refusal_rate")
    print(f"应答题误拒率      : {arr:.2%}" if arr is not None else "应答题误拒率      : —")
    print(f"编造事实/越界引用 : {v.get('unsupported_fact_total', 0)} / {v.get('invalid_citation_total', 0)}")
    lat = summary.get("avg_latency_ms")
    print(f"平均响应时间      : {lat:.0f} ms" if lat is not None else "平均响应时间      : —")
    p95 = summary.get("p95_latency_ms")
    print(f"P95 响应时间      : {p95:.0f} ms" if p95 is not None else "P95 响应时间      : —")
    ndcg = summary.get("ndcg_at_5")
    print(f"NDCG@5            : {ndcg:.4f}" if ndcg is not None else "NDCG@5            : —")
    mrr_val = summary.get("mrr_at_10")
    if mrr_val is None and evidence_summary.get("mode") != "evidence":
        mrr_val = summary.get("mrr")
    print(f"MRR@10            : {mrr_val:.4f}" if mrr_val is not None else "MRR@10            : —")
    recall5 = summary.get("recall_at_5")
    print(
        f"Recall@5          : {recall5:.2%}"
        if recall5 is not None
        else "Recall@5          : —"
    )
    recall = summary.get("recall_at_10")
    print(f"Recall@10         : {recall:.2%}" if recall is not None else "Recall@10         : —")
    hit10 = summary.get("hit_rate_at_10")
    print(f"HitRate@10        : {hit10:.2%}" if hit10 is not None else "HitRate@10        : —")
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
