"""Build cited answers directly from retrieved passages, without an LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_TERM_RE = re.compile(r"[A-Za-z0-9_+.-]+|[\u4e00-\u9fff]+")
_SPACE_RE = re.compile(r"\s+")
_SOURCE_RANGE_RE = re.compile(r"\[\d{1,3}\]\s*[-–—]\s*\[\d{1,3}\]")
_SOURCE_CITE_RE = re.compile(r"\[\d{1,3}\]")
_LEADING_HEADER_RE = re.compile(r"^\s*【[^】\n]{0,120}】\s*")
_MD_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s+.*$", re.MULTILINE)
_PAREN_MARKER_RE = re.compile(r"^\s*[（(][^）)]{0,24}[）)]\s*")
_BULLET_MARKER_RE = re.compile(r"^\s*(?:[-•·*]\s*)+")
_NUM_MARKER_RE = re.compile(r"^\s*\d{1,2}[.、．)]\s*")


_STOP_BIGRAMS = frozenset(
    {
        "什么",
        "怎么",
        "如何",
        "多少",
        "哪些",
        "哪个",
        "是否",
        "可以",
        "请问",
        "一下",
        "这个",
        "那个",
    }
)


@dataclass(frozen=True)
class ExtractiveEvidence:
    """Deterministic evidence diagnostics for the extractive answer mode."""

    query_term_count: int
    best_term_hits: int
    best_query_coverage: float
    best_sentence: str
    candidate_sentences: Tuple[str, ...]


def _query_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in _TERM_RE.finditer(text or ""):
        part = match.group(0)
        if part.isascii():
            if len(part) >= 2 or part.isdigit():
                terms.add(part.lower())
            continue
        cleaned = "".join(ch for ch in part if ch.strip())
        if not cleaned:
            continue
        if len(cleaned) == 1:
            terms.add(cleaned)
            continue
        for index in range(len(cleaned) - 1):
            bigram = cleaned[index : index + 2]
            if bigram not in _STOP_BIGRAMS:
                terms.add(bigram)
    return terms


def _term_hits(terms: Iterable[str], text: str) -> int:
    lowered = (text or "").lower()
    return sum(1 for term in terms if term in lowered)


def _split_sentences(text: str) -> List[str]:
    cleaned = _SOURCE_RANGE_RE.sub("编号", text or "")
    cleaned = _SOURCE_CITE_RE.sub("", cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned).strip()
    sentences = [_strip_heading_markers(match.group(0)) for match in _SENTENCE_RE.finditer(cleaned)]
    return [sentence for sentence in sentences if len(sentence) >= 8]


def _strip_markdown_headings(text: str) -> str:
    """Remove Markdown heading lines to avoid leaking titles into extracts."""
    return _MD_HEADING_LINE_RE.sub("", text or "")


def _strip_heading_markers(text: str) -> str:
    """Strip list/heading markers such as （C1） / （M3） / 1. / - from a sentence."""
    cleaned = text or ""
    for _ in range(3):
        prev = cleaned
        cleaned = _PAREN_MARKER_RE.sub("", cleaned)
        cleaned = _BULLET_MARKER_RE.sub("", cleaned)
        cleaned = _NUM_MARKER_RE.sub("", cleaned)
        if cleaned == prev:
            break
    return cleaned.strip()


def _iter_chunk_sentences(context_chunks: Sequence[Dict[str, Any]]) -> Iterable[Tuple[int, str]]:
    """Yield ``(position, sentence)`` without leaking chunk headings."""
    for chunk in context_chunks:
        content = str(chunk.get("content") or "").strip()
        header = _LEADING_HEADER_RE.match(content)
        if header:
            content = content[header.end():].lstrip()
        content = _strip_markdown_headings(content).strip()
        sentences = _split_sentences(content)
        if not sentences and content:
            sentences = [content[:260].strip()]
        for position, sentence in enumerate(sentences):
            if sentence.strip():
                yield position, sentence.strip()


def assess_extractive_evidence(
    question: str,
    context_chunks: Sequence[Dict[str, Any]],
    *,
    candidate_limit: int = 8,
) -> ExtractiveEvidence:
    """Measure whether retrieved text has enough lexical overlap to answer.

    This function is deliberately side-effect free. The caller may use the top
    candidate sentences for a bounded semantic rescue when a genuinely relevant
    question has low literal overlap (for example, a paraphrase or wrong-premise
    question). The returned coverage is a development-set signal, not a claim of
    universal answer accuracy.
    """
    terms = _query_terms(question)
    ranked: list[tuple[float, int, int, str]] = []
    seen: set[str] = set()
    for position, sentence in _iter_chunk_sentences(context_chunks):
        normalized = _SPACE_RE.sub("", sentence).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        hits = _term_hits(terms, sentence)
        coverage = hits / max(1, len(terms))
        ranked.append((coverage, hits, -position, sentence))

    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    limit = max(1, int(candidate_limit))
    candidates = tuple(item[3] for item in ranked[:limit])
    best_coverage, best_hits, _negative_position, best_sentence = (
        ranked[0] if ranked else (0.0, 0, 0, "")
    )
    return ExtractiveEvidence(
        query_term_count=len(terms),
        best_term_hits=best_hits,
        best_query_coverage=round(best_coverage, 6),
        best_sentence=best_sentence,
        candidate_sentences=candidates,
    )


def select_passages(
    question: str,
    context_chunks: Sequence[Dict[str, Any]],
    *,
    max_chunks: int = 3,
    max_sentences_per_chunk: int = 2,
    max_chars_per_passage: int = 260,
    min_term_hits: int = 1,
    min_query_coverage: float = 0.0,
) -> List[str]:
    """Select the most query-relevant source sentences from each chunk.

    `min_term_hits` is the relevance floor: when no retrieved chunk shares
    at least this many query terms (surface tokens/bigrams), the function
    returns an empty list so the caller can fall back to the honest
    "no relevant content" answer instead of quoting an unrelated passage.
    Set to 0 to restore the previous "always return the top chunk" behavior.
    """
    terms = _query_terms(question)
    selected: List[str] = []
    seen: set[str] = set()

    evidence = assess_extractive_evidence(question, context_chunks)
    coverage_floor = max(0.0, float(min_query_coverage))
    if evidence.query_term_count and evidence.best_query_coverage < coverage_floor:
        return []

    chunks = list(context_chunks)
    chunk_scores = [
        _term_hits(terms, str(chunk.get("content") or "")) for chunk in chunks
    ]
    # 相关性兜底：所有命中块的词面重叠都不足阈值时，直接拒答，绝不硬答。
    if chunk_scores and max(chunk_scores, default=0) < max(0, int(min_term_hits)):
        return []
    has_relevant_chunk = any(score > 0 for score in chunk_scores)
    if has_relevant_chunk:
        ordered = sorted(
            enumerate(zip(chunk_scores, chunks)),
            key=lambda item: (-item[1][0], item[0]),
        )
        chunks = [
            chunk
            for _position, (score, chunk) in ordered
            if score > 0
        ]

    for chunk in chunks[: max(1, int(max_chunks))]:
        index = int(chunk.get("index") or 0)
        if index <= 0:
            continue
        content = str(chunk.get("content") or "").strip()
        # strip breadcrumb prefix and Markdown headings to avoid title noise
        header = _LEADING_HEADER_RE.match(content)
        if header:
            content = content[header.end():].lstrip()
        content = _strip_markdown_headings(content).strip()
        sentences = _split_sentences(content)
        if not sentences:
            sentences = [content[:max_chars_per_passage].strip()] if content else []

        ranked: list[tuple[float, int, str]] = []
        for position, sentence in enumerate(sentences):
            hits = _term_hits(terms, sentence)
            coverage = hits / max(1, len(terms))
            # A small position prior keeps useful opening sentences when a
            # short query has no overlap (for example, a pronoun-only query).
            score = coverage + (0.015 if position == 0 else 0.0)
            ranked.append((score, position, sentence))

        # If some sentences literally match the question, do not dilute the
        # answer with an unrelated heading or neighbouring paragraph.
        if any(_term_hits(terms, sentence) > 0 for _, _, sentence in ranked):
            ranked = [
                item for item in ranked if _term_hits(terms, item[2]) > 0
            ]

        chosen: list[tuple[int, str]] = []
        for _score, position, sentence in sorted(
            ranked, key=lambda item: (-item[0], item[1])
        )[: max(1, int(max_sentences_per_chunk))]:
            chosen.append((position, sentence))

        for _position, sentence in sorted(chosen):
            passage = sentence[:max_chars_per_passage].strip()
            normalized = _SPACE_RE.sub("", passage).lower()
            if not passage or normalized in seen:
                continue
            if any(normalized in prev for prev in seen):
                continue
            seen.add(normalized)
            terminal = passage[-1:] if passage[-1:] in "。！？!?；;.:：" else "。"
            passage = passage[:-1].rstrip() if passage[-1:] == terminal else passage.rstrip()
            selected.append(f"- {passage} [{index}]{terminal}")

    return selected


def build_extractive_answer(
    question: str,
    context_chunks: Sequence[Dict[str, Any]],
    *,
    max_chunks: int = 3,
    max_sentences_per_chunk: int = 2,
    max_chars_per_passage: int = 260,
    min_term_hits: int = 1,
    min_query_coverage: float = 0.0,
) -> str:
    """Return a compact evidence answer where every fact carries a citation."""
    passages = select_passages(
        question,
        context_chunks,
        max_chunks=max_chunks,
        max_sentences_per_chunk=max_sentences_per_chunk,
        max_chars_per_passage=max_chars_per_passage,
        min_term_hits=min_term_hits,
        min_query_coverage=min_query_coverage,
    )
    return "\n".join(passages)


def iter_answer_chunks(answer: str, max_chars: int = 120) -> Iterable[str]:
    """Yield readable chunks for the existing streaming UI."""
    text = answer or ""
    for start in range(0, len(text), max(1, int(max_chars))):
        yield text[start : start + max_chars]


__all__ = [
    "ExtractiveEvidence",
    "assess_extractive_evidence",
    "build_extractive_answer",
    "iter_answer_chunks",
    "select_passages",
]
