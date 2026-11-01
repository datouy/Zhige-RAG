"""Build cited answers directly from retrieved passages, without an LLM."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_TERM_RE = re.compile(r"[A-Za-z0-9_+.-]+|[\u4e00-\u9fff]+")
_SPACE_RE = re.compile(r"\s+")
_SOURCE_RANGE_RE = re.compile(r"\[\d{1,3}\]\s*[-–—]\s*\[\d{1,3}\]")
_SOURCE_CITE_RE = re.compile(r"\[\d{1,3}\]")
_LEADING_HEADER_RE = re.compile(r"^\s*【[^】\n]{0,120}】\s*")
_MD_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s+.*$", re.MULTILINE)

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
    sentences = [match.group(0).strip() for match in _SENTENCE_RE.finditer(cleaned)]
    return [sentence for sentence in sentences if len(sentence) >= 8]


def _strip_markdown_headings(text: str) -> str:
    """Remove Markdown heading lines to avoid leaking titles into extracts."""
    return _MD_HEADING_LINE_RE.sub("", text or "")


def select_passages(
    question: str,
    context_chunks: Sequence[Dict[str, Any]],
    *,
    max_chunks: int = 3,
    max_sentences_per_chunk: int = 2,
    max_chars_per_passage: int = 260,
) -> List[str]:
    """Select the most query-relevant source sentences from each chunk."""
    terms = _query_terms(question)
    selected: List[str] = []
    seen: set[str] = set()

    chunks = list(context_chunks)
    chunk_scores = [
        _term_hits(terms, str(chunk.get("content") or "")) for chunk in chunks
    ]
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
) -> str:
    """Return a compact evidence answer where every fact carries a citation."""
    passages = select_passages(
        question,
        context_chunks,
        max_chunks=max_chunks,
        max_sentences_per_chunk=max_sentences_per_chunk,
        max_chars_per_passage=max_chars_per_passage,
    )
    return "\n".join(passages)


def iter_answer_chunks(answer: str, max_chars: int = 120) -> Iterable[str]:
    """Yield readable chunks for the existing streaming UI."""
    text = answer or ""
    for start in range(0, len(text), max(1, int(max_chars))):
        yield text[start : start + max_chars]


__all__ = ["build_extractive_answer", "iter_answer_chunks", "select_passages"]
