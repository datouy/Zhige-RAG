"""数据合成 pipeline 的单元测试（不依赖真实 LLM）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.synthesize import (
    ExtractedKnowledge,
    QuestionRecord,
    _build_questions,
    _safe_parse_extraction,
    _split_sections,
    load_markdown,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture()
def tmp_doc(tmp_path: Path) -> Path:
    md = tmp_path / "sample.md"
    md.write_text(
        "# 标题\n\n## RAG 核心概念\n\nRAG = 检索增强生成。\n\n特点：可解释、减少幻觉。\n\n## 向量检索\n\n向量检索用于相似性匹配。\n",
        encoding="utf-8",
    )
    return md


# ----------------------------------------------------------------------
# 基础工具函数
# ----------------------------------------------------------------------
def test_split_sections_returns_sections(tmp_doc: Path):
    docs = load_markdown(tmp_doc)
    assert docs
    sections = _split_sections(docs[0])
    titles = [t for t, _ in sections]
    assert "RAG 核心概念" in titles
    assert "向量检索" in titles


# ----------------------------------------------------------------------
# 实体抽取解析
# ----------------------------------------------------------------------
def test_safe_parse_extraction_parses_json():
    raw = '{"entities": ["RAG", "Embedding"], "definition": "RAG=检索增强生成", "key_facts": ["知识更新快", "减少幻觉"]}'
    out = _safe_parse_extraction(raw)
    assert out.entities == ["RAG", "Embedding"]
    assert out.key_facts[:2] == ["知识更新快", "减少幻觉"]


def test_safe_parse_extraction_handles_code_fence():
    raw = '```json\n{"entities": ["RAG"], "definition": "", "key_facts": []}\n```'
    out = _safe_parse_extraction(raw)
    assert out.entities == ["RAG"]


def test_safe_parse_extraction_invalid_returns_empty():
    assert _safe_parse_extraction("not json").entities == []


# ----------------------------------------------------------------------
# 题目生成
# ----------------------------------------------------------------------
def test_template_easy_fills_entity():
    import random
    random.seed(7)   # yields easy_count>=1, medium_count=0
    knowledge = ExtractedKnowledge(entities=["RAG"], definition="检索增强生成", key_facts=["知识更新快", "可解释性强"])
    qs = _build_questions(knowledge, "RAG 核心概念", ["sample.md"])
    easy_qs = [q for q in qs if q.difficulty == "easy"]
    assert easy_qs
    assert any("RAG" in q.question for q in easy_qs)
    assert any(q.template_id.startswith("tmpl_") for q in easy_qs)


def test_template_medium_fills_two_entities():
    import random
    random.seed(3)   # yields easy_count>=1, medium_count>=1
    knowledge = ExtractedKnowledge(entities=["RAG", "微调"], definition="", key_facts=["知识更新快"])
    qs = _build_questions(knowledge, "RAG 核心概念", ["sample.md"])
    medium_qs = [q for q in qs if q.difficulty == "medium"]
    assert medium_qs
    # comparison template (tmpl_5/6) contains both; summary template (tmpl_9) contains at least one
    assert any(
        ("RAG" in q.question and "微调" in q.question) or ("RAG" in q.question)
        for q in medium_qs
    )


def test_deduplication_prevents_duplicate_questions():
    import random
    random.seed(42)
    knowledge = ExtractedKnowledge(entities=["RAG"], definition="", key_facts=["知识更新快"])
    q1 = _build_questions(knowledge, "RAG 核心概念", ["sample.md"])[0]
    random.seed(42)
    q2 = _build_questions(knowledge, "RAG 核心概念", ["sample.md"])[0]
    seen = {hash(q1.question)}
    assert hash(q2.question) in seen


def test_fallback_generated_when_no_entity():
    knowledge = ExtractedKnowledge(entities=[], definition="", key_facts=[])
    qs = _build_questions(knowledge, "RAG 核心概念", ["sample.md"])
    assert len(qs) == 1
    assert qs[0].difficulty == "medium"
    assert "核心内容是什么" in qs[0].question


def test_output_format_fields():
    knowledge = ExtractedKnowledge(entities=["RAG"], definition="", key_facts=["知识更新快"])
    rec = _build_questions(knowledge, "RAG 核心概念", ["sample.md"])[0]
    data = rec.to_dict()
    assert "question" in data
    assert "expected_sources" in data
    assert "expected_keywords" in data
    assert "difficulty" in data
    assert "source_section" in data
    assert "template_id" in data
    assert data["difficulty"] in {"easy", "medium"}


def test_max_questions_respected():
    """采集超过 max 时截断到 max 条。"""
    import random
    random.seed(0)
    # 大量 section → 大量候选
    all_qs: list[QuestionRecord] = []
    for _ in range(10):
        knowledge = ExtractedKnowledge(
            entities=["RAG", "Embedding", "向量检索"],
            definition="",
            key_facts=["知识更新快"],
        )
        all_qs.extend(_build_questions(knowledge, "RAG 核心概念", ["sample.md"]))
    max_q = 5
    truncated = all_qs[:max_q]
    assert len(truncated) == max_q
