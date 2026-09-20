"""document_meta 模块测试：frontmatter / 标题层级切分 / 元数据归一。"""

from __future__ import annotations

from pathlib import Path

from src.document_meta import (
    _normalize_acl,
    _normalize_status,
    enrich_file,
    parse_frontmatter,
)


MD_WITH_FM = """---
title: 员工手册
permissions: hr
status: 草稿
department: 人力资源部
valid_until: 2099-12-31
---

# 员工手册

欢迎阅读本手册。

## 考勤制度

工作时间为每天 8 小时。

### 请假流程

请假需提前一天申请。
"""

MD_PLAIN = """# 简单文档

这是没有 frontmatter 的文档。
"""


class TestFrontmatter:
    def test_parse_basic(self):
        meta, body = parse_frontmatter(MD_WITH_FM)
        assert meta["title"] == "员工手册"
        assert meta["permissions"] == "hr"
        assert body.lstrip().startswith("# 员工手册")

    def test_no_frontmatter(self):
        meta, body = parse_frontmatter(MD_PLAIN)
        assert meta == {}
        assert body == MD_PLAIN


class TestNormalize:
    def test_acl(self):
        assert _normalize_acl(None) == "*"
        assert _normalize_acl("*") == "*"
        assert _normalize_acl("hr") == "hr"
        assert _normalize_acl("hr, admin") == "hr,admin"
        assert _normalize_acl(["a", "b"]) == "a,b"

    def test_status(self):
        assert _normalize_status("草稿") == "draft"
        assert _normalize_status("draft") == "draft"
        assert _normalize_status("expired") == "expired"
        assert _normalize_status("过期") == "expired"
        assert _normalize_status("final") == "active"
        assert _normalize_status("") == "active"


class TestEnrichFile:
    def test_md_sections_with_hierarchy(self, tmp_path: Path):
        p = tmp_path / "handbook.md"
        p.write_text(MD_WITH_FM, encoding="utf-8")
        docs = enrich_file(p)
        # 3 个小节：# 员工手册 正文 / ## 考勤制度 / ### 请假流程
        assert len(docs) == 3
        first, atten, leave = docs
        assert first.metadata["title"] == "员工手册"  # 来自 frontmatter
        assert first.metadata["acl"] == "hr"
        assert first.metadata["doc_status"] == "draft"
        assert first.metadata["department"] == "人力资源部"
        assert first.metadata["valid_until"] == "2099-12-31"
        # 层级路径
        assert first.metadata["heading_path"] == ["员工手册"]
        assert atten.metadata["heading_path"] == ["员工手册", "考勤制度"]
        assert leave.metadata["heading_path"] == ["员工手册", "考勤制度", "请假流程"]
        # breadcrumb
        assert atten.metadata["breadcrumb"] == "员工手册 > 考勤制度"
        # page 为小节序号（保证分块 id 不冲突）
        assert [d.page for d in docs] == [1, 2, 3]
        # 更新时间已写入
        assert first.metadata["updated_at"]

    def test_md_plain_single_section(self, tmp_path: Path):
        p = tmp_path / "plain.md"
        p.write_text(MD_PLAIN, encoding="utf-8")
        docs = enrich_file(p)
        assert len(docs) == 1
        assert docs[0].metadata["title"] == "简单文档"
        assert docs[0].metadata["acl"] == "*"
        assert docs[0].metadata["doc_status"] == "active"

    def test_txt_fallback(self, tmp_path: Path):
        p = tmp_path / "note.txt"
        p.write_text("第一行内容\n第二行内容", encoding="utf-8")
        docs = enrich_file(p)
        assert len(docs) == 1
        assert docs[0].metadata["ext"] == ".txt"
        assert docs[0].metadata["title"] == "note"

    def test_missing_file(self, tmp_path: Path):
        assert enrich_file(tmp_path / "nope.md") == []

    def test_default_acl_applied(self, tmp_path: Path):
        p = tmp_path / "x.md"
        p.write_text("# X\n内容", encoding="utf-8")
        docs = enrich_file(p, default_acl="legal")
        assert docs[0].metadata["acl"] == "legal"


def test_pdf_without_outline_still_enriches(tmp_path):
    """无书签 PDF 应优雅退化（无章节路径），不报错、不改变旧行为。"""
    import pytest

    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    pdf_path = tmp_path / "plain.pdf"
    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    with open(pdf_path, "wb") as f:
        w.write(f)

    from src.document_meta import _pdf_outline_paths, enrich_file

    assert _pdf_outline_paths(pdf_path) == {}
    # 空白页无任何文本：两个引擎均无输出 → 按"疑似扫描件"告警并优雅返回空，
    # 而不是抛异常或静默产出坏分块
    docs = enrich_file(pdf_path)
    assert docs == []


def test_pdf_outline_paths_hierarchical(tmp_path):
    """带书签 PDF 应还原章节层级，且页码区间向后继承。"""
    import pytest

    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    w = PdfWriter()
    for _ in range(3):
        w.add_blank_page(width=595, height=842)
    ch1 = w.add_outline_item("第一章 总论", 0)
    w.add_outline_item("1.1 背景", 0, parent=ch1)
    w.add_outline_item("1.2 目标", 1, parent=ch1)
    w.add_outline_item("第二章 设计", 2)
    pdf_path = tmp_path / "book.pdf"
    with open(pdf_path, "wb") as f:
        w.write(f)

    from src.document_meta import _pdf_outline_paths

    paths = _pdf_outline_paths(pdf_path)
    assert paths[0] == ["第一章 总论", "1.1 背景"]
    assert paths[1] == ["第一章 总论", "1.2 目标"]
    assert paths[2] == ["第二章 设计"]
