"""data_quality 模块测试：深度清洗 / 去重 / 准入门 / 统一管线。"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from src.data_quality import (
    content_hash,
    deep_clean_text,
    dedup_chunks,
    dedup_documents,
    gate_documents,
    run_data_pipeline,
)
from src.document_loader import Document


# ----------------------------------------------------------------------
#  深度清洗
# ----------------------------------------------------------------------
class TestDeepClean:
    def test_removes_page_numbers(self):
        text = "第一段内容。\n\n第 3 页\n第二段内容。\n\n- 4 -\n第三段。"
        cleaned = deep_clean_text(text)
        assert "第一段内容" in cleaned
        assert "第 3 页" not in cleaned
        assert "- 4 -" not in cleaned

    def test_removes_repeated_headers(self):
        pages = [
            "机密文件 内部资料\n正文一的内容。",
            "机密文件 内部资料\n正文二的内容。",
            "机密文件 内部资料\n正文三的内容。",
        ]
        cleaned = deep_clean_text("\n\n".join(pages))
        assert cleaned.count("机密文件") == 0
        assert "正文一" in cleaned and "正文三" in cleaned

    def test_keeps_single_page_content(self):
        # 单块文档不做重复行剔除，避免误删
        text = "重复行示例\n重复行示例\n重复行示例"
        cleaned = deep_clean_text(text)
        assert "重复行示例" in cleaned

    def test_removes_decorative_lines(self):
        text = "标题\n----------\n正文内容。"
        cleaned = deep_clean_text(text)
        assert "----------" not in cleaned
        assert "正文内容" in cleaned

    def test_empty(self):
        assert deep_clean_text("") == ""


# ----------------------------------------------------------------------
#  去重
# ----------------------------------------------------------------------
def _doc(text: str, source: str = "a.md") -> Document:
    return Document(page=1, content=text, metadata={"source": source})


class TestDedup:
    def test_content_hash_normalizes_whitespace(self):
        assert content_hash("你好 世界") == content_hash("你好世界")
        assert content_hash("a") != content_hash("b")

    def test_document_exact_dedup(self):
        docs = [_doc("内容A", "1.md"), _doc("内容B", "2.md"), _doc("内容A", "3.md")]
        kept, dups = dedup_documents(docs)
        assert dups == 1
        assert len(kept) == 2
        assert kept[0].metadata["content_hash"]

    def test_chunk_exact_dedup(self):
        chunks = [
            SimpleNamespace(text="相同的分块内容", metadata={}),
            SimpleNamespace(text="不同的分块内容", metadata={}),
            SimpleNamespace(text="相同的分块内容", metadata={}),
        ]
        kept, dropped = dedup_chunks(chunks)
        assert dropped == 1
        assert len(kept) == 2

    def test_chunk_near_dedup(self):
        base = "这是一段足够长的正文内容用于测试近重复检测逻辑" * 3
        near = base[:-3] + "微调"
        chunks = [SimpleNamespace(text=base, metadata={}), SimpleNamespace(text=near, metadata={})]
        kept, dropped = dedup_chunks(chunks, near_dup_threshold=0.9)
        assert dropped == 1
        assert kept[0].text == base

    def test_near_dedup_disabled_by_default(self):
        base = "这是一段足够长的正文内容用于测试近重复检测逻辑" * 3
        near = base[:-3] + "微调"
        chunks = [SimpleNamespace(text=base, metadata={}), SimpleNamespace(text=near, metadata={})]
        kept, _ = dedup_chunks(chunks)
        assert len(kept) == 2


# ----------------------------------------------------------------------
#  准入门
# ----------------------------------------------------------------------
def _doc_with_meta(**meta) -> Document:
    base = {"source": "doc.md", "title": "标题", "breadcrumb": "标题"}
    base.update(meta)
    return Document(page=1, content="正文内容", metadata=base)


class TestGate:
    def test_rejects_draft_and_expired_status(self):
        docs = [
            _doc_with_meta(doc_status="active"),
            _doc_with_meta(doc_status="draft", source="b.md"),
            _doc_with_meta(doc_status="expired", source="c.md"),
        ]
        kept, report = gate_documents(docs, {"allowed_status": ["active"]})
        assert len(kept) == 1
        assert report.rejected == 2
        assert report.reasons and "status=draft" in str(report.reasons)

    def test_rejects_unclear_acl_when_required(self):
        docs = [_doc_with_meta(acl="*")]
        kept, report = gate_documents(docs, {"require_explicit_acl": True})
        assert not kept
        assert "权限不清" in report.details[0]

    def test_keeps_star_acl_by_default(self):
        docs = [_doc_with_meta(acl="*")]
        kept, _ = gate_documents(docs, {})
        assert len(kept) == 1

    def test_rejects_multi_group_acl(self):
        docs = [_doc_with_meta(acl="hr,admin")]
        kept, report = gate_documents(docs, {"reject_multi_group_acl": True})
        assert not kept
        assert "多分组" in report.details[0]

    def test_valid_until_in_past_rejected(self):
        docs = [_doc_with_meta(valid_until="2020-01-01")]
        kept, report = gate_documents(docs, {})
        assert not kept
        assert "有效期" in report.details[0]

    def test_expiry_days(self):
        old = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d %H:%M:%S")
        fresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        docs = [_doc_with_meta(updated_at=old, source="old.md"), _doc_with_meta(updated_at=fresh, source="new.md")]
        kept, report = gate_documents(docs, {"expiry_days": 365})
        assert [d.metadata["source"] for d in kept] == ["new.md"]
        assert "内容过期" in report.details[0]

    def test_no_expiry_check_by_default(self):
        old = (datetime.now() - timedelta(days=4000)).strftime("%Y-%m-%d %H:%M:%S")
        docs = [_doc_with_meta(updated_at=old)]
        kept, _ = gate_documents(docs, {})
        assert len(kept) == 1


# ----------------------------------------------------------------------
#  统一管线
# ----------------------------------------------------------------------
class _StubSplitter:
    """按行分块的桩分块器，验证管线衔接。"""

    def split_text(self, text, metadata=None):
        return [SimpleNamespace(text=line, metadata=dict(metadata or {})) for line in text.splitlines() if line.strip()]


class TestRunPipeline:
    def test_end_to_end_with_breadcrumb_and_gate(self):
        docs = [
            _doc_with_meta(breadcrumb="手册 > 考勤", doc_status="active"),
            _doc_with_meta(source="b.md", doc_status="draft", breadcrumb="通知"),
        ]
        docs[1].content = "另一份草稿内容"  # 内容不同，确保是准入门而非文档去重拦截
        cfg = {"data_quality": {"deep_clean": False, "breadcrumb_in_chunk": True, "gate": {}}}
        chunks, report = run_data_pipeline(docs, cfg, _StubSplitter())
        # 草稿被拒绝，只剩 1 个小节的 1 行
        assert len(chunks) == 1
        assert chunks[0].text.startswith("【手册 > 考勤】")
        assert report["gate"]["rejected"] == 1
        assert report["chunks"] == 1

    def test_breadcrumb_disabled(self):
        docs = [_doc_with_meta()]
        cfg = {"data_quality": {"deep_clean": False, "breadcrumb_in_chunk": False, "gate": {}}}
        chunks, _ = run_data_pipeline(docs, cfg, _StubSplitter())
        assert chunks[0].text == "正文内容"

    def test_chunk_dedup_across_sections(self):
        # 文档级内容不同（doc 去重不触发），但共享相同的行 → 分块级去重生效
        docs = [
            _doc_with_meta(source="a.md", breadcrumb="A"),
            _doc_with_meta(source="b.md", breadcrumb="B"),
        ]
        docs[0].content = "第一节独有的开头\n两节共享的结尾句"
        docs[1].content = "第二节独有的开头\n两节共享的结尾句"
        cfg = {"data_quality": {"deep_clean": False, "gate": {}}}
        chunks, report = run_data_pipeline(docs, cfg, _StubSplitter())
        assert report["doc_dedup_dropped"] == 0
        assert report["chunk_dedup_dropped"] == 1
        assert len(chunks) == 3
