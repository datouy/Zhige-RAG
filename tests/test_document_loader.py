"""文档解析器单元测试。

仅依赖纯文本解析（不依赖真实 PDF/DOCX 文件），因此无需额外资源。
"""

from pathlib import Path

import pytest

from src.document_loader import load_document
from src.utils import resolve_path  # noqa: F401  (用于演示 import 路径)


def _write_tmp(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_load_txt(tmp_path):
    p = _write_tmp(tmp_path, "a.txt", "第一行\n第二行\n\n第三段")
    docs = load_document(p)
    assert docs, "应至少解析出一个 Document"
    assert "第一行" in docs[0].content
    assert docs[0].metadata["ext"] == ".txt"


def test_load_markdown(tmp_path):
    md = """# 标题一

正文段落。

## 小节

- 列表 1
- 列表 2

[链接](http://example.com)
"""
    p = _write_tmp(tmp_path, "a.md", md)
    docs = load_document(p)
    assert docs
    assert "标题一" in docs[0].content
    assert "链接" in docs[0].content  # 链接被还原为可见文本
    assert "列表 1" in docs[0].content


def test_load_unknown_ext_returns_empty(tmp_path):
    p = _write_tmp(tmp_path, "a.xyz", "x")
    docs = load_document(p)
    assert docs == []


def test_load_docx_if_available(tmp_path):
    """若安装了 python-docx，写一个最小 docx 验证解析。"""
    docx = pytest.importorskip("docx")
    from docx import Document  # type: ignore

    d = Document()
    d.add_heading("测试标题")
    d.add_paragraph("这是正文段落。")
    p = tmp_path / "t.docx"
    d.save(str(p))
    docs = load_document(p)
    assert docs
    assert "测试标题" in docs[0].content


def test_load_pdf_if_available(tmp_path):
    """若安装了 pdfplumber，构造最小 PDF 进行解析（需 pypdf/pdfplumber 库）。"""
    pytest.importorskip("pdfplumber")
    # 这里仅做冒烟测试：使用 reportlab 或最小 PDF 都可。简化处理：直接跳过构造。
    pytest.skip("PDF 构造需要 reportlab 等额外库，跳过自动测试")

def test_load_txt_gbk_auto_detected(tmp_path):
    """GBK 编码的 TXT 应被自动探测解码，而不是读成乱码。"""
    p = tmp_path / "gbk.txt"
    p.write_bytes("公司的年假制度：入职满一年可休五天。".encode("gbk"))
    from src.document_loader import _load_txt

    docs = _load_txt(p)
    assert docs, "GBK 文件应能解析出内容"
    assert "年假制度" in docs[0].content
    assert docs[0].metadata["encoding"] == "gb18030"


def test_load_txt_explicit_encoding_still_supported(tmp_path):
    """显式指定 encoding 时保持旧行为（按指定编码解码）。"""
    p = tmp_path / "u16.txt"
    p.write_bytes("你好知识库".encode("utf-16"))
    from src.document_loader import _load_txt

    docs = _load_txt(p, encoding="utf-16")
    assert docs and "知识库" in docs[0].content
