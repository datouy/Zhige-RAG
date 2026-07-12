"""中文分块器单元测试。"""

import pytest

from src.text_splitter import ChineseTextSplitter, RecursiveTextSplitter


def test_basic_chunk():
    text = "今天天气真好。适合出门散步！" * 20
    sp = ChineseTextSplitter(chunk_size=80, chunk_overlap=10)
    chunks = sp.split_text(text, metadata={"source": "t.txt"})
    assert len(chunks) >= 2
    assert all(c.metadata["source"] == "t.txt" for c in chunks)
    assert all(c.index == i for i, c in enumerate(chunks))


def test_long_sentence_force_split():
    """超长单句应被强制切片。"""
    text = "这是一段很长的文本。" + "中" * 1000
    sp = ChineseTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = sp.split_text(text)
    assert any(len(c.text) >= 100 for c in chunks)


def test_empty_text_returns_empty():
    sp = ChineseTextSplitter()
    assert sp.split_text("") == []
    assert sp.split_text("   \n\n  ") == []


def test_overlap_overflow_is_capped():
    text = "句子一。句子二。句子三。句子四。句子五。" * 5
    sp = ChineseTextSplitter(chunk_size=30, chunk_overlap=8)
    chunks = sp.split_text(text)
    for c in chunks:
        # 合并 overlap 后允许略大于 chunk_size，但有上限
        assert len(c.text) <= 30 + 8 + 5


def test_invalid_params():
    with pytest.raises(ValueError):
        ChineseTextSplitter(chunk_size=0)
    with pytest.raises(ValueError):
        ChineseTextSplitter(chunk_size=100, chunk_overlap=100)


def test_recursive_splitter_fallback():
    sp = RecursiveTextSplitter(chunk_size=50, chunk_overlap=5)
    chunks = sp.split_text("A" * 200, metadata={"source": "x"})
    assert len(chunks) >= 3
    assert all(c.metadata["source"] == "x" for c in chunks)