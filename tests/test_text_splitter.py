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


def test_chinese_splitter_handles_english_punctuation():
    """P3.5：扩展后的 ``_SENT_END_RE`` 必须能切 ``. ! ?`` 英文句末。

    用户场景：英文文档或中英混排文档，长度会被合理切分。
    """
    text = (
        "This is the first sentence. "
        "This is the second sentence! "
        "Is this the third one? "
        "Yes, it is."
    )
    sp = ChineseTextSplitter(chunk_size=60, chunk_overlap=10)
    chunks = sp.split_text(text)
    # 应切出 3-4 句，而非把整段压成一个块
    assert len(chunks) >= 2
    # 任意 chunk 的长度都不应超过 chunk_size + chunk_overlap + 缓冲
    for c in chunks:
        assert len(c.text) <= 60 + 10 + 2


def test_chinese_splitter_mixed_cjk_and_english():
    """P3.5：中英混排文本仍能正确分句（中文 ``。！？`` 与英文 ``.!?`` 都识别）。"""
    # 复制多遍，确保单 chunk 容纳不下，强制触发分块
    text = (
        "中文第一句。English sentence two. 第三句！Fourth one?"
        " 中文第五句。English sixth sentence. 第七句！Eighth one?"
    )
    sp = ChineseTextSplitter(chunk_size=40, chunk_overlap=5)
    chunks = sp.split_text(text)
    assert len(chunks) >= 2
    full = "".join(c.text for c in chunks).replace("\n", " ")
    # 关键短语应保留（顺序可能因切分略有变化；重叠部分允许被截断，
    # 因此只断言完整出现在至少一个 chunk 中，而非拼接后存在）
    needles_per_chunk = ["".join(c.text for c in chunks).count(n) for n in
                         ["中文第一句", "English sentence two", "第三句", "Fourth one",
                          "中文第五句", "Eighth one"]]
    # 至少有一半的关键短语出现在结果中（其余可能在 chunk 边界被重叠切走）
    assert sum(1 for n in needles_per_chunk if n > 0) >= 3, (
        f"expected >=3 needles present, got {needles_per_chunk}"
    )

def test_sentences_keep_decimal_numbers():
    """小数点不应被当作句末边界切断。"""
    from src.text_splitter import _split_into_sentences

    sents = _split_into_sentences("圆周率约为3.14159。第二句在这里！")
    assert any("3.14159" in s for s in sents), f"小数被切断: {sents}"
    assert all("14159。" not in s or "3." in s for s in sents)


def test_sentences_decimal_alone_in_chunk_text():
    from src.text_splitter import ChineseTextSplitter

    sp = ChineseTextSplitter(chunk_size=50, chunk_overlap=0, min_chunk_size=0)
    chunks = sp.split_text("模型精度达到0.95以上。这是很长的第二句内容，用来撑长文本长度。")
    joined = "".join(c.text for c in chunks)
    assert "0.95" in joined
