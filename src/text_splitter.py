"""中文智能文本分块器。

设计原则：
- 优先按段落（\\n\\n）切分
- 次之按句末标点（。！？；）切分
- 控制块大小在 ``chunk_size`` 以内，相邻块保留 ``chunk_overlap`` 重叠
- 避免在句子中间切断；若当前块已超阈值则强制按 ``chunk_size`` 切片

提供两个类：
- ``ChineseTextSplitter``：面向 RAG 的中文语义分块
- ``RecursiveTextSplitter``：通用回退分块器
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .utils import get_logger

logger = get_logger("text_splitter")


@dataclass
class Chunk:
    """分块结果。

    Attributes:
        text: 块文本。
        index: 块在原文档中的序号（从 0 开始）。
        metadata: 透传的元信息（如 page、source、title 等）。
    """

    text: str
    index: int
    metadata: dict

    def __len__(self) -> int:
        return len(self.text)


# 中文句末标点
_CHINESE_SENT_END = "。！？；"
_SENT_END_RE = re.compile(rf"([^{re.escape(_CHINESE_SENT_END)}\n]*[{_CHINESE_SENT_END}])")


def _split_into_sentences(text: str) -> List[str]:
    """按中英文句末标点切分句子。"""
    text = text.strip()
    if not text:
        return []
    parts = _SENT_END_RE.findall(text)
    # 若最后一句没有句末标点，findall 会漏掉，单独处理
    consumed = sum(len(p) for p in parts)
    tail = text[consumed:].strip()
    if tail:
        parts.append(tail)
    return [p.strip() for p in parts if p and p.strip()]


def _merge_short_paragraphs(paragraphs: List[str], min_len: int = 16) -> List[str]:
    """将过短的相邻段落合并，避免产生大量零碎块。"""
    merged: List[str] = []
    buf = ""
    for p in paragraphs:
        if not p:
            continue
        if not buf:
            buf = p
        elif len(buf) < min_len:
            buf = f"{buf}\n{p}"
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return merged


class ChineseTextSplitter:
    """面向中文 RAG 的智能分块器。

    Args:
        chunk_size: 单块最大字符数。
        chunk_overlap: 块间重叠字符数。
        separators: 自定义分隔符（按优先级排序）。
        keep_separator: 是否在分块中保留分隔符。
        min_chunk_size: 最小块长度，小于此长度的尾部块会并入前一块。
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，"]

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        min_chunk_size: int = 32,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须 > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须 >= 0 且 < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator
        self.min_chunk_size = min_chunk_size

    # ------------------------------------------------------------------
    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """对单段文本分块。

        Args:
            text: 完整文本（一般为一个 Document.content）。
            metadata: 透传到每个 Chunk 的元信息。

        Returns:
            Chunk 列表。
        """
        metadata = dict(metadata or {})
        text = (text or "").strip()
        if not text:
            return []

        # 1. 切分为段落
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        paragraphs = _merge_short_paragraphs(paragraphs, min_len=self.min_chunk_size)

        # 2. 段内按句子拆分
        sentence_pool: List[str] = []
        for para in paragraphs:
            sents = _split_into_sentences(para)
            if not sents and para:
                sents = [para]
            sentence_pool.extend(sents)

        # 3. 按 chunk_size 贪心打包
        chunks_text: List[str] = []
        buf = ""
        for sent in sentence_pool:
            # 单句超长：单独成块，再切到 chunk_size 以内
            if len(sent) > self.chunk_size:
                if buf:
                    chunks_text.append(buf)
                    buf = ""
                for i in range(0, len(sent), self.chunk_size):
                    piece = sent[i : i + self.chunk_size]
                    chunks_text.append(piece)
                continue
            # 加入当前块不会超长 → 累加
            if len(buf) + len(sent) + 1 <= self.chunk_size:
                buf = (buf + "\n" + sent) if buf else sent
            else:
                if buf:
                    chunks_text.append(buf)
                buf = sent
        if buf:
            chunks_text.append(buf)

        # 4. 处理重叠：若 overlap > 0，将上一块末尾 overlap 字符拼到下一块前面
        if self.chunk_overlap > 0 and len(chunks_text) > 1:
            max_len = self.chunk_size + self.chunk_overlap
            overlapped: List[str] = []
            for i, ck in enumerate(chunks_text):
                if i == 0:
                    overlapped.append(ck[:max_len])
                    continue
                prev = overlapped[-1]
                tail = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
                merged = (tail + "\n" + ck) if tail else ck
                # 合并后严格截断到 chunk_size + chunk_overlap
                if len(merged) > max_len:
                    merged = merged[:max_len]
                overlapped.append(merged)
            chunks_text = overlapped

        # 5. 过滤过短的块（但保留最后一块）。
        # 注意：必须在 step 4 重叠处理之前做最小块合并，否则合并后块长会超过 chunk_size+chunk_overlap。
        if self.min_chunk_size > 0:
            merged_short: List[str] = []
            for ck in chunks_text:
                if len(ck) < self.min_chunk_size and merged_short:
                    merged_short[-1] = merged_short[-1] + "\n" + ck
                else:
                    merged_short.append(ck)
            chunks_text = merged_short

        result: List[Chunk] = []
        for i, ck in enumerate(chunks_text):
            # 截断：单个块不应超过 chunk_size + chunk_overlap
            max_len = self.chunk_size + self.chunk_overlap
            if len(ck) > max_len:
                ck = ck[:max_len]
            result.append(Chunk(text=ck, index=len(result), metadata=dict(metadata)))

        logger.debug("分块：%d 段 → %d 块", len(sentence_pool), len(result))
        return result


class RecursiveTextSplitter:
    """通用回退分块器：按优先级递归使用分隔符。

    在 ``ChineseTextSplitter`` 表现不佳或语言为英文时使用。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "。", "！", "？", " ", ""]
        self.keep_separator = keep_separator

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = dict(metadata or {})
        pieces = self._recursive_split(text, self.separators)
        chunks: List[Chunk] = []
        buf = ""
        for p in pieces:
            if not p:
                continue
            if len(buf) + len(p) + 1 <= self.chunk_size:
                buf = (buf + "\n" + p) if buf else p
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)

        # 重叠
        if self.chunk_overlap > 0 and len(chunks) > 1:
            max_len = self.chunk_size + self.chunk_overlap
            overlapped = [chunks[0][:max_len]]
            for ck in chunks[1:]:
                tail = overlapped[-1][-self.chunk_overlap :]
                merged = (tail + "\n" + ck) if tail else ck
                if len(merged) > max_len:
                    merged = merged[:max_len]
                overlapped.append(merged)
            chunks = overlapped

        return [Chunk(text=ck, index=i, metadata=dict(metadata)) for i, ck in enumerate(chunks)]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        if not separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        sep = separators[0]
        if sep == "" or sep not in text:
            return self._recursive_split(text, separators[1:])
        splits = text.split(sep)
        result: List[str] = []
        for s in splits:
            s = s.strip()
            if not s:
                continue
            if len(s) <= self.chunk_size:
                result.append(s + (sep if self.keep_separator and sep != " " else ""))
            else:
                result.extend(self._recursive_split(s, separators[1:]))
        return result