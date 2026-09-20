"""中文智能文本分块器 - 优化版。

优化要点：
1. 递归字符切分 + 中文标点优先（。！？；，）
2. chunk_size 300-500，chunk_overlap 50-80（中文经验值）
3. 支持 Markdown 标题层级切分
4. 语义完整性优先，避免句子中间切断
5. 支持自定义词典加载

推荐配置：
    chunk_size: 400（中文 300-500 为宜）
    chunk_overlap: 50（必须有重叠，防止边界语义丢失）
    separators: ["\n\n", "\n", "。", "！", "？", "；", "，", " "]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

from .utils import get_logger, resolve_path

logger = get_logger("text_splitter")


def load_custom_dict_from_file(dict_path: str) -> List[str]:
    """从文件加载自定义词典（每行一个词，支持 # 注释）。"""
    try:
        path = resolve_path(dict_path)
        if not path.exists():
            logger.warning(f"自定义词典文件不存在: {dict_path}")
            return []
        
        words = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith('#'):
                    words.append(word)
        
        logger.info(f"从 {dict_path} 加载了 {len(words)} 个自定义词典词")
        return words
    except Exception as exc:
        logger.warning(f"加载自定义词典失败: {exc}")
        return []

# 中文标点（按语义完整性排序）
_CHINESE_PUNCTUATION = "。！？；"
# 英文句末标点
_ENGLISH_PUNCTUATION = ".!?"
# 所有句末标点
_ALL_SENTENCE_END = _CHINESE_PUNCTUATION + _ENGLISH_PUNCTUATION

# 默认分隔符（中文标点优先）
DEFAULT_SEPARATORS = [
    "\n\n",      # 段落分隔（最高优先级）
    "\n",        # 换行
    "。",        # 句号
    "！",        # 感叹号
    "？",        # 问号
    "；",        # 分号
    "，",        # 逗号
    ". ",        # 英文句点+空格
    " ",         # 空格
    "",          # 强制按字符切分（最后手段）
]


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


# 小数点占位符（切句时避免 "3.14" 被切断）
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_DECIMAL_DOT_TOKEN = "\x00"


def _split_into_sentences(text: str) -> List[str]:
    """按中英文句末标点切分句子（小数点不做切分边界）。"""
    text = _DECIMAL_DOT_RE.sub(_DECIMAL_DOT_TOKEN, text.strip())
    if not text:
        return []
    
    # 找所有句子
    sentences = []
    pattern = rf'([^{re.escape(_ALL_SENTENCE_END)}\n]*[{re.escape(_ALL_SENTENCE_END)}])'
    parts = re.findall(pattern, text)
    
    consumed = sum(len(p) for p in parts)
    tail = text[consumed:].strip()
    if tail:
        parts.append(tail)
    
    for p in parts:
        s = p.replace(_DECIMAL_DOT_TOKEN, ".").strip()
        if s:
            sentences.append(s)
    
    return sentences


def _merge_short_paragraphs(paragraphs: List[str], min_len: int = 20) -> List[str]:
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
    """面向中文 RAG 的智能分块器 - 优化版。

    核心改进：
    1. 中文标点优先切分（。！？；，）
    2. 递归切分确保语义完整
    3. 智能 overlap 保持上下文连贯

    Args:
        chunk_size: 单块最大字符数（推荐 300-500）
        chunk_overlap: 块间重叠字符数（推荐 50-80）
        separators: 自定义分隔符（按优先级排序）
        keep_separator: 是否在分块中保留分隔符
        min_chunk_size: 最小块长度
        custom_dict_words: 自定义词典词汇（用于更好的分词）
    """

    DEFAULT_SEPARATORS = DEFAULT_SEPARATORS

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        min_chunk_size: int = 32,
        custom_dict_words: Optional[List[str]] = None,
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
        self.custom_dict_words: Set[str] = set(custom_dict_words or [])
        
        # 加载自定义词典到 jieba（如果有）
        if self.custom_dict_words:
            self._load_custom_dict()

    def _load_custom_dict(self) -> None:
        """加载自定义词典到 jieba。"""
        try:
            import jieba
            for word in self.custom_dict_words:
                jieba.add_word(word)
            logger.info(f"已加载 {len(self.custom_dict_words)} 个自定义词典词")
        except ImportError:
            logger.warning("jieba 未安装，自定义词典将不生效")
        except Exception as exc:
            logger.warning(f"加载自定义词典失败: {exc}")

    def _split_by_separators(self, text: str) -> List[str]:
        """按分隔符递归切分文本。"""
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """递归切分文本。"""
        if not text:
            return []
        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        sep = separators[0]
        if not sep or sep not in text:
            return self._recursive_split(text, separators[1:])
        
        splits = text.split(sep)
        result: List[str] = []
        
        for s in splits:
            s = s.strip()
            if not s:
                continue
            # 如果片段太长，继续切分
            if len(s) <= self.chunk_size:
                result.append(s + (sep if self.keep_separator and sep != " " else ""))
            else:
                result.extend(self._recursive_split(s, separators[1:]))
        
        return result

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """对单段文本分块。

        切分策略：
        1. 先按 Markdown 标题切分（保留标题上下文）
        2. 按段落 (\n\n) 切分
        3. 按句末标点切分
        4. 按 chunk_size 打包

        Args:
            text: 完整文本（一般为一个 Document.content）
            metadata: 透传到每个 Chunk 的元信息

        Returns:
            Chunk 列表
        """
        metadata = dict(metadata or {})
        text = (text or "").strip()
        if not text:
            return []

        # 1. 检测并处理 Markdown 标题结构
        sections = self._split_by_markdown_headings(text)
        
        chunks_text: List[str] = []
        for section in sections:
            # 2. 在每个 section 内按段落切分
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", section) if p.strip()]
            paragraphs = _merge_short_paragraphs(paragraphs, min_len=self.min_chunk_size)
            
            # 3. 段内按句子拆分
            for para in paragraphs:
                sentences = _split_into_sentences(para)
                if not sentences and para:
                    sentences = [para]
                
                # 4. 按 chunk_size 贪心打包
                buf = ""
                for sent in sentences:
                    # 单句超长处理
                    if len(sent) > self.chunk_size:
                        if buf:
                            chunks_text.append(buf)
                            buf = ""
                        # 长句按 chunk_size 强制切分（保留最小语义单元）
                        for i in range(0, len(sent), self.chunk_size - self.chunk_overlap):
                            piece = sent[i:i + self.chunk_size]
                            chunks_text.append(piece)
                        continue
                    
                    # 加入当前块检查
                    new_len = len(buf) + len(sent) + (1 if buf else 0)
                    if new_len <= self.chunk_size:
                        buf = (buf + "\n" + sent) if buf else sent
                    else:
                        if buf:
                            chunks_text.append(buf)
                        buf = sent
                
                if buf:
                    chunks_text.append(buf)

        # 5. 合并过短的块
        if self.min_chunk_size > 0:
            chunks_text = self._merge_short_chunks(chunks_text)

        # 6. 处理 overlap
        if self.chunk_overlap > 0 and len(chunks_text) > 1:
            chunks_text = self._add_overlap(chunks_text)

        # 构建 Chunk 对象
        result: List[Chunk] = []
        for i, ck in enumerate(chunks_text):
            result.append(Chunk(text=ck, index=i, metadata=dict(metadata)))

        logger.debug("分块：文本 → %d 个分块", len(result))
        return result

    def _split_by_markdown_headings(self, text: str) -> List[str]:
        """按 Markdown 标题切分，保持标题上下文。"""
        # Markdown 标题模式 (# ## ### 等)
        heading_pattern = r'(^#{1,6}\s+.+)$'
        
        # 找出所有标题位置
        matches = list(re.finditer(heading_pattern, text, re.MULTILINE))
        
        if not matches:
            return [text]
        
        sections = []
        prev_end = 0
        
        for match in matches:
            start = match.start()
            # 标题前的内容
            if start > prev_end:
                content = text[prev_end:start].strip()
                if content:
                    sections.append(content)
            
            # 标题本身（作为新section的开始）
            heading = match.group(1)
            sections.append(heading)
            prev_end = match.end()
        
        # 最后一个标题后的内容
        if prev_end < len(text):
            content = text[prev_end:].strip()
            if content:
                sections.append(content)
        
        return sections if sections else [text]

    def _merge_short_chunks(self, chunks: List[str]) -> List[str]:
        """合并过短的相邻块。"""
        if not chunks:
            return []
            
        merged: List[str] = []
        buf = chunks[0] if chunks else ""
        
        for ck in chunks[1:]:
            if len(ck) < self.min_chunk_size:
                # 短块尝试合并
                if len(buf) + len(ck) + 1 <= self.chunk_size + self.chunk_overlap:
                    buf = buf + "\n" + ck
                else:
                    merged.append(buf)
                    buf = ck
            else:
                merged.append(buf)
                buf = ck
        
        if buf:
            merged.append(buf)
        
        return merged

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加块间 overlap，保持上下文连贯。"""
        if len(chunks) <= 1:
            return chunks
            
        overlapped: List[str] = []
        overlapped.append(chunks[0])
        
        for i in range(1, len(chunks)):
            prev = overlapped[-1]
            # 取前一块的末尾 overlap 字符
            tail = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            
            # 合并
            merged = tail + "\n" + chunks[i]
            
            # 如果超长，压缩 tail 而不是丢弃 chunks[i] 的内容
            max_len = self.chunk_size + self.chunk_overlap
            if len(merged) > max_len:
                keep = max_len - len(chunks[i]) - 1
                if keep > 0:
                    merged = tail[-keep:] + "\n" + chunks[i]
                else:
                    merged = chunks[i]
            
            overlapped.append(merged)
        
        return overlapped


class RecursiveTextSplitter:
    """通用回退分块器：按优先级递归使用分隔符。

    与 ChineseTextSplitter 类似，但使用不同的默认分隔符策略。
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
        self.separators = separators or DEFAULT_SEPARATORS
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
            chunks = self._add_overlap(chunks)

        return [Chunk(text=ck, index=i, metadata=dict(metadata)) for i, ck in enumerate(chunks)]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        sep = separators[0]
        if not sep or sep not in text:
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

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加块间 overlap。"""
        if len(chunks) <= 1:
            return chunks
            
        overlapped = [chunks[0]]
        for ck in chunks[1:]:
            tail = overlapped[-1][-self.chunk_overlap:]
            merged = (tail + "\n" + ck) if tail else ck
            max_len = self.chunk_size + self.chunk_overlap
            if len(merged) > max_len:
                keep = max_len - len(ck) - 1
                if keep > 0:
                    merged = tail[-keep:] + "\n" + ck
                else:
                    merged = ck
            overlapped.append(merged)
        
        return overlapped
