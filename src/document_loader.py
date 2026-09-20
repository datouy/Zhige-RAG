"""多格式文档解析器。

支持的格式：
- PDF  ：使用 pdfplumber（对中文更友好）或 pypdf 作为后备
- DOCX ：使用 python-docx
- MD   ：使用 markdown 库转 HTML，再剥离标签保留纯文本
- TXT  ：直接按行读取，按段落合并

统一返回 ``Document`` 列表，每份 ``Document`` 包含：
    page      : int            页码（txt/md 时为 1）
    content   : str            正文内容
    metadata  : dict           源文件、文件名、扩展名等
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .utils import clean_text, get_logger, resolve_path

logger = get_logger("document_loader")


@dataclass
class Document:
    """统一文档数据结构。

    Attributes:
        page: 页码（单文件内从 1 计数；无页概念时为 1）。
        content: 段落正文，已做基础清洗。
        metadata: 元信息，包括 source、filename、ext、title 等。
    """

    page: int
    content: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"page": self.page, "content": self.content, "metadata": dict(self.metadata)}


# ----------------------------------------------------------------------
#  PDF
# ----------------------------------------------------------------------
def _load_pdf(path: Path, engine: str = "pdfplumber") -> List[Document]:
    """解析 PDF，每页一个 Document。

    主引擎无输出（异常或全部页未提取到文本）时自动切换备用引擎重试一次；
    两个引擎都失败大概率是扫描件，明确告警提示需要 OCR，而不是静默返回空。
    """
    docs = _load_pdf_with_engine(path, engine)
    if not docs:
        other = "pypdf" if engine == "pdfplumber" else "pdfplumber"
        logger.warning("PDF 引擎 %s 未提取到内容，回退到 %s 重试: %s", engine, other, path.name)
        docs = _load_pdf_with_engine(path, other)
    if not docs:
        logger.warning(
            "PDF 两个引擎均未提取到文本，可能是扫描件（图片型 PDF 需 OCR 后再入库）: %s",
            path.name,
        )
    return docs


def _load_pdf_with_engine(path: Path, engine: str) -> List[Document]:
    """用指定引擎解析 PDF。"""
    docs: List[Document] = []
    try:
        if engine == "pdfplumber":
            import pdfplumber  # type: ignore

            with pdfplumber.open(str(path)) as pdf:
                total = len(pdf.pages)
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    text = clean_text(text)
                    if not text:
                        continue
                    docs.append(
                        Document(
                            page=i,
                            content=text,
                            metadata={
                                "source": str(path.name),
                                "filepath": str(path),
                                "ext": path.suffix.lower(),
                                "page_count": total,
                            },
                        )
                    )
        else:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            total = len(reader.pages)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = clean_text(text)
                if not text:
                    continue
                docs.append(
                    Document(
                        page=i,
                        content=text,
                        metadata={
                            "source": str(path.name),
                            "filepath": str(path),
                            "ext": path.suffix.lower(),
                            "page_count": total,
                        },
                    )
                )
    except Exception as exc:
        logger.error("解析 PDF 失败 %s: %s", path, exc)
    return docs


# ----------------------------------------------------------------------
#  DOCX
# ----------------------------------------------------------------------
# DOCX 标题样式（python-docx 的 style.name，兼容英文/中文 Word 模板）
_DOCX_HEADING_RE = re.compile(r"^(?:heading|标题)\s*(\d+)$", re.IGNORECASE)


def _iter_docx_blocks_ordered(path: Path) -> List[tuple]:
    """按文档顺序产出 DOCX 块：(heading_level 或 None, text)。

    通过 body 子元素顺序遍历，让表格出现在其真实位置——旧实现把所有
    表格行追加到全文末尾，导致表格内容与文档尾部章节混入同一分块。
    标题级别由段落样式名（Heading N / 标题 N）识别，供按章节切分使用。
    """
    from docx import Document as DocxDocument  # type: ignore
    from docx.table import Table  # type: ignore
    from docx.text.paragraph import Paragraph  # type: ignore

    d = DocxDocument(str(path))
    blocks: List[tuple] = []
    for child in d.element.body.iterchildren():
        tag = child.tag if isinstance(child.tag, str) else ""
        if tag.endswith("}p"):
            para = Paragraph(child, d)
            text = (para.text or "").strip()
            if not text:
                continue
            level: Optional[int] = None
            try:
                style_name = (para.style.name or "") if para.style is not None else ""
            except Exception:
                style_name = ""
            hm = _DOCX_HEADING_RE.match(style_name.strip())
            if hm:
                level = int(hm.group(1))
            blocks.append((level, text))
        elif tag.endswith("}tbl"):
            tbl = Table(child, d)
            for row in tbl.rows:
                row_text = " | ".join((cell.text or "").strip() for cell in row.cells)
                if row_text.strip(" |"):
                    blocks.append((None, row_text))
    return blocks


def _load_docx(path: Path) -> List[Document]:
    """解析 Word 文档，按段落聚合为单个 Document。"""
    docs: List[Document] = []
    try:
        blocks = _iter_docx_blocks_ordered(path)
        paragraphs = [t for _, t in blocks]
        full_text = clean_text("\n".join(paragraphs))
        if full_text:
            docs.append(
                Document(
                    page=1,
                    content=full_text,
                    metadata={
                        "source": str(path.name),
                        "filepath": str(path),
                        "ext": path.suffix.lower(),
                        "paragraph_count": len(paragraphs),
                    },
                )
            )
    except Exception as exc:
        logger.error("解析 DOCX 失败 %s: %s", path, exc)
    return docs


# ----------------------------------------------------------------------
#  Markdown
# ----------------------------------------------------------------------
_MD_TAG_RE = re.compile(r"<[^>]+>")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s*(.+?)\s*$", re.MULTILINE)


def _md_to_text(md_text: str) -> str:
    """将 Markdown 转为保留结构的纯文本。

    规则：
    - 标题行保留为 `## 标题`
    - 链接转为可见文本 `[文本](url)` → `文本`
    - 代码块保留原文
    - HTML 标签被剥离
    """
    # 抽取代码块并占位
    code_blocks = []

    def _code_sub(m: re.Match) -> str:
        code_blocks.append(m.group(0))
        return f"\n__CODE_BLOCK_{len(code_blocks) - 1}__\n"

    md_text = re.sub(r"```.*?```", _code_sub, md_text, flags=re.DOTALL)
    md_text = re.sub(r"`[^`]+`", lambda m: m.group(0)[1:-1], md_text)
    md_text = _MD_LINK_RE.sub(r"\1", md_text)
    md_text = _MD_TAG_RE.sub("", md_text)
    # 还原代码块
    for i, blk in enumerate(code_blocks):
        inner = re.sub(r"^```[^\n]*\n|```$", "", blk.strip())
        md_text = md_text.replace(f"__CODE_BLOCK_{i}__", f"\n{inner}\n")
    return clean_text(md_text)


def _load_markdown(path: Path) -> List[Document]:
    docs: List[Document] = []
    try:
        text = path.read_text(encoding="utf-8")
        plain = _md_to_text(text)
        if not plain:
            return docs

        # 抽取第一个标题作为文档标题
        m = _MD_HEADING_RE.search(text)
        title = m.group(1).strip() if m else path.stem

        docs.append(
            Document(
                page=1,
                content=plain,
                metadata={
                    "source": str(path.name),
                    "filepath": str(path),
                    "ext": path.suffix.lower(),
                    "title": title,
                },
            )
        )
    except Exception as exc:
        logger.error("解析 Markdown 失败 %s: %s", path, exc)
    return docs


# 公开别名，便于外部调用
load_markdown = _load_markdown


# ----------------------------------------------------------------------
#  TXT
# ----------------------------------------------------------------------
# 编码探测链：中文场景最常见的乱码来源是 GBK/GB2312 文件被硬按 UTF-8 读。
# 逐个"严格"解码尝试，全部失败再用 UTF-8 宽松兜底（坏字节替换为 �，
# 至少保住文件名等可用信息，不致整篇 mojibake 混入知识库）。
_TXT_ENC_CANDIDATES = ("utf-8-sig", "gb18030", "big5")


def _decode_text_bytes(data: bytes) -> tuple[str, str]:
    """按候选编码链解码字节流，返回 (文本, 实际使用的编码)。"""
    for enc in _TXT_ENC_CANDIDATES:
        try:
            return data.decode(enc), enc
        except (UnicodeDecodeError, ValueError):
            continue
    return data.decode("utf-8", errors="replace"), "utf-8(replace)"


def _load_txt(path: Path, encoding: str = "auto") -> List[Document]:
    docs: List[Document] = []
    try:
        data = path.read_bytes()
        if encoding and encoding.lower() != "auto":
            text = data.decode(encoding, errors="replace")
            used_enc = encoding
        else:
            text, used_enc = _decode_text_bytes(data)
        text = clean_text(text)
        if text:
            docs.append(
                Document(
                    page=1,
                    content=text,
                    metadata={
                        "source": str(path.name),
                        "filepath": str(path),
                        "ext": path.suffix.lower(),
                        "encoding": used_enc,
                    },
                )
            )
    except Exception as exc:
        logger.error("解析 TXT 失败 %s: %s", path, exc)
    return docs


# ----------------------------------------------------------------------
#  统一入口
# ----------------------------------------------------------------------
def load_document(
    file_path: str | Path,
    pdf_engine: str = "pdfplumber",
    encoding: str = "auto",
    clean: bool = True,
) -> List[Document]:
    """根据文件扩展名分发到对应解析器。

    Args:
        file_path: 文件路径。
        pdf_engine: PDF 解析引擎，可选 pdfplumber / pypdf（主引擎失败自动互为回退）。
        encoding: TXT 编码，``auto`` 时按 utf-8-sig → gb18030 → big5 自动探测。
        clean: 是否清洗文本。

    Returns:
        Document 列表（解析失败时返回空列表）。
    """
    path = resolve_path(file_path)
    if not path.exists():
        logger.warning("文件不存在: %s", path)
        return []

    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = _load_pdf(path, engine=pdf_engine)
    elif ext == ".docx":
        docs = _load_docx(path)
    elif ext in (".md", ".markdown"):
        docs = _load_markdown(path)
    elif ext == ".txt":
        docs = _load_txt(path, encoding=encoding)
    else:
        logger.warning("不支持的扩展名 %s，已跳过: %s", ext, path)
        return []

    if clean:
        for d in docs:
            d.content = clean_text(d.content)
    logger.info("解析 %s 得到 %d 个 Document", path.name, len(docs))
    return docs


def load_directory(
    root: str | Path,
    pdf_engine: str = "pdfplumber",
    recursive: bool = True,
    encoding: str = "utf-8",
) -> List[Document]:
    """批量解析目录下所有受支持文档。"""
    from .utils import iter_doc_files

    root = resolve_path(root)
    all_docs: List[Document] = []
    for f in iter_doc_files(root, recursive=recursive):
        all_docs.extend(load_document(f, pdf_engine=pdf_engine, encoding=encoding))
    logger.info("目录 %s 共解析 %d 个 Document", root, len(all_docs))
    return all_docs