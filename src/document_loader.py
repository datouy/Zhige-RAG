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
    """解析 PDF，每页一个 Document。"""
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
def _load_docx(path: Path) -> List[Document]:
    """解析 Word 文档，按段落聚合为单个 Document。"""
    docs: List[Document] = []
    try:
        from docx import Document as DocxDocument  # type: ignore

        d = DocxDocument(str(path))
        paragraphs = []
        for p in d.paragraphs:
            t = (p.text or "").strip()
            if t:
                paragraphs.append(t)

        # 也尝试读取表格内容（可选）
        for tbl in d.tables:
            for row in tbl.rows:
                row_text = " | ".join((cell.text or "").strip() for cell in row.cells)
                if row_text.strip(" |"):
                    paragraphs.append(row_text)

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
def _load_txt(path: Path, encoding: str = "utf-8") -> List[Document]:
    docs: List[Document] = []
    try:
        text = path.read_text(encoding=encoding, errors="ignore")
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
    encoding: str = "utf-8",
    clean: bool = True,
) -> List[Document]:
    """根据文件扩展名分发到对应解析器。

    Args:
        file_path: 文件路径。
        pdf_engine: PDF 解析引擎，可选 pdfplumber / pypdf。
        encoding: TXT 文件编码。
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