"""多格式文档解析器。

支持的格式：
- PDF  ：使用 pdfplumber（对中文更友好）或 pypdf 作为后备
- DOCX ：使用 python-docx
- PPTX ：使用 python-pptx（幻灯片逐页提取，表格转 Markdown 表）
- XLSX/XLS ：使用 openpyxl（工作表逐表提取，表格转 Markdown 表）
- CSV  ：标准库 csv（表格转 Markdown 表，自动探测编码/分隔符）
- HTML ：使用 beautifulsoup4（剥离导航/页脚噪音，保留标题层级）
- MD   ：使用 markdown 库转 HTML，再剥离标签保留纯文本
- TXT  ：直接按行读取，按段落合并

表格统一转 Markdown 表（`| a | b |`），让检索/生成侧看到结构；
HTML 转纯文本时保留 `#` 标题标记，供后续标题切分恢复层级。

统一返回 ``Document`` 列表，每份 ``Document`` 包含：
    page      : int            页码（txt/md 时为 1；pptx=幻灯片号；xlsx=工作表序号）
    content   : str            正文内容
    metadata  : dict           源文件、文件名、扩展名等
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .utils import IMAGE_EXTENSIONS, clean_text, get_logger, resolve_path

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
_OCR_ENGINE = None
_OCR_TRIED = False


def _get_ocr_engine():
    """懒加载 OCR 引擎（RapidOCR，纯 CPU 可跑）。未安装时返回 None。

    OCR 是**可选能力**：不装依赖时整个流程照常工作，只是扫描件提不出文本。
    RapidOCR 基于 onnxruntime，约 100 MB，远轻于 PaddleOCR，适合本地部署。
    """
    global _OCR_ENGINE, _OCR_TRIED
    if _OCR_TRIED:
        return _OCR_ENGINE
    _OCR_TRIED = True
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        _OCR_ENGINE = RapidOCR()
        logger.info("OCR 引擎就绪：RapidOCR（CPU 推理）")
    except Exception as exc:  # noqa: BLE001
        _OCR_ENGINE = None
        logger.info(
            "扫描件 OCR 不可用（%s）。需要时执行：pip install -r requirements-ocr.txt",
            type(exc).__name__,
        )
    return _OCR_ENGINE


def ocr_available() -> bool:
    """OCR 依赖是否可用（供 doctor / API 做能力提示）。"""
    return _get_ocr_engine() is not None


def _import_pymupdf():
    """导入 PyMuPDF，优先用 1.24+ 的新名 ``pymupdf``，回退老版本的 ``fitz``。

    直接 ``import fitz`` 在新版本会打印弃用警告，污染日志。
    """
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except ImportError:
        pass
    try:
        import fitz  # type: ignore

        return fitz
    except ImportError:
        return None


def _pdf_page_count(path: Path, docs: List[Document]) -> Optional[int]:
    """推断 PDF 总页数：优先用解析结果里的 page_count，否则用 PyMuPDF 数。"""
    for d in docs:
        count = d.metadata.get("page_count")
        if count:
            try:
                return int(count)
            except (TypeError, ValueError):
                break
    pymupdf = _import_pymupdf()
    if pymupdf is None:
        return None
    try:
        with pymupdf.open(str(path)) as doc:
            return int(doc.page_count)
    except Exception:  # noqa: BLE001
        return None


def _ocr_pdf_page(path: Path, page_no: int, engine, dpi: int = 200) -> str:
    """把 PDF 的第 ``page_no`` 页渲染成图片后 OCR，返回识别出的文本。"""
    try:
        import io

        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        pymupdf = _import_pymupdf()
        if pymupdf is None:
            return ""
        with pymupdf.open(str(path)) as doc:
            if page_no < 1 or page_no > doc.page_count:
                return ""
            pix = doc.load_page(page_no - 1).get_pixmap(dpi=dpi)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        result, _ = engine(np.array(image))
    except Exception as exc:  # noqa: BLE001
        logger.warning("第 %d 页 OCR 失败：%s", page_no, exc)
        return ""
    if not result:
        return ""
    # RapidOCR 返回形如 [[box, text, score], ...]
    return "\n".join(str(item[1]) for item in result if len(item) > 1)


def _augment_pdf_with_ocr(path: Path, docs: List[Document], ocr_cfg: Dict) -> List[Document]:
    """对「文本过少」的页做 OCR 补齐。

    逐页判断而不是"整份文件失败才 OCR"：这样**混合型 PDF**
    （正文页可正常提取、附件页是扫描图）也能完整入库。
    """
    if not ocr_cfg.get("enabled", True):
        return docs
    engine = _get_ocr_engine()
    if engine is None:
        return docs

    min_chars = int(ocr_cfg.get("min_text_chars", 20) or 20)
    dpi = int(ocr_cfg.get("dpi", 200) or 200)
    max_pages = int(ocr_cfg.get("max_pages", 50) or 50)

    total = _pdf_page_count(path, docs)
    if not total:
        return docs

    ok_pages = {d.page for d in docs if len(d.content.strip()) >= min_chars}
    missing = [p for p in range(1, total + 1) if p not in ok_pages]
    if not missing:
        return docs

    # 安全阀：一次上传不该把 CPU 占满
    if len(missing) > max_pages:
        logger.warning(
            "待 OCR 页数 %d 超过上限 %d，仅处理前 %d 页：%s",
            len(missing),
            max_pages,
            max_pages,
            path.name,
        )
        missing = missing[:max_pages]

    by_page = {d.page: d for d in docs}
    added = 0
    for page_no in missing:
        text = _ocr_pdf_page(path, page_no, engine, dpi=dpi)
        if not text.strip():
            continue
        metadata = {
            "source": path.name,
            "filepath": str(path),
            "ext": path.suffix.lower(),
            "page_count": total,
            "ocr": True,  # 标记来源，便于溯源与质量评估
            "ocr_engine": "rapidocr",
        }
        existing = by_page.get(page_no)
        if existing is not None:
            existing.content = clean_text(text)
            existing.metadata.update(metadata)
        else:
            docs.append(
                Document(page=page_no, content=clean_text(text), metadata=metadata)
            )
        added += 1

    if added:
        docs.sort(key=lambda d: d.page)
        logger.info("OCR 补齐 %d/%d 页（%s）", added, len(missing), path.name)
    return docs


def _ocr_image_file(path, engine, max_side=3000):
    """对单张图片文件做 OCR，返回识别出的文本。

    RapidOCR 直接吃 numpy 数组。这里用 PIL 读文件而不是 cv2.imread，
    因为 Windows 下 cv2.imread 对非 ASCII 路径会返回 None，中文文件名会失败。
    超长边先等比缩放，避免超大图占满内存。
    """
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        image = Image.open(str(path)).convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / longest
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale)))
            )
        result, _ = engine(np.array(image))
    except Exception as exc:  # noqa: BLE001
        logger.warning("图片 OCR 失败（%s）：%s", path.name, exc)
        return ""
    if not result:
        return ""
    # RapidOCR 返回形如 [[box, text, score], ...]
    return "\n".join(str(item[1]) for item in result if len(item) > 1)


def _load_image(path, ocr_cfg):
    """解析单张图片文件：用 RapidOCR 提取文字，作为单页 Document 返回。

    图片没有文本层，OCR 是唯一提取手段：引擎不可用或 OCR 被显式关闭时
    返回空列表（图片无法入库），由调用方决定提示。
    """
    ocr_cfg = ocr_cfg or {}
    if not ocr_cfg.get("enabled", True):
        logger.info("扫描件 OCR 已关闭，跳过图片解析：%s", path.name)
        return []
    engine = _get_ocr_engine()
    if engine is None:
        return []
    text = clean_text(_ocr_image_file(path, engine))
    if not text:
        return []
    return [
        Document(
            page=1,
            content=text,
            metadata={
                "source": path.name,
                "filepath": str(path),
                "ext": path.suffix.lower(),
                "page_count": 1,
                "ocr": True,
                "ocr_engine": "rapidocr",
            },
        )
    ]


# ----------------------------------------------------------------------
#  Xiaomi-OCR-0 / Any2Markdown 高质量版面解析（可选）
# ----------------------------------------------------------------------
def _load_pdf_via_xiaomi(path: Path, cfg: Dict) -> List[Document]:
    """用本地 Any2Markdown（默认 Xiaomi-OCR-0）把 PDF 解析成 Markdown，再转 Document。

    相比 pdfplumber/pypdf 的纯文本提取，这条路径能还原表格、公式与阅读顺序，
    是扫描件 / 复杂版面 PDF 的高质量兜底。

    设计：服务不可用 / 返回空时抛异常或返回空列表，由 ``_load_pdf`` 决定是否
    回落到传统引擎。本函数不引入 requests 等重依赖，全部标准库 + 现有 http_client。
    """
    import json as _json
    import time as _time
    import urllib.error
    import urllib.request

    from .http_client import (
        RemoteServiceError,
        describe_http_error,
        describe_url_error,
        opener_for,
    )

    base_url = str(cfg.get("base_url") or "http://127.0.0.1:8088").rstrip("/")
    parse_mode = str(cfg.get("parse_mode") or "ocr")
    model = str(cfg.get("model") or "xiaomi-ocr-0")
    timeout = float(cfg.get("timeout", 300.0) or 300.0)

    url = f"{base_url}/convert/file"
    fields: Dict[str, str] = {"parse_mode": parse_mode}
    if model:
        fields["model"] = model

    # 用标准库构造 multipart/form-data（项目远程调用统一不依赖 requests）
    boundary = "----zhige_xiaomi_%x" % (int(_time.time() * 1000) & 0xFFFFFFFF)
    payload = path.read_bytes()
    parts: List[bytes] = []
    for key, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)

    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    try:
        with opener_for(url).open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        raise RemoteServiceError(
            describe_http_error(exc, label="Xiaomi 解析", url=url, model=model)
        ) from exc
    except urllib.error.URLError as exc:
        raise RemoteServiceError(
            describe_url_error(exc, label="Xiaomi 解析", base_url=url, model=model)
        ) from exc

    # Any2Markdown 可能返回 JSON（含 markdown 字段）或纯 Markdown 文本，兼容两种
    markdown = ""
    try:
        decoded = _json.loads(raw)
        if isinstance(decoded, dict):
            markdown = str(decoded.get("markdown") or decoded.get("content") or "")
        else:
            markdown = raw
    except _json.JSONDecodeError:
        markdown = raw

    markdown = clean_text(markdown) if markdown else ""
    if not markdown:
        return []

    total = _pdf_page_count(path, [])
    # 整份 Markdown 作为一个 Document（page=1）；标题层级交给 text_splitter 恢复。
    # 需要页码级溯源时走原 pdfplumber 路径，或用 Any2Markdown 的 /convert/file/stream。
    return [
        Document(
            page=1,
            content=markdown,
            metadata={
                "source": str(path.name),
                "filepath": str(path),
                "ext": path.suffix.lower(),
                "page_count": total,
                "xiaomi_engine": True,
                "xiaomi_model": model,
                "parse_mode": parse_mode,
            },
        )
    ]


def _load_pdf(
    path: Path,
    engine: str = "pdfplumber",
    ocr_cfg: Optional[Dict] = None,
    xiaomi_cfg: Optional[Dict] = None,
) -> List[Document]:
    """解析 PDF，每页一个 Document（xiaomi 路径例外：整份一个 Document）。

    优先级：

    0. 可选 ``xiaomi`` 高质量版面解析（Any2Markdown / Xiaomi-OCR-0）；
    1. 主引擎 ``pdfplumber``（对中文更友好）；
    2. 主引擎无输出时，自动换备用引擎 ``pypdf`` 重试；
    3. 仍有「文本过少」的页 → OCR 兜底（需安装 ``requirements-ocr.txt``）。

    第 3 层让扫描件、以及"正文可提取 + 附件页是扫描图"的混合型 PDF 都能入库；
    第 0 层是更强的兜底：一次调用直接产出带表格/公式/阅读顺序的 Markdown。
    """
    # 0. Xiaomi-OCR-0 / Any2Markdown（服务不可用时回退，不中断入库）
    if xiaomi_cfg and xiaomi_cfg.get("enabled"):
        try:
            xdocs = _load_pdf_via_xiaomi(path, xiaomi_cfg)
        except Exception as exc:  # noqa: BLE001 - 远程服务异常必须回退
            logger.warning("Xiaomi 解析失败，回退传统引擎（%s）: %s", type(exc).__name__, exc)
            xdocs = []
        if xdocs:
            return xdocs
        if not xiaomi_cfg.get("fallback_on_error", True):
            logger.warning("Xiaomi 解析无产出且已关闭回退：%s", path.name)
            return []

    docs = _load_pdf_with_engine(path, engine)
    if not docs:
        other = "pypdf" if engine == "pdfplumber" else "pdfplumber"
        logger.warning("PDF 引擎 %s 未提取到内容，回退到 %s 重试: %s", engine, other, path.name)
        docs = _load_pdf_with_engine(path, other)

    docs = _augment_pdf_with_ocr(path, docs, ocr_cfg or {})

    if not docs:
        logger.warning(
            "PDF 未提取到任何文本：%s（若是扫描件，请安装 OCR 依赖，见 requirements-ocr.txt）",
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
            rows = [
                [re.sub(r"\s+", " ", (cell.text or "").strip()) for cell in row.cells]
                for row in tbl.rows
            ]
            # 表格转标准 Markdown 表（| a | b |），检索/生成侧能识别列结构；
            # 不再用 " | " 拼成一行让列结构丢失（XLSX/CSV 已是此处理，DOCX 补齐）。
            md_table = _rows_to_markdown_table(rows)
            if md_table:
                blocks.append((None, md_table))
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
#  表格 → Markdown（PPTX / XLSX / CSV 共用）
# ----------------------------------------------------------------------
def _rows_to_markdown_table(rows: List[List[str]], max_rows: int = 100,
                             max_cols: int = 20) -> str:
    """二维行转 Markdown 表，超限截断并标注。"""
    rows = [[(c or "").strip() for c in r[:max_cols]] for r in rows[:max_rows]]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    normed = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(normed[0]) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    for r in normed[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------
#  PPTX
# ----------------------------------------------------------------------
def _load_pptx(path: Path) -> List[Document]:
    """逐幻灯片提取：文本框按阅读顺序 + 表格转 Markdown 表。"""
    docs: List[Document] = []
    try:
        from pptx import Presentation  # type: ignore
    except ImportError:
        logger.warning("未安装 python-pptx，跳过 %s（pip install python-pptx）", path.name)
        return []
    try:
        prs = Presentation(str(path))
        for i, slide in enumerate(prs.slides, start=1):
            parts: List[str] = []
            for shape in slide.shapes:
                try:
                    if shape.has_table:
                        rows = [[cell.text or "" for cell in row.cells]
                                for row in shape.table.rows]
                        tbl = _rows_to_markdown_table(rows)
                        if tbl:
                            parts.append(tbl)
                    elif shape.has_text_frame:
                        t = (shape.text or "").strip()
                        if t:
                            parts.append(t)
                except Exception:
                    continue
            text = clean_text("\n\n".join(parts))
            if text:
                docs.append(Document(page=i, content=text, metadata={
                    "source": str(path.name), "filepath": str(path),
                    "ext": path.suffix.lower(), "slide": i,
                    "slide_count": len(prs.slides)}))
    except Exception as exc:
        logger.error("解析 PPTX 失败 %s: %s", path, exc)
    return docs


# ----------------------------------------------------------------------
#  XLSX / XLS
# ----------------------------------------------------------------------
def _load_excel(path: Path) -> List[Document]:
    """逐工作表提取为 Markdown 表，每表一个 Document（page=表序号）。"""
    docs: List[Document] = []
    try:
        import openpyxl  # type: ignore
    except ImportError:
        logger.warning("未安装 openpyxl，跳过 %s（pip install openpyxl）", path.name)
        return []
    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        for i, ws in enumerate(wb.worksheets, start=1):
            rows: List[List[str]] = []
            for row in ws.iter_rows(values_only=True):
                vals = [(str(v).strip() if v is not None else "") for v in row]
                if any(vals):
                    rows.append(vals)
                if len(rows) >= 500:
                    break
            if not rows:
                continue
            tbl = _rows_to_markdown_table(rows, max_rows=500)
            title = str(ws.title or f"Sheet{i}")
            text = clean_text(f"# {title}\n\n{tbl}" if tbl else f"# {title}")
            if text.strip("# \n"):
                docs.append(Document(page=i, content=text, metadata={
                    "source": str(path.name), "filepath": str(path),
                    "ext": path.suffix.lower(), "sheet": title,
                    "sheet_index": i, "title": title}))
        try:
            wb.close()
        except Exception:
            pass
    except Exception as exc:
        logger.error("解析 Excel 失败 %s: %s", path, exc)
    return docs


# ----------------------------------------------------------------------
#  CSV
# ----------------------------------------------------------------------
def _load_csv(path: Path, encoding: str = "auto") -> List[Document]:
    """CSV 转 Markdown 表（自动探测编码与分隔符）。"""
    docs: List[Document] = []
    try:
        import csv as _csv
        import io
        data = path.read_bytes()
        if encoding and encoding.lower() != "auto":
            text = data.decode(encoding, errors="replace")
            used_enc = encoding
        else:
            text, used_enc = _decode_text_bytes(data)
        sample = text[:4096]
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            delim = dialect.delimiter
        except Exception:
            delim = ","
        reader = _csv.reader(io.StringIO(text), delimiter=delim)
        rows = [[(c or "").strip() for c in r] for r in reader]
        rows = [r for r in rows if any(r)][:2000]
        if not rows:
            return []
        tbl = _rows_to_markdown_table(rows, max_rows=2000)
        if tbl:
            docs.append(Document(page=1, content=clean_text(tbl), metadata={
                "source": str(path.name), "filepath": str(path),
                "ext": path.suffix.lower(), "encoding": used_enc,
                "delimiter": delim, "rows": len(rows)}))
    except Exception as exc:
        logger.error("解析 CSV 失败 %s: %s", path, exc)
    return docs


# ----------------------------------------------------------------------
#  HTML（本地 .html 文件）
# ----------------------------------------------------------------------
def _load_html(path: Path, encoding: str = "auto") -> List[Document]:
    """本地 HTML 转纯文本：剥离 nav/footer/script，标题转 # 标记。"""
    docs: List[Document] = []
    try:
        data = path.read_bytes()
        if encoding and encoding.lower() != "auto":
            text = data.decode(encoding, errors="replace")
        else:
            text, _ = _decode_text_bytes(data)
    except OSError as exc:
        logger.error("读取 HTML 失败 %s: %s", path, exc)
        return []
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        lines: List[str] = []
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "pre"]):
            t = el.get_text(" ", strip=True)
            if not t:
                continue
            if el.name.startswith("h"):
                level = int(el.name[1])
                lines.append("#" * level + " " + t)
            else:
                lines.append(t)
        body = soup.body
        plain = clean_text("\n".join(lines) or (body.get_text("\n", strip=True) if body else ""))
        if plain:
            docs.append(Document(page=1, content=plain, metadata={
                "source": str(path.name), "filepath": str(path),
                "ext": path.suffix.lower(),
                "title": title or path.stem}))
    except ImportError:
        logger.warning("未安装 beautifulsoup4，跳过 %s", path.name)
    except Exception as exc:
        logger.error("解析 HTML 失败 %s: %s", path, exc)
    return docs


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
    ocr_cfg: Optional[Dict] = None,
    xiaomi_cfg: Optional[Dict] = None,
) -> List[Document]:
    """根据文件扩展名分发到对应解析器。

    Args:
        file_path: 文件路径。
        pdf_engine: PDF 解析引擎，可选 pdfplumber / pypdf（主引擎失败自动互为回退）。
        encoding: TXT 编码，``auto`` 时按 utf-8-sig → gb18030 → big5 自动探测。
        clean: 是否清洗文本。
        ocr_cfg: 扫描件 OCR 配置（对应 ``config.document_loader.ocr``）。缺省则不启用 OCR。
        xiaomi_cfg: Xiaomi-OCR-0 / Any2Markdown 高质量版面解析配置
            （对应 ``config.document_loader.xiaomi``）。缺省则不启用，仅 PDF 生效。

    Returns:
        Document 列表（解析失败时返回空列表）。
    """
    path = resolve_path(file_path)
    if not path.exists():
        logger.warning("文件不存在: %s", path)
        return []

    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = _load_pdf(path, engine=pdf_engine, ocr_cfg=ocr_cfg, xiaomi_cfg=xiaomi_cfg)
    elif ext == ".docx":
        docs = _load_docx(path)
    elif ext in (".md", ".markdown"):
        docs = _load_markdown(path)
    elif ext == ".txt":
        docs = _load_txt(path, encoding=encoding)
    elif ext in (".pptx", ".ppt"):
        docs = _load_pptx(path)
    elif ext in (".xlsx", ".xls"):
        docs = _load_excel(path)
    elif ext == ".csv":
        docs = _load_csv(path, encoding=encoding)
    elif ext in (".html", ".htm", ".xhtml"):
        docs = _load_html(path, encoding=encoding)
    elif ext in IMAGE_EXTENSIONS:
        docs = _load_image(path, ocr_cfg)
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
    ocr_cfg: Optional[Dict] = None,
    xiaomi_cfg: Optional[Dict] = None,
) -> List[Document]:
    """批量解析目录下所有受支持文档。"""
    from .utils import iter_doc_files

    root = resolve_path(root)
    all_docs: List[Document] = []
    for f in iter_doc_files(root, recursive=recursive):
        all_docs.extend(
            load_document(
                f,
                pdf_engine=pdf_engine,
                encoding=encoding,
                ocr_cfg=ocr_cfg,
                xiaomi_cfg=xiaomi_cfg,
            )
        )
    logger.info("目录 %s 共解析 %d 个 Document", root, len(all_docs))
    return all_docs
