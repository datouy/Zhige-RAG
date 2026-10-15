"""文档元数据增强：标题、层级、更新时间、权限、状态。

数据层的第一道工序——原始 ``document_loader`` 只产出"纯文本 + 基础来源
信息"，企业资料真正需要的是：

- **title**          ：文档标题（frontmatter > 首个标题 > 文件名）
- **heading_path**   ：分块所在章节的层级路径（如 ``["员工手册", "考勤", "请假"]``），
                       作为 breadcrumb 前缀写入分块文本，检索命中时保留章节语境
- **updated_at**     ：源文件最后修改时间（docx 读 core properties，其余取 mtime）
- **acl**            ：可见范围（frontmatter ``permissions``/``access``，逗号分隔；
                       ``*`` = 全员）。缺省 ``*``——是否强制显式权限由 DataGate 策略决定
- **doc_status**     ：draft / active / expired（frontmatter ``status``，默认 active）

Markdown 与 DOCX 会按标题**切分为多个小节**（每个小节一个 ``Document``），
而不是整篇压成一个大块——这是"合理切块"的前提：分块器在小节内工作，
天然不会把不相干的两个章节拼进同一块。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .document_loader import Document
from .utils import get_logger

logger = get_logger("document_meta")

# MD frontmatter 围栏
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
# MD 标题行
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# DOCX 标题样式（python-docx 的 style.name，兼容英文/中文 Word 模板）
_DOCX_HEADING_RE = re.compile(r"^(?:heading|标题)\s*(\d+)$", re.IGNORECASE)

# frontmatter 中可映射到权限的字段名（按优先级）
_ACL_KEYS = ("permissions", "access", "acl", "visible_to")
# 可映射到状态的字段
_STATUS_KEYS = ("status", "doc_status", "state")
_STATUS_VALUES = {"draft", "草稿", "final", "active", "生效", "expired", "过期", "archived"}

# 行首编号条目（如参考文献 "[12] Vaswani A, ..."）
_NUMBERED_ENTRY_RE = re.compile(r"^\[(\d{1,3})\]", re.MULTILINE)


def _numbered_entry_annot(text: str) -> Tuple[str, Optional[int]]:
    """检测"编号条目列表"小节（参考文献/附录清单等），产出 (计数注记, 条目数)。

    条目编号连续且无重复（如 [1]~[30] 全在）时才注记条数——这是对
    "引用了多少文献"这类聚合统计问题唯一可靠的线索：向量检索只会
    召回列表的部分分块，靠模型数碎片必然出错。
    """
    nums = {int(m) for m in _NUMBERED_ENTRY_RE.findall(text)}
    if len(nums) < 5:
        return "", None
    lo, hi = min(nums), max(nums)
    if hi - lo + 1 != len(nums):
        return "", None
    span = f"（[{lo}]-[{hi}]）" if hi > lo else f"（[{lo}]）"
    return f"本节为编号条目列表，共{len(nums)}条{span}", len(nums)


# ----------------------------------------------------------------------
#  Frontmatter（MD 顶部元信息块）
# ----------------------------------------------------------------------
def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """解析 MD 文件顶部的 ``---`` frontmatter，返回 (meta, 正文)。

    使用 YAML 解析以支持列表/日期；解析失败时按纯文本 ``key: value`` 降级。
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    body = text[m.end():]
    meta: Dict[str, Any] = {}
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw)
        if isinstance(loaded, dict):
            meta = {str(k): v for k, v in loaded.items()}
            return meta, body
    except Exception:
        pass
    # 降级：逐行 key: value
    for line in raw.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip().strip("\"'")
            if k and v:
                meta[k] = v
    return meta, body


def _normalize_acl(value: Any) -> str:
    """权限字段归一化为逗号分隔字符串；空值 → ``*``（全员可见）。"""
    if value is None:
        return "*"
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip() for v in value if str(v).strip()]
    else:
        items = [p.strip() for p in str(value).replace("，", ",").split(",") if p.strip()]
    if not items or items == ["*"] or items == ["all"]:
        return "*"
    # 去重保序
    seen: List[str] = []
    for it in items:
        if it not in seen:
            seen.append(it)
    return ",".join(seen)


def _normalize_status(value: Any) -> str:
    """状态字段归一化：draft/草稿 → draft；expired/过期/archived → expired；其余 → active。"""
    v = str(value or "").strip().lower()
    if v in {"draft", "草稿", "wip", "pending"}:
        return "draft"
    if v in {"expired", "过期", "archived", "作废", "deprecated"}:
        return "expired"
    return "active"


def _parse_valid_until(value: Any) -> Optional[datetime]:
    """解析 frontmatter 的 valid_until / expires_at 日期字段。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "date"):  # yaml 的 date 对象
        return datetime(value.year, value.month, value.day)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            continue
    return None


# ----------------------------------------------------------------------
#  标题层级切分
# ----------------------------------------------------------------------
def _split_markdown_sections(body: str, fallback_title: str) -> List[Tuple[List[str], str, str]]:
    """按 MD 标题切分小节。

    Returns:
        [(heading_path, section_title, section_text), ...]。
        无任何标题时返回单节：([], fallback_title, body)。
    """
    lines = body.splitlines()
    sections: List[Tuple[List[str], str, str]] = []
    stack: List[Tuple[int, str]] = []  # [(level, title)]
    buf: List[str] = []
    first_title: Optional[str] = None

    def _flush() -> None:
        text = "\n".join(buf).strip("\n")
        if not text.strip():
            buf.clear()
            return
        path = [t for _, t in stack]
        title = path[-1] if path else (first_title or fallback_title)
        sections.append((path, title, text))
        buf.clear()

    for line in lines:
        hm = _MD_HEADING_RE.match(line.strip())
        if hm:
            _flush()
            level = len(hm.group(1))
            title = hm.group(2).strip()
            if first_title is None:
                first_title = title
            # 维护层级栈：弹出所有 >= 当前级别 的标题
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            buf.append(line)  # 标题行保留在小节文本里，检索/生成时可见
        else:
            buf.append(line)
    _flush()

    if not sections:
        return [([], first_title or fallback_title, body.strip())]
    return sections


def _iter_docx_blocks(path: Path) -> List[Tuple[Optional[int], str]]:
    """按文档顺序产出 DOCX 段落/表格：(heading_level 或 None, text)。

    复用 ``document_loader._iter_docx_blocks_ordered``，保证表格插在其
    真实位置（旧实现把所有表格追加到全文末尾，导致表格行与尾章内容
    混入同一小节）。
    """
    from .document_loader import _iter_docx_blocks_ordered

    return _iter_docx_blocks_ordered(path)


def _split_docx_sections(
    path: Path, fallback_title: str
) -> Tuple[List[Tuple[List[str], str, str]], Dict[str, Any]]:
    """按 DOCX 标题样式切分小节；同时返回核心属性中的元数据。"""
    blocks = _iter_docx_blocks(path)
    sections: List[Tuple[List[str], str, str]] = []
    stack: List[Tuple[int, str]] = []
    buf: List[str] = []
    first_title: Optional[str] = None

    def _flush() -> None:
        text = "\n".join(buf).strip("\n")
        if not text.strip():
            buf.clear()
            return
        path_ = [t for _, t in stack]
        title = path_[-1] if path_ else (first_title or fallback_title)
        sections.append((path_, title, text))
        buf.clear()

    for level, text in blocks:
        if level is not None:
            _flush()
            if first_title is None:
                first_title = text
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            buf.append(text)
        else:
            buf.append(text)
    _flush()

    if not sections:
        sections = [([], fallback_title, "\n".join(t for _, t in blocks).strip())]

    core_meta: Dict[str, Any] = {}
    try:
        from docx import Document as DocxDocument  # type: ignore

        cp = DocxDocument(str(path)).core_properties
        if cp.modified:
            core_meta["updated_at"] = cp.modified
        if cp.title:
            core_meta["core_title"] = cp.title
        if cp.category:
            core_meta["department"] = cp.category
        if cp.keywords:
            core_meta["core_keywords"] = cp.keywords
    except Exception:
        pass
    return sections, core_meta


def _file_updated_at(path: Path) -> Optional[datetime]:
    """源文件最后修改时间（本地时区，去 tzinfo 便于入库）。"""
    try:
        st = path.stat()
        return datetime.fromtimestamp(st.st_mtime)
    except OSError:
        return None


def _pdf_outline_paths(path: Path) -> Dict[int, List[str]]:
    """读取 PDF 书签大纲，返回 {0 起始页码: heading_path}。

    带书签的 PDF（论文/报告/手册常见）通过大纲恢复章节层级，让 PDF 分块
    与 MD/DOCX 一样拥有 breadcrumb 语境；无书签或解析失败返回空 dict，
    行为与旧版一致（按页平铺，无章节路径）。
    """
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        flat: List[Tuple[int, int, str]] = []  # (层级, 起始页码0based, 标题)

        def _walk(items: Any, level: int = 1) -> None:
            for it in items or []:
                if isinstance(it, (list, tuple)):
                    _walk(it, level + 1)
                    continue
                try:
                    pg = reader.get_destination_page_number(it)
                except Exception:  # noqa: BLE001
                    pg = None
                title = str(getattr(it, "title", "") or "").strip()
                if title and pg is not None and pg >= 0:
                    flat.append((level, pg, title))

        outline = reader.outline
        if outline:
            _walk(outline)
        if not flat:
            return {}
        # 按页码稳定排序（同页书签保持大纲文档顺序），再维护层级栈生成
        # 每条书签的 heading_path，页码区间向后继承
        flat.sort(key=lambda t: t[1])
        paths: Dict[int, List[str]] = {}
        stack: List[Tuple[int, str]] = []
        for level, pg, title in flat:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            paths[pg] = [t for _, t in stack]
        result: Dict[int, List[str]] = {}
        last: List[str] = []
        for pg in range(len(reader.pages)):
            if pg in paths:
                last = paths[pg]
            if last:
                result[pg] = list(last)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.debug("PDF 大纲解析跳过（%s）: %s", path.name, exc)
        return {}


# ----------------------------------------------------------------------
#  主入口
# ----------------------------------------------------------------------
def enrich_file(
    path: str | Path,
    default_acl: str = "*",
    default_status: str = "active",
    ocr_cfg: Optional[Dict] = None,
) -> List[Document]:
    """解析单个文件并产出带完整元数据的小节 Document 列表。

    与 ``document_loader.load_document`` 的区别：
    - MD/DOCX 按标题切分为多个小节（每节自带 heading_path）
    - 统一补齐 title / breadcrumb / updated_at / acl / doc_status / department
    - MD frontmatter 中的 status / permissions / valid_until 参与判定

    Args:
        ocr_cfg: 扫描件 OCR 配置（对应 ``config.document_loader.ocr``）。
            **必须在此透传**：PDF 入库走的是本函数而不是直接调 ``load_document``，
            漏传会让扫描件静默提不出文本 —— 用户只看到"入库成功但搜不到"。
    """
    p = Path(path)
    if not p.exists():
        logger.warning("文件不存在: %s", p)
        return []
    ext = p.suffix.lower()
    fallback_title = p.stem
    now = datetime.now()

    # 1. 读取原始文本 + frontmatter（仅 MD）
    front: Dict[str, Any] = {}
    if ext in (".md", ".markdown"):
        try:
            # 编码自动探测（utf-8-sig → gb18030 → big5），避免 GBK 文件被硬按
            # UTF-8 读成乱码入库
            from .document_loader import _decode_text_bytes

            raw, _used_enc = _decode_text_bytes(p.read_bytes())
        except OSError as exc:
            logger.error("读取 MD 失败 %s: %s", p, exc)
            return []
        front, body = parse_frontmatter(raw)
        sections = _split_markdown_sections(body, fallback_title)
        core_meta: Dict[str, Any] = {}
    elif ext == ".docx":
        try:
            sections, core_meta = _split_docx_sections(p, fallback_title)
        except Exception as exc:
            logger.error("解析 DOCX 失败 %s: %s", p, exc)
            return []
        body = ""
    else:
        # PDF / TXT：复用既有解析器（PDF 已按页切分）。PDF 若带书签大纲，
        # 按大纲恢复每页的章节层级（与 MD/DOCX 的 heading_path 对齐）；
        # 无书签则退化为按页平铺、无章节路径。
        from .document_loader import load_document

        docs = load_document(p, clean=True, ocr_cfg=ocr_cfg)
        outline = _pdf_outline_paths(p) if ext == ".pdf" else {}
        sections: List[Tuple[List[str], str, str]] = []
        for d in docs:
            page0 = (d.page or 1) - 1
            hpath = list(outline.get(page0, []))
            sec_title = hpath[-1] if hpath else (d.metadata or {}).get("title") or fallback_title
            sections.append((hpath, sec_title, d.content))
        core_meta = {}
        body = ""

    # 2. 归一元数据（frontmatter > core properties > 文件名）
    title = (
        str(front.get("title") or core_meta.get("core_title") or "").strip()
        or (sections[0][1] if sections and sections[0][0] else fallback_title)
    )
    acl = _normalize_acl(front.get("permissions") or front.get("access") or front.get("acl"))
    if acl == "*" and default_acl and default_acl != "*":
        acl = _normalize_acl(default_acl)
    raw_status = front.get("status")
    if raw_status is not None and str(raw_status).strip().lower() in _STATUS_VALUES:
        status = _normalize_status(raw_status)
    else:
        status = _normalize_status(default_status)
    department = str(front.get("department") or core_meta.get("department") or "").strip() or None
    valid_until = _parse_valid_until(front.get("valid_until") or front.get("expires_at"))
    updated_at = core_meta.get("updated_at") or _file_updated_at(p) or now

    documents: List[Document] = []
    for sec_no, (heading_path, _sec_title, text) in enumerate(sections, start=1):
        if not text.strip():
            continue
        breadcrumb = " > ".join(
            heading_path
            if heading_path and heading_path[0] == title
            else ([title] + heading_path if heading_path else [title])
        )
        # 编号列表小节（参考文献等）在 breadcrumb 上追加条目数注记，
        # 随 breadcrumb 前缀进入每个分块，聚合统计类问题由此获得可靠线索
        annot, entry_count = _numbered_entry_annot(text)
        if annot:
            breadcrumb = f"{breadcrumb}｜{annot}"
        documents.append(
            Document(
                # page 语义：PDF = 物理页码；MD/DOCX = 小节序号。
                # 分块 id = md5(source|page|chunk_index)，按节编号才能保证
                # 同一文件不同小节的 id 不冲突（否则 upsert 会互相覆盖）。
                page=sec_no,
                content=text,
                metadata={
                    "source": p.name,
                    "filepath": str(p),
                    "ext": ext,
                    "title": title,
                    "heading_path": heading_path,
                    # MD/DOCX 的首个标题通常就是文档标题，此时直接用层级路径，
                    # 避免 "员工手册 > 员工手册 > 考勤制度" 式的重复拼接
                    "breadcrumb": breadcrumb,
                    # 分块前缀用短版：只保留末级章节名（+计数注记）。完整层级
                    # 路径（含文档全名）在低参数量模型上会稀释章节语义信号、
                    # 挤占上下文 token，还容易被误当作正文复读。
                    "chunk_prefix": " > ".join(heading_path[-1:]) + (f"｜{annot}" if annot else ""),
                    # 编号条目小节的条目数（如参考文献共 30 条），供生成侧
                    # 在上下文标题中直接展示，聚合统计类问题据此可靠作答
                    "entry_count": entry_count or "",
                    "acl": acl,
                    "doc_status": status,
                    "department": department or "",
                    "updated_at": updated_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "valid_until": valid_until.strftime("%Y-%m-%d") if valid_until else "",
                },
            )
        )
    logger.info("元数据增强 %s：%d 个小节（title=%s, acl=%s, status=%s）", p.name, len(documents), title, acl, status)
    return documents


def enrich_directory(
    root: str | Path,
    default_acl: str = "*",
    recursive: bool = True,
) -> List[Document]:
    """批量增强目录下所有受支持文档。"""
    from .utils import iter_doc_files

    root_p = Path(root)
    all_docs: List[Document] = []
    for f in iter_doc_files(root_p, recursive=recursive):
        all_docs.extend(enrich_file(f, default_acl=default_acl))
    logger.info("目录 %s 元数据增强完成，共 %d 个小节", root_p, len(all_docs))
    return all_docs
