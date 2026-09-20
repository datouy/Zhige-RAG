"""数据质量管线：深度清洗、去重、数据准入门（DataGate）。

对应四层架构中的**数据层**——"来源明确、过滤过期/草稿/权限不清内容"：

1. :func:`deep_clean_text`   去格式噪音：页眉页脚、页码水印、装饰线、
   全角空白，得到干净正文（比 ``utils.clean_text`` 更激进，仅用于入库）。
2. :func:`dedup_documents` / :func:`dedup_chunks`
   文档级（内容哈希完全相同）与分块级（精确重复 + 可选近重复）去重。
3. :func:`gate_documents`    准入门：按策略过滤 **draft / expired /
   权限不清** 的文档，产出可审计的拒绝报告。

入库路径（API ``/ingest`` 与 ``scripts/ingest.py``）统一走
:func:`run_data_pipeline`，保证两条入口行为一致。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .document_loader import Document
from .utils import get_logger

logger = get_logger("data_quality")

# ----------------------------------------------------------------------
#  1. 深度清洗
# ----------------------------------------------------------------------
# 纯页码行：第 3 页 / 3 / - 3 - / Page 3 / 3/12 / 第 3 页 共 10 页
_PAGE_NO_RE = re.compile(
    r"^\s*(?:[-–—•·]?\s*\d{1,4}\s*[-–—•·]?|第\s*\d+\s*页(\s*[,，/共].*)?|Page\s+\d+(\s*of\s*\d+)?)\s*$",
    re.IGNORECASE,
)
# 装饰分隔线：--- === ___ *** ……
_DECOR_RE = re.compile(r"^[\s\-—=_*~·•#.]{6,}$")
# 全角空白/不间断空格 → 普通空格
_NBSP_RE = re.compile(r"[\u00a0\u3000\u2007\u202f]")


def _strip_repeated_lines(pages: List[str], min_repeat: int = 3) -> Tuple[List[str], int]:
    """去除在多页中重复出现的页眉/页脚行。

    同一行（去空白后）在 ≥ ``min_repeat`` 个"页"中出现 → 视为页眉页脚剔除。
    单页文档不做该处理（避免误删正文重复句）。
    """
    if len(pages) < min_repeat:
        return pages, 0
    counter: Dict[str, int] = {}
    for pg in pages:
        seen_in_page = set()
        for line in pg.splitlines():
            key = line.strip()
            if not key or _PAGE_NO_RE.match(key):
                continue
            if key not in seen_in_page:
                seen_in_page.add(key)
                counter[key] = counter.get(key, 0) + 1
    # 行长度过短（<4 字符）不剔除，噪声概率低
    repeated = {k for k, c in counter.items() if c >= min_repeat and len(k) >= 4}
    if not repeated:
        return pages, 0
    cleaned: List[str] = []
    removed = 0
    for pg in pages:
        kept_lines = []
        for line in pg.splitlines():
            if line.strip() in repeated:
                removed += 1
                continue
            kept_lines.append(line)
        cleaned.append("\n".join(kept_lines))
    return cleaned, removed


def deep_clean_text(text: str, treat_blank_line_as_page: bool = True) -> str:
    """入库专用深度清洗（原 ``clean_text`` 只合并空白，不去格式噪音）。

    步骤：
    1. 归一不间断空格/全角空格
    2. 按"页"（双换行分块近似）剔除重复页眉页脚与纯页码行、装饰线
    3. 回落到基础清洗（合并空白与多余空行）
    """
    if not text:
        return ""
    text = _NBSP_RE.sub(" ", text)
    if treat_blank_line_as_page:
        pages = re.split(r"\n{2,}", text)
        pages, _removed = _strip_repeated_lines(pages)
        text = "\n\n".join(pages)
    # 剔除纯页码行与装饰线（页内单次出现也剔）
    kept: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if _PAGE_NO_RE.match(s) or _DECOR_RE.match(s):
            continue
        kept.append(line)
    from .utils import clean_text

    return clean_text("\n".join(kept))


# ----------------------------------------------------------------------
#  2. 去重
# ----------------------------------------------------------------------
def content_hash(text: str) -> str:
    """内容指纹：SHA256（先做空白归一，避免仅空格差异导致漏判）。"""
    norm = re.sub(r"\s+", "", text or "")
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 8) -> set:
    """字符级 k-shingle 集合（近重复判定用）。"""
    t = re.sub(r"\s+", "", text or "")
    if len(t) <= k:
        return {t} if t else set()
    return {t[i : i + k] for i in range(len(t) - k + 1)}


def dedup_documents(docs: Sequence[Document]) -> Tuple[List[Document], int]:
    """文档级精确去重：全文内容哈希相同 → 保留第一份。"""
    kept: List[Document] = []
    seen: set = set()
    dups = 0
    for d in docs:
        h = content_hash(d.content)
        if h in seen:
            dups += 1
            logger.info("文档级去重：丢弃 %s（与已入库文档内容完全相同）", (d.metadata or {}).get("source"))
            continue
        seen.add(h)
        d.metadata = {**(d.metadata or {}), "content_hash": h}
        kept.append(d)
    return kept, dups


def dedup_chunks(
    chunks: Sequence[Any],
    near_dup_threshold: float = 0.0,
    max_near_dup_candidates: int = 3000,
) -> Tuple[List[Any], int]:
    """分块级去重。

    - 精确去重：分块文本哈希相同 → 丢弃后者
    - 近重复（可选，``0 < threshold < 1``）：8-shingle Jaccard ≥ 阈值 → 丢弃后者。
      候选量大时（> ``max_near_dup_candidates``）自动跳过近重复，退化为仅精确去重。

    Returns:
        (去重后的 chunks, 丢弃数)
    """
    kept: List[Any] = []
    seen_exact: set = set()
    # 保留每块 shingle 集，O(n²) 只在候选量可控时启用
    kept_shingles: List[set] = []
    dropped = 0
    use_near = 0.0 < near_dup_threshold < 1.0 and len(chunks) <= max_near_dup_candidates
    for c in chunks:
        text = c.text
        h = content_hash(text)
        if h in seen_exact:
            dropped += 1
            continue
        if use_near:
            sh = _shingles(text)
            is_dup = False
            for prev in kept_shingles:
                inter = len(sh & prev)
                if not inter:
                    continue
                union = len(sh | prev)
                if union and inter / union >= near_dup_threshold:
                    is_dup = True
                    break
            if is_dup:
                dropped += 1
                continue
            kept_shingles.append(sh)
        seen_exact.add(h)
        kept.append(c)
    return kept, dropped


# ----------------------------------------------------------------------
#  3. 数据准入门（DataGate）
# ----------------------------------------------------------------------
@dataclass
class GateReport:
    """准入门决策报告（可审计：哪些文档因什么原因被拒）。"""

    kept: int = 0
    rejected: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)
    details: List[str] = field(default_factory=list)

    def reject(self, source: str, breadcrumb: str, reason: str) -> None:
        self.rejected += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        self.details.append(f"{source}::{breadcrumb} — {reason}")

    def summary(self) -> Dict[str, Any]:
        return {
            "kept": self.kept,
            "rejected": self.rejected,
            "reasons": dict(self.reasons),
            "details": self.details[:50],
        }


def gate_documents(docs: Sequence[Document], policy: Optional[Dict[str, Any]] = None) -> Tuple[List[Document], GateReport]:
    """数据准入门：过滤过期、草稿、权限不清的内容。

    策略字段（config ``data_quality.gate``）：
    - ``allowed_status``      : 允许入库的状态列表，默认 ``["active"]``
    - ``require_explicit_acl``: true 时拒绝 acl == "*"（权限不清）的文档
    - ``expiry_days``         : >0 时，源文件更新时间早于 now - N 天 → 过期拒绝
                                （0 = 不按时间过期，但 frontmatter valid_until 仍生效）
    """
    policy = policy or {}
    allowed_status = set(policy.get("allowed_status", ["active"]))
    require_acl = bool(policy.get("require_explicit_acl", False))
    expiry_days = int(policy.get("expiry_days", 0) or 0)
    now = datetime.now()

    report = GateReport()
    kept: List[Document] = []
    for d in docs:
        md = d.metadata or {}
        source = str(md.get("source") or "unknown")
        breadcrumb = str(md.get("breadcrumb") or md.get("title") or "")

        status = str(md.get("doc_status") or "active")
        if status not in allowed_status:
            report.reject(source, breadcrumb, f"status={status}（仅允许 {sorted(allowed_status)}）")
            continue

        acl = str(md.get("acl") or "*")
        if require_acl and acl == "*":
            report.reject(source, breadcrumb, "权限不清（acl=* 且策略要求显式权限）")
            continue
        # Chroma 0.4.x metadata 只能存标量，多分组 acl 无法在查询侧精确过滤
        # （$in 只做全值匹配）。宁可拒绝入库，也不留下"越权可见/全员不可见"
        # 的灰色文档——需要多组可见时请按组拆分文档或显式使用 *。
        if "," in acl and policy.get("reject_multi_group_acl", True):
            report.reject(source, breadcrumb, f"多分组 acl 暂不支持精确过滤（{acl}），请按组拆分")
            continue

        # frontmatter 显式 valid_until
        valid_until = str(md.get("valid_until") or "").strip()
        if valid_until:
            try:
                vu = datetime.strptime(valid_until, "%Y-%m-%d")
                if vu < now:
                    report.reject(source, breadcrumb, f"已过有效期 valid_until={valid_until}")
                    continue
            except ValueError:
                logger.warning("valid_until 格式无法解析，忽略：%s（%s）", valid_until, source)

        # 按源文件更新时间滚动过期
        if expiry_days > 0:
            updated_raw = str(md.get("updated_at") or "").strip()
            try:
                updated = datetime.strptime(updated_raw, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                updated = None
            if updated is not None and updated < now - timedelta(days=expiry_days):
                report.reject(
                    source,
                    breadcrumb,
                    f"内容过期：更新于 {updated_raw}，超过 {expiry_days} 天",
                )
                continue

        report.kept += 1
        kept.append(d)

    if report.rejected:
        logger.info("数据准入门：保留 %d，拒绝 %d，原因分布 %s", report.kept, report.rejected, report.reasons)
    return kept, report


# ----------------------------------------------------------------------
#  4. 统一管线
# ----------------------------------------------------------------------
def run_data_pipeline(
    docs: Sequence[Document],
    cfg: Dict[str, Any],
    splitter: Any,
) -> Tuple[List[Any], Dict[str, Any]]:
    """数据层统一管线：深度清洗 → 文档去重 → 准入门 → 分块 → 分块去重。

    Args:
        docs: ``document_meta.enrich_file/enrich_directory`` 产出的小节文档。
        cfg:  全局配置（读取 ``data_quality`` 段）。
        splitter: 已构造的文本分块器（``build_splitter`` 产物）。

    Returns:
        (chunks, report_dict) — report_dict 含 gate / dedup 统计，供入库方
        写入 DocumentRecord 并在 API 响应中返回，保证数据层可审计。
    """
    dq_cfg = cfg.get("data_quality", {}) or {}
    report: Dict[str, Any] = {}

    # 1. 深度清洗（开关默认开）
    if dq_cfg.get("deep_clean", True):
        docs = [Document(page=d.page, content=deep_clean_text(d.content), metadata=d.metadata) for d in docs]

    # 2. 文档级去重（content_hash 写回各文档 metadata）
    docs, doc_dups = dedup_documents(docs)
    report["doc_dedup_dropped"] = doc_dups
    report["content_hashes"] = [(d.metadata or {}).get("content_hash") for d in docs]

    # 3. 准入门
    docs, gate = gate_documents(docs, dq_cfg.get("gate"))
    report["gate"] = gate.summary()

    # 4. 分块（先在"裸文本"上去重，再注入 breadcrumb 前缀——前缀含章节路径，
    #    若先拼前缀，不同文档的相同正文会因前缀不同而漏判）
    add_breadcrumb = bool(dq_cfg.get("breadcrumb_in_chunk", True))
    chunks: List[Any] = []
    for d in docs:
        # page 必须随 metadata 下传：分块 id = md5(source|page|chunk_index)，
        # 而 chunk_index 在每个小节内从 0 重新计数——小节序号缺失时所有小节
        # 的同序号块 id 碰撞，upsert 会静默覆盖（137 块只剩 19 块的事故根源）。
        md = dict(d.metadata or {})
        md.setdefault("page", d.page)
        chunks.extend(splitter.split_text(d.content, metadata=md))

    # 5. 分块级去重
    chunks, chunk_dups = dedup_chunks(chunks, near_dup_threshold=float(dq_cfg.get("near_dup_threshold", 0.0) or 0.0))

    # 6. breadcrumb 前缀注入（保留章节语境，检索命中时块自带出处层级）。
    #    优先使用短版 chunk_prefix（末级章节名），缺失时回退完整 breadcrumb。
    if add_breadcrumb:
        for ck in chunks:
            md = ck.metadata or {}
            prefix = md.get("chunk_prefix") or md.get("breadcrumb") or ""
            if prefix:
                ck.text = f"【{prefix}】\n{ck.text}"

    report["chunk_dedup_dropped"] = chunk_dups
    report["chunks"] = len(chunks)

    return chunks, report
