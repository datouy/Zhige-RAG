"""入库内容质检与安全门禁（Ingest Guard）。

设计来源
--------
参考 Liquid AI **d1 决策模型** 的原语思路，对应文章点名的两处落地场景：

    「对内容做合规检测、涉黄涉暴与毒性过滤」
    「自动化质检里给……做等级判定」

在本项目里，内容进库前本来只经过**规则式**数据准入门（:mod:`src.data_quality`
的 ``DataGate``：状态 / 权限 / 过期）。规则擅长抓"结构性缺陷"，抓不到"语义级"
的问题：一段排版完好、权限合规，但**通篇是乱码目录残片**（低质，入库只会污染
检索）或**含违规内容**（该拦或至少该打标）的文本，规则无能为力。

本模块用 d1 两个原语补齐：

- ``score`` —— **质量分级**：该分块作为"可检索的有效知识"值几分。
  乱码 / 页眉页脚 / 目录残片 / 表格断裂 → 低分；信息完整的正文 → 高分。
- ``noul``  —— **合规判定**：该分块是否含明显违规内容（涉黄涉暴 / 违法 /
  极端仇恨）。命中即打标，``action=drop`` 时剔除。

效率：两个原语在**同一次** ``decide`` 调用里一次问完（对齐 d1 的
``system_one(text, questions)``，多个问题共享一次前向），不是两次调用。

设计约束（与 :mod:`src.decision_provider` 既有约定一致）：

1. **默认关闭**：``data_quality.ingest_guard.enabled=false`` 时**逐字节零变化**；
2. **非破坏默认**：默认 ``action="flag"`` —— 只给低质 / 违规块打 metadata 标记，
   不删用户内容；只有显式配 ``action="drop"`` 才拦截；
3. **绝不抛出**：任何异常都吞掉，返回"全部保留"的结果（门禁不能拖垮入库）；
4. **成本护栏**：``max_chunks`` 限制单文件最多检查多少块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .utils import get_logger

logger = get_logger("ingest_guard")

# ----------------------------------------------------------------------
#  判定指令与阶梯
# ----------------------------------------------------------------------
# 质量分级阶梯（5 级，落在 d1 的 2~10 级约束内，由低到高）
QUALITY_LADDER: List[str] = [
    "很差：乱码、无意义字符或与主题完全无关的噪声",
    "较差：多是页眉页脚、目录残片、表格断裂等碎片信息",
    "一般：有部分可用信息，但残缺或语焉不详",
    "较好：信息基本完整、表述连贯的知识内容",
    "很好：信息完整、结构清晰、可直接用于回答问题的正文",
]

QUALITY_INSTRUCTION = (
    "评估这段「分块文本」作为知识库可检索内容的整体质量。"
    "重点看它是否是一段信息完整、表述连贯、可被用来回答问题的正文；"
    "乱码、页眉页脚、目录残片、断裂的表格应判为低质量。"
)

COMPLIANCE_INSTRUCTION = (
    "判断这段「分块文本」是否包含明显违规内容，包括：色情低俗、暴力血腥、"
    "违法犯罪教程、极端仇恨与煽动性言论等。仅当确有明显违规内容时才判定为“是”，"
    "正常的业务／技术文本一律判定为“否”。"
)

DEFAULT_MIN_QUALITY = 0.3
DEFAULT_COMPLIANCE_THRESHOLD = 0.7


@dataclass
class IngestGuardResult:
    """一次入库门禁的产出。"""

    kept: List[Any] = field(default_factory=list)
    dropped: List[Any] = field(default_factory=list)
    flagged: List[Dict[str, Any]] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)


def _chunk_text(chunk: Any) -> str:
    return str(getattr(chunk, "text", "") or "")


def _chunk_meta(chunk: Any) -> Dict[str, Any]:
    md = getattr(chunk, "metadata", None)
    return md if isinstance(md, dict) else {}


def guard_chunks(
    chunks: Sequence[Any],
    provider: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> IngestGuardResult:
    """对分块做 d1 质量分级 + 合规判定。

    Args:
        chunks: 待检查分块（需具备 ``text`` / ``metadata`` 属性）。
        provider: :class:`src.decision_provider.BaseDecisionProvider` 实例；
            ``None`` 时直接返回"全部保留"。
        cfg: ``data_quality.ingest_guard`` 子配置。

    Returns:
        :class:`IngestGuardResult`。``cfg`` 未启用 / provider 为空 / 检查失败时，
        ``kept`` 即入参全集、``dropped`` 为空，行为与不做门禁完全一致。
    """
    cfg = cfg or {}
    chunks = list(chunks)
    result = IngestGuardResult(kept=chunks, dropped=[], flagged=[], report={"enabled": False})
    if provider is None or not cfg.get("enabled") or not chunks:
        return result

    max_chunks = max(1, int(cfg.get("max_chunks", 200)))
    min_quality = float(cfg.get("min_quality_score", DEFAULT_MIN_QUALITY))
    comp_threshold = float(cfg.get("compliance_min_probability", DEFAULT_COMPLIANCE_THRESHOLD))
    action = str(cfg.get("action", "flag")).strip().lower()
    drop_mode = action == "drop"

    kept: List[Any] = []
    dropped: List[Any] = []
    flagged: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        # 超出预算的块不检查，直接保留（成本护栏）
        if i >= max_chunks:
            kept.append(chunk)
            continue
        text = _chunk_text(chunk).strip()
        if not text:
            kept.append(chunk)
            continue

        quality_norm: Optional[float] = None
        compliance_prob: Optional[float] = None
        try:
            # 一次调用问两个问题（对齐 d1 system_one 的"共享一次前向"）
            decisions = provider.decide(
                text[:2000],
                {
                    "quality": {"type": "score", "instructions": QUALITY_INSTRUCTION, "criteria": QUALITY_LADDER},
                    "compliance": {"type": "noul", "instructions": COMPLIANCE_INSTRUCTION},
                },
            )
        except Exception as exc:  # noqa: BLE001 - 门禁失败一律放行，不能拖垮入库
            logger.warning("入库门禁决策失败（该块直接放行）: %s", exc)
            kept.append(chunk)
            continue

        if decisions:
            q = decisions.get("quality")
            if q is not None:
                quality_norm = float(getattr(q, "normalized", 0.0))
            c = decisions.get("compliance")
            if c is not None:
                compliance_prob = float(getattr(c, "probability", 0.0))

        low_quality = quality_norm is not None and quality_norm < min_quality
        suspicious = compliance_prob is not None and compliance_prob >= comp_threshold

        if not low_quality and not suspicious:
            kept.append(chunk)
            continue

        md = _chunk_meta(chunk)
        md["_guard_low_quality"] = bool(low_quality)
        md["_guard_suspicious"] = bool(suspicious)
        if quality_norm is not None:
            md["_guard_quality"] = round(quality_norm, 4)
        if compliance_prob is not None:
            md["_guard_compliance_prob"] = round(compliance_prob, 4)

        flagged.append(
            {
                "index": i,
                "source": md.get("source") or md.get("filepath") or "",
                "low_quality": bool(low_quality),
                "suspicious": bool(suspicious),
                "quality": None if quality_norm is None else round(quality_norm, 4),
                "compliance_probability": None if compliance_prob is None else round(compliance_prob, 4),
            }
        )
        if drop_mode:
            dropped.append(chunk)
        else:
            kept.append(chunk)

    result.kept = kept
    result.dropped = dropped
    result.flagged = flagged
    result.report = {
        "enabled": True,
        "action": action,
        "checked": min(len(chunks), max_chunks),
        "flagged": len(flagged),
        "dropped": len(dropped),
        "low_quality": sum(1 for f in flagged if f["low_quality"]),
        "suspicious": sum(1 for f in flagged if f["suspicious"]),
    }
    if flagged:
        logger.info(
            "入库门禁：检查 %d 块，标记 %d（低质 %d / 可疑 %d），拦截 %d",
            result.report["checked"],
            len(flagged),
            result.report["low_quality"],
            result.report["suspicious"],
            len(dropped),
        )
    return result


__all__ = [
    "QUALITY_LADDER",
    "QUALITY_INSTRUCTION",
    "COMPLIANCE_INSTRUCTION",
    "IngestGuardResult",
    "guard_chunks",
]
