"""反馈自动分流（Feedback Auto-Routing）。

设计来源
--------
参考 Liquid AI **d1 决策模型** 的原语思路，对应文章点名的最佳落地场景之一：

    「客服工单进来瞬间自动分流到具体业务组」

在本项目里，"工单"就是一条用户差评（``FeedbackRecord``）。此前 ``question_type``
完全依赖前端手填——多数时候是空的，导致 ``/feedback/stats`` 的"差评按类型归因"
基本失效，bad case 无法自动流到该管的人手里。

本模块用 d1 的 ``choice`` 原语（多选一分类）把一条反馈归因到最该处理的环节：

==============  ==========================================  ==================
归因标签         含义                                         该看的人/环节
==============  ==========================================  ==================
``retrieval``   知识库里有，但这次没召回（召回缺失）              切块 / 检索 / 向量库
``answer``      召回了但答案错，或与文档相抵触（事实错误）          生成 / 提示词
``grounding``   答案里的说法在知识库里找不到依据（编造）            校验 / 提示词
``incomplete``  方向没错但没答全、漏了关键点（不完整）             检索 + 生成
``format``      内容大体对，只是啰嗦 / 排版乱 / 语气不合适（格式）   交互 / 提示词
``other``       以上都不是，或无法判断                            人工
==============  ==========================================  ==================

设计约束（与 :mod:`src.decision_provider` 的既有约定一致）：

1. **provider 为空 → 返回 ``None``**，调用方保持"未分类"原行为（零破坏）；
2. **绝不抛出**：任何异常都被吞掉并返回 ``None``（分流是锦上添花，不能拖垮反馈提交）；
3. **置信度门槛**：低于 ``min_confidence`` 时宁可不分类，也不给一条误导性的标签；
4. **可测**：只依赖 provider 的 ``choice()`` 接口，测试用 fake provider 即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .utils import get_logger

logger = get_logger("feedback_router")

# ----------------------------------------------------------------------
#  归因标签
# ----------------------------------------------------------------------
CATEGORY_RETRIEVAL = "retrieval"    # 召回缺失
CATEGORY_ANSWER = "answer"          # 事实错误
CATEGORY_GROUNDING = "grounding"    # 无依据 / 编造
CATEGORY_INCOMPLETE = "incomplete"  # 不完整
CATEGORY_FORMAT = "format"          # 表述 / 格式
CATEGORY_OTHER = "other"            # 其他

# key 即写入 ``FeedbackRecord.question_type`` 的值；value 是给 d1 choice 的判定标准
FEEDBACK_CATEGORIES: Dict[str, str] = {
    CATEGORY_RETRIEVAL: "知识库里本应有相关内容，但这次没有检索到（召回缺失）",
    CATEGORY_ANSWER: "检索到了相关内容，但答案是错的、或与文档内容相矛盾（事实错误）",
    CATEGORY_GROUNDING: "答案里的说法在知识库中找不到任何依据，像是编造的（无依据）",
    CATEGORY_INCOMPLETE: "答案方向没错，但没有答全、漏掉了关键部分（不完整）",
    CATEGORY_FORMAT: "内容大体正确，只是表述啰嗦、排版混乱或语气不合适（格式问题）",
    CATEGORY_OTHER: "以上都不是，或根据反馈内容无法判断",
}

# 便于前端/报表展示的中文短标签（无需再查表）
CATEGORY_LABELS: Dict[str, str] = {
    CATEGORY_RETRIEVAL: "召回缺失",
    CATEGORY_ANSWER: "事实错误",
    CATEGORY_GROUNDING: "无依据",
    CATEGORY_INCOMPLETE: "回答不完整",
    CATEGORY_FORMAT: "表述格式",
    CATEGORY_OTHER: "其他",
}

_FEEDBACK_INSTRUCTION = (
    "阅读用户对一次知识库问答的反馈（含原问题、当时的回答、以及用户填写的纠正），"
    "判断这次失败最主要属于哪一类问题，以便分流到对应的处理环节。"
    "只依据反馈内容本身判断，不要臆测。"
)


@dataclass
class FeedbackRoute:
    """一条反馈的分流结果。"""

    category: str = CATEGORY_OTHER
    label: str = CATEGORY_LABELS[CATEGORY_OTHER]
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


def build_feedback_text(query: str, answer: str = "", correction: str = "") -> str:
    """把一条反馈拼成便于决策的纯文本（空字段自动省略）。"""
    parts = []
    if query and query.strip():
        parts.append(f"原问题：{query.strip()}")
    if answer and answer.strip():
        parts.append(f"当时的回答：{answer.strip()[:1500]}")
    if correction and correction.strip():
        parts.append(f"用户认为应该是：{correction.strip()[:1500]}")
    return "\n".join(parts)


def classify_feedback(
    query: str,
    answer: str = "",
    correction: str = "",
    provider: Optional[Any] = None,
    *,
    min_confidence: float = 0.5,
    label_only: bool = True,
) -> Optional[FeedbackRoute]:
    """用 d1 ``choice`` 原语把一条反馈分流到处理组。

    与规则式归因的取舍：这里的判断需要"读懂"用户自然语言的抱怨（"我说的是年假，
    它答的是调休""答案里那个数字文档里根本没有"），正则很难覆盖，恰好是决策模型
    擅长而生成模型过重的场景。

    Args:
        query: 被评价的问题。
        answer: 当时的回答（可空）。
        correction: 用户填写的纠正（可空，差评时最有信息量）。
        provider: :class:`src.decision_provider.BaseDecisionProvider` 实例；
            ``None`` 时直接返回 ``None``（调用方保持未分类）。
        min_confidence: 最低置信度，低于此值返回 ``None``（宁缺毋滥）。
        label_only: 预留参数（当前恒为短标签）。

    Returns:
        :class:`FeedbackRoute`；未启用 / 无文本 / 决策失败 / 置信度不足时返回 ``None``。
    """
    if provider is None:
        return None
    text = build_feedback_text(query, answer, correction)
    if not text:
        return None

    try:
        res = provider.choice(
            text, _FEEDBACK_INSTRUCTION, FEEDBACK_CATEGORIES, key="category"
        )
    except Exception as exc:  # noqa: BLE001 - provider 已兜底，这里再保险一层
        logger.warning("反馈分流决策失败（已跳过）: %s", exc)
        return None

    if res is None or not res.label:
        return None
    category = str(res.label).strip()
    if category not in FEEDBACK_CATEGORIES:
        return None
    if res.confidence < min_confidence:
        logger.info(
            "反馈分流置信度 %.2f < %.2f，保持未分类（原问题：%s）",
            res.confidence,
            min_confidence,
            (query or "")[:30],
        )
        return None

    route = FeedbackRoute(
        category=category,
        label=CATEGORY_LABELS.get(category, category),
        confidence=float(res.confidence),
        probabilities=dict(res.probabilities or {}),
    )
    logger.info(
        "反馈已分流：%s（%s，置信度 %.2f）", query[:30], route.label, route.confidence
    )
    return route


__all__ = [
    "CATEGORY_RETRIEVAL",
    "CATEGORY_ANSWER",
    "CATEGORY_GROUNDING",
    "CATEGORY_INCOMPLETE",
    "CATEGORY_FORMAT",
    "CATEGORY_OTHER",
    "FEEDBACK_CATEGORIES",
    "CATEGORY_LABELS",
    "FeedbackRoute",
    "build_feedback_text",
    "classify_feedback",
]
