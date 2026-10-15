"""查询意图路由：轻量规则分类，零模型开销。

架构位置：检索层最前置的一道"分流器"——在花费 embedding / GPU 资源之前，
先判断问题属于哪类意图，把明显不需要知识库的问题拦在外面：

- **chitchat**  寒暄/身份/致谢/道别 → 直接模板回应，跳过检索与 LLM
  （省一次 GPU 推理，也避免小模型对着空上下文瞎编自我介绍）；
- **aggregate** 聚合统计类（"引用了多少文献"）→ 走检索 + 结构化元数据
  直答，不进 LLM（由 ``RAGPipeline._structured_count_answer`` 实现）；
- **fact**      默认事实问答 → 正常"混合检索 → 重排 → 生成"链路。

设计约束：**规则必须保守**。误把真问题当寒暄吞掉比多跑一次检索严重得多，
所以寒暄判定要求"短查询 + 全句匹配"（如"你好，请问年假怎么申请"会正常
走检索），聚合判定沿用入库侧验证过的正则。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from .utils import get_logger

logger = get_logger("query_router")

# ----------------------------------------------------------------------
#  意图判定
# ----------------------------------------------------------------------
# 寒暄类：全句匹配（允许少量标点/语气词尾巴），杜绝"你好，请问……"
# 这类带真实问题的句子被误拦。
_GREETING_RE = re.compile(r"^(?:你好|您好|嗨|哈喽|hi|hello|hey)[\s!！.。~～，,]*$", re.IGNORECASE)
_FAREWELL_RE = re.compile(r"^(?:再见|拜拜|晚安|明天见|回头见|bye+)[\s!！.。~～，,]*$", re.IGNORECASE)
_THANKS_RE = re.compile(r"^(?:谢谢|多谢|感谢|辛苦了|thanks|thank you)[\s!！.。~～，,]*$", re.IGNORECASE)
# 身份/能力类：关键词匹配即可（这类问题本身偏长）
_IDENTITY_RE = re.compile(
    r"你是谁|你叫什么|你的名字|介绍[一下]*你自己|自我介绍|你能(?:做|干)什么|你会什么|你有什么(?:功能|能力)|怎么使用你"
)

# 聚合统计类问题意图检测（供 _structured_count_answer 触发）：
# 分支一：数量词在前、条目名词在后——"引用了多少文献"、"参考了多少篇文献"；
# 分支二：名词在前——"参考文献有多少条"。
# 名词与数量词间隔限制在 2 字内，避免"文献综述写了多少字"这类误触发。
_COUNT_INTENT_RE = re.compile(
    r"(?:多少|几)[一二三四五六七八九十\d]*[条篇个项种份]?[^，。？?!]{0,2}(?:参考文献|参考书目|参考资料|文献|条目)"
    r"|(?:参考文献|参考书目|参考资料|文献|条目)[^，。？?!]{0,2}(?:有)?(?:多少|几)"
)


def is_aggregate_query(question: str) -> bool:
    """是否为聚合统计类问题（条目计数等）。"""
    return bool(_COUNT_INTENT_RE.search(question or ""))


@dataclass
class RouteDecision:
    """路由决策。

    Attributes:
        intent: chitchat / aggregate / fact。
        reply: chitchat 意图的模板回应（含 {assistant_name}/{org_name} 占位，
               由调用方用运行时变量填充）；其他意图为 None。
    """

    intent: str
    reply: Optional[str] = None


_CHITCHAT_TEMPLATES = {
    "greeting": "{assistant_name}你好！我是{org_name}的本地知识库助手，关于知识库中的内容随时可以问我。",
    "identity": "我是{assistant_name}，{org_name}的本地部署知识库问答助手。我的回答全部来自当前知识库中已入库的文档，并标注来源；如果知识库中没有相关内容，我会如实告知。",
    "thanks": "不客气！如果还有关于知识库内容的问题，随时找我。",
    "farewell": "再见！有问题欢迎随时回来查询。",
}


def classify_query(question: str) -> RouteDecision:
    """对用户问题做意图分类。

    顺序：寒暄（全句匹配）→ 身份 → 聚合 → 默认事实问答。
    聚合类仍需要检索（要靠检索找到带 entry_count 元数据的分块），
    因此 intent=aggregate 不携带模板回应。
    """
    q = (question or "").strip()
    if not q:
        return RouteDecision(intent="fact")

    if _GREETING_RE.match(q) or _FAREWELL_RE.match(q) or _THANKS_RE.match(q):
        kind = (
            "farewell"
            if _FAREWELL_RE.match(q)
            else "thanks"
            if _THANKS_RE.match(q)
            else "greeting"
        )
        logger.info("查询路由：chitchat(%s)「%s」", kind, q[:30])
        return RouteDecision(intent="chitchat", reply=_CHITCHAT_TEMPLATES[kind])

    if _IDENTITY_RE.search(q):
        logger.info("查询路由：chitchat(identity)「%s」", q[:30])
        return RouteDecision(intent="chitchat", reply=_CHITCHAT_TEMPLATES["identity"])

    if is_aggregate_query(q):
        logger.info("查询路由：aggregate「%s」", q[:30])
        return RouteDecision(intent="aggregate")

    return RouteDecision(intent="fact")


# ======================================================================
#  d1 决策路由（choice 原语）
# ======================================================================
# 意图选项与判定标准（供 d1 choice 原语使用）。key 必须与 RouteDecision.intent
# 的取值一致，便于直接复用下游分支。
INTENT_CRITERIA = {
    "chitchat": "寒暄、打招呼、道谢、道别，或询问助手是谁/能做什么等，无需知识库内容",
    "aggregate": "询问知识库中某类条目的数量，如'引用了多少篇文献''参考文献有多少条'",
    "fact": "需要依据知识库文档内容才能回答的事实性问题",
}

_ROUTE_INSTRUCTION = "判断用户这句话属于哪一类意图，只依据字面语义。"


def classify_query_with_decision(
    question: str,
    provider: Optional[Any] = None,
    *,
    base: Optional[RouteDecision] = None,
    min_confidence: float = 0.6,
    chitchat_min_confidence: float = 0.75,
) -> RouteDecision:
    """在规则路由之上叠加 d1 ``choice`` 决策，提升意图判定鲁棒性。

    设计原则（延续本模块"规则必须保守"的约束）：

    1. **provider 为空 → 完全等价于** :func:`classify_query`（零行为变化）；
    2. **规则已判定 chitchat → 直接采信**，不再花一次决策（省延迟，且寒暄
       规则是全句匹配、几乎不会误判）；
    3. **决策要把 fact/aggregate 改判为 chitchat 时要求更高置信度**
       （``chitchat_min_confidence``），因为"把真问题当寒暄吞掉"的代价远大于
       多跑一次检索；改判为 aggregate 则风险低，用 ``min_confidence`` 即可；
    4. 决策失败 / 置信度不足 → 回落规则结果。

    Args:
        question: 用户问题。
        provider: 决策 provider（``None`` 时退化为纯规则）。
        base: 预先算好的规则结果，避免重复计算。
        min_confidence: 一般改判所需的最低置信度。
        chitchat_min_confidence: 改判为 chitchat 所需的最低置信度（更严）。
    """
    base = base if base is not None else classify_query(question)
    if provider is None:
        return base
    q = (question or "").strip()
    if not q:
        return base
    # 规则已高置信判定寒暄 → 直接采信
    if base.intent == "chitchat":
        return base

    try:
        res = provider.choice(q, _ROUTE_INSTRUCTION, INTENT_CRITERIA, key="intent")
    except Exception as exc:  # noqa: BLE001 - provider 自身已兜底，这里再保险一层
        logger.warning("决策路由失败，回落规则：%s", exc)
        return base
    if res is None or not res.label:
        return base

    intent = str(res.label).strip()
    if intent not in INTENT_CRITERIA:
        return base
    if intent == base.intent:
        return base

    threshold = chitchat_min_confidence if intent == "chitchat" else min_confidence
    if res.confidence < threshold:
        logger.info(
            "决策路由改判 %s→%s 但置信度 %.2f < %.2f，保守回落规则",
            base.intent,
            intent,
            res.confidence,
            threshold,
        )
        return base

    logger.info("决策路由：%s → %s（置信度 %.2f）「%s」", base.intent, intent, res.confidence, q[:30])
    reply = _CHITCHAT_TEMPLATES["greeting"] if intent == "chitchat" else None
    return RouteDecision(intent=intent, reply=reply)
