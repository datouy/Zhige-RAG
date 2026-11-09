"""查询意图路由：轻量规则分类，零模型开销。

架构位置：检索层最前置的一道"分流器"——在花费 embedding / GPU 资源之前，
先判断问题属于哪类意图，把明显不需要知识库的问题拦在外面：

- **chitchat**  寒暄/身份/致谢/道别/能力边界/当前时间/通用创作 → 直接模板回应，跳过检索与 LLM
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
    r"你是谁|你叫什么|你的名字|介绍[一下]*你自己|自我介绍"
    r"|你(?:都|还)?(?:会|能|可以)(?:帮我)?(?:做|干|提供)什么"
    r"|你会什么|你(?:有|支持)哪些?(?:功能|能力)|怎么使用你|如何用你"
)

# 助手回答边界类：用户在问“没有知识库依据时怎么办”，不是在问知识库文档
# 里的某个事实。这类问题必须直接说明能力边界，不能先检索一段文档再总结。
# 规则要求同时出现“助手/系统”和“缺失/越界依据”两条线索，避免把
# “论文里如何设计无答案处理机制”误判为元问题。
_CAPABILITY_BOUNDARY_RE = re.compile(
    r"(?:"
    r"(?:你|您|助手|系统).{0,12}(?:会|能|可以|是否|是不是).{0,16}"
    r"(?:回答|处理|应对|编造|胡编|虚构|检索|查找).{0,24}"
    r"(?:不存在.{0,8}知识库|知识库.{0,10}(?:没有|不存在|未收录|之外|以外|外)|"
    r"(?:没有|未)(?:检索到|找到|命中)|无(?:相关)?(?:依据|资料|内容))"
    r"|"
    r"(?:知识库(?:中|里|内)?(?:没有|不存在|未收录)|没有(?:检索到|找到))"
    r".{0,16}(?:你|您|助手|系统).{0,10}(?:会|能|怎么|如何|是否)"
    r"|"
    r"(?:你|您|助手|系统).{0,10}(?:会不会|能否|是否可以|是否会)"
    r"(?:编造|胡编|虚构).{0,12}(?:答案|内容|事实)?"
    r")",
    re.IGNORECASE,
)

# 明确在问某份文档/设计/代码时，交给正常 RAG 链路，不能被上面的
# 能力边界规则吞掉。
_DOCUMENT_CONTEXT_RE = re.compile(
    r"(?:这篇|该|上述|这份)?(?:论文|文档|材料|报告|需求|代码|实现|设计)"
    r"(?:中|里|内|提到|说明|规定|要求|描述|如何设计)"
    r"|(?:根据|基于|按照).{0,8}(?:资料|文档|论文|材料|报告)"
    r"|(?:这篇|这份|该|上述)(?:论文|文档|材料|报告)"
)

# 当前日期/时间：应由服务端时钟确定性回答，不需要检索知识库。
# 使用整句匹配，避免把“论文中这个时间点发生了什么”误判成系统时间问题。
_DATETIME_QUERY_RE = re.compile(
    r"^(?:请问|请告诉我|你知道|能告诉我|可以告诉我)?[，,]?\s*"
    r"(?:"
    r"(?:现在|当前|此刻|这会儿)(?:是)?"
    r"(?:几点了?|几点钟?|几点几分|什么时间|什么日期|几号了?|星期几|周几|哪一天)"
    r"|(?:现在|当前|此刻|这会儿)(?:的)?(?:时间|日期)"
    r"(?:是|为|是多少|是什么)?(?:多少|什么)?"
    r"|今天(?:是)?(?:几号了?|几月几[号日]|星期几|周几|什么日子|什么日期|的?日期)"
    r"|(?:what\s+time\s+is\s+it|current\s+time|today'?s\s+date)"
    r")"
    r"(?:吗|呢|啊)?\s*[?？。!！]?$",
    re.IGNORECASE,
)

# 实时外部信息：本地知识库没有时效性数据，直接说明能力边界，避免检索
# 无关文档后再让模型拒答。若问题明确在问某份资料中的记录，则不走此路由。
_LIVE_INFO_RE = re.compile(
    r"(?:(?:今天|明天|现在|当前|最新|实时).{0,12}"
    r"(?:天气|气温|新闻|股价|股票|汇率|金价|黄金价格|油价|比分|赛况|航班|路况|空气质量|热搜))"
    r"|(?:(?:天气|股价|股票|汇率|金价|黄金价格|油价|比分|赛况|航班|路况|空气质量|热搜)"
    r".{0,12}(?:怎么样|如何|多少|吗))",
    re.IGNORECASE,
)

# 脱离知识库的创作/翻译/编程请求。知识库助手只做基于资料的任务；直接说明
# 边界，避免检索无关文档后再花一次生成模型去拒绝。
_GENERAL_GENERATION_RE = re.compile(
    r"(?:帮我|请|给我)?(?:写|创作|生成|编).{0,20}"
    r"(?:诗|歌词|故事|小说|文章|作文|邮件|周报|文案|台词|剧本|笑话|代码|程序|脚本)"
    r"|(?:作诗|写诗|讲(?:个|一个)?笑话|翻译成|润色一下|改写一下|重写一下)",
    re.IGNORECASE,
)


def is_capability_boundary_question(question: str) -> bool:
    """是否为询问助手如何处理知识库外或缺失依据的元问题。"""
    q = question or ""
    return bool(_CAPABILITY_BOUNDARY_RE.search(q)) and not _DOCUMENT_CONTEXT_RE.search(q)


def is_datetime_query(question: str) -> bool:
    """是否为询问当前日期/时间的系统问题。"""
    q = (question or "").strip()
    return bool(_DATETIME_QUERY_RE.fullmatch(q)) and not _DOCUMENT_CONTEXT_RE.search(q)


def is_live_info_query(question: str) -> bool:
    """是否为需要实时外部数据的问题。"""
    q = (question or "").strip()
    return bool(_LIVE_INFO_RE.search(q)) and not _DOCUMENT_CONTEXT_RE.search(q)


def is_general_generation_query(question: str) -> bool:
    """是否为不依赖知识库的创作、翻译或改写请求。"""
    q = (question or "").strip()
    return bool(_GENERAL_GENERATION_RE.search(q)) and not _DOCUMENT_CONTEXT_RE.search(q)


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
    "identity": "我是{assistant_name}，{org_name}的本地知识库助手。我可以帮你检索知识库、总结和梳理文档、对比材料、核对原文依据、解释数据与概念，并标注来源；没有依据时我会明确说明，不补造答案。上传文档、导入网页和切换知识库可在界面左侧操作。",
    "capability": "会分情况。知识库相关的问题，我会基于检索到的文档回答并标注来源；如果知识库中没有相关依据，我会明确说明“知识库中未找到相关信息”，不会补造答案。助手能力、使用方法等不依赖知识库的问题，我会直接回答。",
    "datetime": "今天是 {current_date}，当前时间是 {current_time}，时区为 {timezone}。",
    "live_info": "我目前只能依据知识库和本地信息作答，无法获取实时天气、新闻、股价等外部数据。你可以在知识库中补充相关资料，或直接提供需要分析的内容。",
    "general_generation": "这个知识库助手只基于已入库资料作答，不能直接进行脱离资料的创作、翻译或改写。你可以上传相关资料后让我基于资料完成，例如写摘要、改写、提炼要点或生成对比说明。",
    "thanks": "不客气！如果还有关于知识库内容的问题，随时找我。",
    "farewell": "再见！有问题欢迎随时回来查询。",
}


def classify_query(question: str) -> RouteDecision:
    """对用户问题做意图分类。

    顺序：寒暄（全句匹配）→ 身份 → 回答边界 → 当前日期时间 →
    实时外部信息 → 通用创作 → 聚合 → 默认事实问答。
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

    if is_capability_boundary_question(q):
        logger.info("查询路由：chitchat(capability)「%s」", q[:30])
        return RouteDecision(intent="chitchat", reply=_CHITCHAT_TEMPLATES["capability"])

    if is_datetime_query(q):
        logger.info("查询路由：chitchat(datetime)「%s」", q[:30])
        return RouteDecision(intent="chitchat", reply=_CHITCHAT_TEMPLATES["datetime"])

    if is_live_info_query(q):
        logger.info("查询路由：chitchat(live_info)「%s」", q[:30])
        return RouteDecision(intent="chitchat", reply=_CHITCHAT_TEMPLATES["live_info"])

    if is_general_generation_query(q):
        logger.info("查询路由：chitchat(general_generation)「%s」", q[:30])
        return RouteDecision(intent="chitchat", reply=_CHITCHAT_TEMPLATES["general_generation"])

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
    "chitchat": "寒暄、道谢、道别，或询问助手身份、能力边界、当前日期时间、实时外部信息，以及脱离资料的创作/翻译请求等无需知识库内容的问题",
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
    reply = None
    if intent == "chitchat":
        if is_capability_boundary_question(q):
            reply = _CHITCHAT_TEMPLATES["capability"]
        elif is_datetime_query(q):
            reply = _CHITCHAT_TEMPLATES["datetime"]
        elif is_live_info_query(q):
            reply = _CHITCHAT_TEMPLATES["live_info"]
        elif is_general_generation_query(q):
            reply = _CHITCHAT_TEMPLATES["general_generation"]
        elif _IDENTITY_RE.search(q):
            reply = _CHITCHAT_TEMPLATES["identity"]
        else:
            reply = _CHITCHAT_TEMPLATES["greeting"]
    return RouteDecision(intent=intent, reply=reply)
