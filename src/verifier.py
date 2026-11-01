"""可验证性校验器（Verifiability Checker）。

核心指标："可被验证"——答案中每条事实都应能回溯到检索到的上下文，
而不是模型自己的先验知识。本模块用**纯规则**（无 LLM、毫秒级）对生成
答案做事后校验，产出结构化 ``verification`` 报告：

- ``refused``            是否为"未命中拒答"（fallback 文案）
- ``invalid_citations``  越界/无效的引用编号（如只给了 3 个片段却引用 [5]）
- ``cited_sentence_ratio`` 正文句子中带有效引用的比例
- ``unsupported_facts``  答案出现、但所有上下文中都不存在的"硬事实"
                         （数字/百分比/日期/金额）——编造风险的最强信号
- ``status``             verified / partial / unverified / refused

该报告随 ``RAGResult.verification`` 返回给调用方与反馈层，评估脚本据此
统计引用有效率，形成"准确率可度量 → 差评可归因 → 迭代有抓手"的闭环。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .utils import get_logger

logger = get_logger("verifier")

# 正文与"参考资料"附录的分隔（按 prompt 约定）
_REF_SECTION_RE = re.compile(r"参考资料[:：]?\s*$", re.MULTILINE)
# 引用编号：[1]、[1][2]、[1,2]
_CITE_RE = re.compile(r"\[([0-9]{1,2})\]")
# 硬事实：带单位/量级的数字（百分比、金额、日期、数量），忽略普通序号
_FACT_PATTERNS = [
    re.compile(r"\d+(?:\.\d+)?\s*%"),
    re.compile(r"\d{4}\s*年(?:\d{1,2}\s*月)?(?:\d{1,2}\s*日)?"),
    re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:万元|亿元|元|天|日|小时|分钟|人|次|个|条|台|月|年|km|kg|m²|㎡)"),
    re.compile(r"\d{3,}"),
]
# 剔除列表序号（"1. "、"（1）"、"[1]" 引用本身）
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|\n")
# 接地检查词元：CJK 单字 + 拉丁/数字词（词面级，捕获"上下文里根本没有的词"）
_GROUND_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
# 引用标记（从句子文本中剥离）
_CITE_MARKER_RE = re.compile(r"\[[0-9]{1,2}(?:\s*[,，，]\s*[0-9]{1,2})*\]")

# 提炼 / 判断句的标记词。
#
# 为什么需要它：接地检查（下面 4.5 节）用"句子词元有多大比例出现在所引
# 上下文里"来抓编造。但**提炼本身就是用原文没有的词重组原文有的信息**——
# "受害顺序与受损顺序并不重合""决策是被数据逼出来的，并非主动设计"这类
# 判断句，词面接地率天然极低，于是被判成 unsupported，两个就 unverified。
# 结果是任何做了综合的回答都不合格，模型只能退回去逐句摘抄原文。
# 有了标记词，才能把"会总结"和"在编造"区分开。
_SYNTHESIS_MARKERS = (
    # 因果 / 推论
    "所以", "因此", "因而", "导致", "使得", "意味着", "说明", "表明", "反映", "印证",
    # 判断 / 归因
    "本质", "实质", "真正", "核心", "关键", "根源", "动机", "意图", "想表达", "想论证",
    "目的", "无异于", "并非", "并不", "不是", "而是", "恰恰", "反而",
    # 对比 / 结构
    "对比", "相较", "相比", "转折", "矛盾", "悖论", "错位", "代价", "受益", "受损",
    "总体", "整体", "归纳", "总结", "概括", "层次", "递进",
    # 程度 / 时间强调
    "始终", "更大", "越来越", "最终",
)


def _is_synthesis_sentence(sentence: str) -> bool:
    """该句是否属于"提炼 / 判断"而非对原文的具体陈述。

    判定依据两条，缺一不可：

    1. 句中**不含硬事实**（数字、日期、金额、比例）——带硬事实的句子仍要
       逐字接地，数字是硬约束，不允许"提炼"。
    2. 句中带概括性连接词或判断标记（见 :data:`_SYNTHESIS_MARKERS`）。
    """
    if _extract_facts(sentence or ""):
        return False
    return any(marker in sentence for marker in _SYNTHESIS_MARKERS)



@dataclass
class VerificationResult:
    """一次答案的可验证性报告。"""

    refused: bool = False
    invalid_citations: List[int] = field(default_factory=list)
    cited_sentence_ratio: float = 0.0
    unsupported_facts: List[str] = field(default_factory=list)
    # 带引用但词面内容在所引上下文中找不到依据的句子（编造/改写的直接信号）
    unsupported_sentences: List[str] = field(default_factory=list)
    status: str = "unverified"  # verified / partial / unverified / refused
    # d1 决策增强结果（noul 接地判定 + score 质量分级）；
    # 仅在调用 verify_answer_with_decision 时非空，规则路径为 None。
    decision: Optional[Dict[str, Any]] = None

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "refused": self.refused,
            "invalid_citations": list(self.invalid_citations),
            "cited_sentence_ratio": round(self.cited_sentence_ratio, 3),
            "unsupported_facts": list(self.unsupported_facts),
            "unsupported_sentences": list(self.unsupported_sentences),
            "status": self.status,
            "verified": self.verified,
        }
        # 仅在启用决策增强时追加，保持规则路径的对外字段完全不变
        if self.decision is not None:
            d["decision"] = self.decision
        return d


def _split_sentences(body: str) -> List[str]:
    """把答案正文切成句子（保留引用标记所在句）。"""
    parts = _SENT_SPLIT_RE.split(body)
    return [p.strip() for p in parts if p and p.strip() and len(p.strip()) >= 4]


def _extract_facts(text: str) -> List[str]:
    """提取硬事实 token（去重保序）。"""
    facts: List[str] = []
    for pat in _FACT_PATTERNS:
        for m in pat.finditer(text):
            tok = m.group(0).strip()
            # 过滤被空格包裹的列表序号误报（如 "1. 万元"）与纯年份泛指
            if tok and tok not in facts:
                facts.append(tok)
    return facts


def verify_answer(
    answer: str,
    context_chunks: Sequence[Dict[str, Any]],
    fallback_answer: Optional[str] = None,
    min_cited_ratio: float = 0.5,
    min_grounding: float = 0.6,
    synthesis_aware: bool = False,
) -> VerificationResult:
    """校验答案对上下文的可验证性。

    Args:
        answer: 模型生成的答案全文。
        context_chunks: 送入 prompt 的上下文块（需含 ``index``，从 1 起）。
        fallback_answer: 检索未命中时的兜底文案；答案与之相同 → ``refused``。
        min_cited_ratio: 判定 ``verified`` 所需的最低句子引用覆盖率。
        min_grounding: 引用句的词面接地率下限——句子词元（CJK 字/英文词）
          出现在所引上下文中的最低比例，低于此值视为"引用了但内容无据"。
        synthesis_aware: 是否区分"提炼句"与"陈事句"。默认 ``False`` 时行为
          与旧版完全一致。为 ``True`` 时：提炼/判断句豁免词面接地检查，
          且引用覆盖率只按"带硬事实的句子"计算 —— 否则任何做了综合、
          因果分析的回答都会被判为不合格，系统实际上在逼模型逐句照抄。
    """

    res = VerificationResult()
    answer = (answer or "").strip()
    if not answer:
        res.status = "refused"
        res.refused = True
        return res

    # 0. 拒答识别
    if fallback_answer and answer == str(fallback_answer).strip():
        res.refused = True
        res.status = "refused"
        return res

    # 1. 拆正文 / 参考资料
    m = _REF_SECTION_RE.search(answer)
    body = answer[: m.start()].strip() if m else answer

    # 2. 引用编号合法性
    valid_idx = {int(c.get("index", 0)) for c in context_chunks if c.get("index")}
    cited = [int(x) for x in _CITE_RE.findall(body)]
    res.invalid_citations = sorted({n for n in cited if n not in valid_idx})

    # 3. 句子引用覆盖率
    sentences = _split_sentences(body)
    if sentences:
        cited_sents = sum(1 for s in sentences if _CITE_RE.search(s))
        res.cited_sentence_ratio = cited_sents / len(sentences)

    # 4. 硬事实支持度：答案中的数字/日期必须出现在任一上下文块中
    context_text = "".join(str(c.get("content") or "") for c in context_chunks)
    context_norm = re.sub(r"\s+", "", context_text)
    for fact in _extract_facts(body):
        if re.sub(r"\s+", "", fact) not in context_norm:
            res.unsupported_facts.append(fact)

    # 4.5 句子级词面接地：带引用的句子，其词元必须大部分出现在所引
    # 上下文中——捕获"引用了 [1] 但内容是编造/改写"的情况（如编造的
    # 概念名、与原文相抵触的表述）。
    #
    # ``synthesis_aware=True`` 时**提炼句豁免**：判断句（"受害顺序与受损
    # 顺序并不重合""决策是被数据逼出来的"）的价值恰恰在于用原文没有的词
    # 重组原文有的信息，词面接地必然很低。豁免它，才不至于把"会总结"
    # 当成"在编造"。含硬事实的句子不豁免——数字仍是硬约束。
    chunk_tokens: Dict[int, set] = {}
    for c in context_chunks:
        idx = c.get("index")
        if idx:
            chunk_tokens[int(idx)] = set(_GROUND_TOKEN_RE.findall(re.sub(r"\s+", "", str(c.get("content") or ""))))
    for s in sentences:
        cite_nums = {int(x) for x in _CITE_RE.findall(s)}
        if not cite_nums or cite_nums - valid_idx:
            continue
        if synthesis_aware and _is_synthesis_sentence(s):
            continue
        s_tokens = _GROUND_TOKEN_RE.findall(_CITE_MARKER_RE.sub("", s))
        if not s_tokens:
            continue
        ref_tokens: set = set()
        for n in cite_nums:
            ref_tokens |= chunk_tokens.get(n, set())
        if not ref_tokens:
            continue
        grounded = sum(1 for t in s_tokens if t in ref_tokens) / len(s_tokens)
        if grounded < min_grounding:
            res.unsupported_sentences.append(s[:80])

    # 5. 综合评级
    # 覆盖率只对"带硬事实的句子"提出要求（仅 synthesis_aware 时）：要求过半
    # 句子都挂引用，等于逼模型逐句照抄原文，而整体概括与因果分析本来就没有
    # 单一出处可挂。``cited_sentence_ratio`` 字段仍是纯统计值，不受影响。
    coverage_ratio = res.cited_sentence_ratio
    if synthesis_aware:
        fact_sentences = [s for s in sentences if _extract_facts(s)]
        if fact_sentences:
            coverage_ratio = sum(
                1 for s in fact_sentences if _CITE_RE.search(s)
            ) / len(fact_sentences)
        else:
            coverage_ratio = 1.0

    if res.invalid_citations or len(res.unsupported_facts) >= 3 or len(res.unsupported_sentences) >= 2:
        res.status = "unverified"
    elif res.unsupported_facts or res.unsupported_sentences or coverage_ratio < min_cited_ratio:
        res.status = "partial"
    else:
        res.status = "verified"

    logger.debug(
        "verify: status=%s cited_ratio=%.2f invalid=%s unsupported=%s ungrounded=%d",
        res.status,
        res.cited_sentence_ratio,
        res.invalid_citations,
        res.unsupported_facts,
        len(res.unsupported_sentences),
    )
    return res


# ======================================================================
#  d1 决策增强校验（noul + score 原语）
# ======================================================================
# 质量分级阶梯（5 级，落在 d1 的 2~10 级约束内）
DEFAULT_QUALITY_LADDER = [
    "很差：内容错误或完全没有上下文依据",
    "较差：存在明显无依据的表述",
    "一般：部分内容有依据、部分存疑",
    "较好：基本都有依据，表述完整",
    "很好：完全有上下文依据且回答完整",
]

_GROUNDING_INSTRUCTION = (
    "判断「答案」中的每条事实性表述是否都能在给定的「上下文」中找到依据，"
    "不要使用任何外部知识。若答案引入了上下文里没有的信息，即视为无依据。"
)

_QUALITY_INSTRUCTION = (
    "综合评估该「答案」的质量：是否忠实于「上下文」、是否完整回答了问题、"
    "是否存在与上下文矛盾或无据编造的内容。"
)

# 状态严格度排序，便于"升/降一档"
_STATUS_ORDER = ["unverified", "partial", "verified"]


def _blend_status(
    rule_status: str,
    grounded: Optional[bool],
    grounded_prob: float,
    quality_norm: float,
    *,
    has_hard_defect: bool,
    ground_yes_threshold: float = 0.7,
    ground_no_threshold: float = 0.6,
    low_quality: float = 0.3,
    high_quality: float = 0.7,
) -> str:
    """把规则评级与 d1 决策结论融合成最终评级。

    保守原则：**降级容易、升级难**。d1 说"无依据"时倾向下调；只有当 d1
    明确说"有依据"且质量高、且规则层没有任何硬缺陷时，才允许从 partial
    上调为 verified。
    """
    status = rule_status
    if status == "refused":
        return status

    downgrade = False
    if grounded is False and grounded_prob >= ground_no_threshold:
        downgrade = True
    if quality_norm < low_quality:
        downgrade = True

    if downgrade:
        idx = _STATUS_ORDER.index(status) if status in _STATUS_ORDER else 0
        return _STATUS_ORDER[max(0, idx - 1)]

    # 升级：仅当规则无硬缺陷 + d1 明确有据 + 质量高
    if (
        status == "partial"
        and not has_hard_defect
        and grounded is True
        and grounded_prob >= ground_yes_threshold
        and quality_norm >= high_quality
    ):
        return "verified"
    return status


def verify_answer_with_decision(
    answer: str,
    context_chunks: Sequence[Dict[str, Any]],
    provider: Optional[Any] = None,
    fallback_answer: Optional[str] = None,
    min_cited_ratio: float = 0.5,
    min_grounding: float = 0.6,
    quality_ladder: Optional[Sequence[str]] = None,
    synthesis_aware: bool = False,
) -> VerificationResult:
    """在规则校验之上叠加 d1 决策校验（``noul`` 接地 + ``score`` 质量）。

    规则校验（词面接地 / 事实支持）擅长抓"字面级"编造，但抓不到"换一种说法
    的无据表述"；d1 决策恰好互补 —— 它做的是**语义级**的"是否完全有据"判断。
    两者融合后，``status`` 更保守可信，同时报告里保留规则与决策各自的结论。

    ``provider`` 为 ``None`` 时**完全等价于** :func:`verify_answer`（零行为变化）。

    Args:
        answer: 模型生成的答案全文。
        context_chunks: 送入 prompt 的上下文块（需含 ``index``，从 1 起）。
        provider: :class:`src.decision_provider.BaseDecisionProvider` 实例。
        fallback_answer: 未命中拒答文案；答案与之相同 → ``refused``。
        min_cited_ratio / min_grounding: 透传给规则校验。
        quality_ladder: 质量分级阶梯，默认 :data:`DEFAULT_QUALITY_LADDER`。

    Returns:
        :class:`VerificationResult`；``decision`` 字段含决策层原始结论。
    """
    res = verify_answer(
        answer,
        context_chunks,
        fallback_answer=fallback_answer,
        min_cited_ratio=min_cited_ratio,
        min_grounding=min_grounding,
        synthesis_aware=synthesis_aware,
    )
    if provider is None or res.refused or not (answer or "").strip():
        return res

    context_text = "\n".join(
        f"[{c.get('index', i + 1)}] {str(c.get('content') or '')[:800]}"
        for i, c in enumerate(context_chunks)
    )
    text = f"上下文：\n{context_text}\n\n答案：\n{answer.strip()}"
    ladder = list(quality_ladder or DEFAULT_QUALITY_LADDER)

    grounded_res = None
    quality_res = None
    try:
        grounded_res = provider.noul(text, _GROUNDING_INSTRUCTION, key="grounded")
        quality_res = provider.score(text, _QUALITY_INSTRUCTION, ladder, key="quality")
    except Exception as exc:  # noqa: BLE001 - provider 已兜底，这里再保险一层
        logger.warning("决策校验失败（沿用规则结果）: %s", exc)
    if grounded_res is None and quality_res is None:
        return res

    grounded = None if grounded_res is None else (grounded_res.label == "yes")
    grounded_prob = 0.0 if grounded_res is None else grounded_res.probability
    quality_score = 0.0 if quality_res is None else quality_res.score
    quality_norm = 0.0 if quality_res is None else quality_res.normalized

    has_hard_defect = bool(res.invalid_citations) or len(res.unsupported_facts) >= 3
    final_status = _blend_status(
        res.status,
        grounded,
        grounded_prob,
        quality_norm,
        has_hard_defect=has_hard_defect,
    )

    res.decision = {
        "grounded": grounded,
        "grounded_probability": round(grounded_prob, 4),
        "quality_score": round(quality_score, 4),
        "quality_normalized": round(quality_norm, 4),
        "quality_confidence": 0.0 if quality_res is None else round(quality_res.confidence, 4),
        "quality_levels": ladder,
        "status_rule": res.status,
        "status_final": final_status,
    }
    if final_status != res.status:
        logger.info("决策校验调整评级：%s → %s（grounded=%s q=%.2f）", res.status, final_status, grounded, quality_norm)
    res.status = final_status
    return res


__all__ = [
    "VerificationResult",
    "verify_answer",
    "verify_answer_with_decision",
    "DEFAULT_QUALITY_LADDER",
]
