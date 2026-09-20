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

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "refused": self.refused,
            "invalid_citations": list(self.invalid_citations),
            "cited_sentence_ratio": round(self.cited_sentence_ratio, 3),
            "unsupported_facts": list(self.unsupported_facts),
            "unsupported_sentences": list(self.unsupported_sentences),
            "status": self.status,
            "verified": self.verified,
        }


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
) -> VerificationResult:
    """校验答案对上下文的可验证性。

    Args:
        answer: 模型生成的答案全文。
        context_chunks: 送入 prompt 的上下文块（需含 ``index``，从 1 起）。
        fallback_answer: 检索未命中时的兜底文案；答案与之相同 → ``refused``。
        min_cited_ratio: 判定 ``verified`` 所需的最低句子引用覆盖率。
        min_grounding: 引用句的词面接地率下限——句子词元（CJK 字/英文词）
          出现在所引上下文中的最低比例，低于此值视为"引用了但内容无据"。
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
    chunk_tokens: Dict[int, set] = {}
    for c in context_chunks:
        idx = c.get("index")
        if idx:
            chunk_tokens[int(idx)] = set(_GROUND_TOKEN_RE.findall(re.sub(r"\s+", "", str(c.get("content") or ""))))
    for s in sentences:
        cite_nums = {int(x) for x in _CITE_RE.findall(s)}
        if not cite_nums or cite_nums - valid_idx:
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
    if res.invalid_citations or len(res.unsupported_facts) >= 3 or len(res.unsupported_sentences) >= 2:
        res.status = "unverified"
    elif res.unsupported_facts or res.unsupported_sentences or res.cited_sentence_ratio < min_cited_ratio:
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
