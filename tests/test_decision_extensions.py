"""d1 决策层扩展测试：反馈自动分流 / 检索充分性 / 入库质检门禁。

策略与 ``test_decision_provider.py`` 一致：用可编程 fake provider 驱动，
不依赖真实模型权重；重点验证"默认关闭时零行为变化"与"失败必降级"两条铁律。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from src.decision_provider import (
    BaseDecisionProvider,
    ChoiceResult,
    _coerce_result,
    _validate_spec,
)
from src.feedback_router import (
    CATEGORY_RETRIEVAL,
    FeedbackRoute,
    build_feedback_text,
    classify_feedback,
)
from src.ingest_guard import IngestGuardResult, QUALITY_LADDER, guard_chunks
from src.rag_pipeline import FALLBACK_NO_HIT_ANSWER, RAGPipeline
from src.vector_store import Hit


# ======================================================================
#  测试替身
# ======================================================================
class _FakeProvider(BaseDecisionProvider):
    """可编程 provider：handler(name, spec, text) -> payload(dict|float|None)。"""

    name = "fake"

    def __init__(self, handler):
        self.handler = handler
        self.decide_calls = 0

    def decide(self, text: str, questions: Dict[str, Dict[str, Any]]):
        self.decide_calls += 1
        out = {}
        for name, spec in questions.items():
            payload = self.handler(name, spec, text)
            if payload is None:
                continue
            out[name] = _coerce_result(name, _validate_spec(name, spec), payload)
        return out


class _BoomProvider(BaseDecisionProvider):
    name = "boom"

    def decide(self, text, questions):
        raise RuntimeError("provider 炸了")


class _ChoiceProvider(BaseDecisionProvider):
    """直接返回预设 ChoiceResult / NoulResult 的极简 provider。"""

    name = "stub"

    def __init__(self, choice_label=None, choice_conf=0.9, noul_prob=None):
        self.choice_label = choice_label
        self.choice_conf = choice_conf
        self.noul_prob = noul_prob

    def decide(self, text, questions):
        out = {}
        for name, spec in questions.items():
            qtype = _validate_spec(name, spec)["type"]
            if qtype == "choice" and self.choice_label is not None:
                out[name] = _coerce_result(
                    name,
                    _validate_spec(name, spec),
                    {"label": self.choice_label, "confidence": self.choice_conf},
                )
            elif qtype == "noul" and self.noul_prob is not None:
                out[name] = _coerce_result(
                    name, _validate_spec(name, spec), {"probability": self.noul_prob}
                )
        return out


class _RawChoiceProvider(BaseDecisionProvider):
    """绕过原语归一化，直接给一个预设 label 的 provider（测防御分支）。"""

    name = "raw"

    def __init__(self, label, conf=0.9):
        self.label, self.conf = label, conf

    def decide(self, text, questions):  # pragma: no cover - 不走这条路
        return {}

    def choice(self, text, instructions, criteria, key="choice"):
        return ChoiceResult(
            label=self.label, confidence=self.conf, probabilities={self.label: self.conf}
        )


def _chunk(text: str, **meta) -> SimpleNamespace:
    return SimpleNamespace(text=text, metadata=dict(meta))


def _hit(text: str, score: float = 0.5) -> Hit:
    return Hit(id="h1", text=text, score=score, metadata={"source": "doc.md"})


# ======================================================================
#  1. 反馈自动分流
# ======================================================================
class TestFeedbackRouter:
    def test_provider_none_returns_none(self):
        assert classify_feedback("为什么答错了", provider=None) is None

    def test_empty_text_returns_none(self):
        p = _ChoiceProvider(choice_label=CATEGORY_RETRIEVAL)
        assert classify_feedback("", "", "", p) is None

    def test_high_confidence_returns_route(self):
        p = _ChoiceProvider(choice_label=CATEGORY_RETRIEVAL, choice_conf=0.88)
        route = classify_feedback("知识库里有但没搜到", "抱歉没找到", provider=p)
        assert isinstance(route, FeedbackRoute)
        assert route.category == CATEGORY_RETRIEVAL
        assert route.label  # 有中文短标签
        assert abs(route.confidence - 0.88) < 1e-9

    def test_low_confidence_returns_none(self):
        p = _ChoiceProvider(choice_label=CATEGORY_RETRIEVAL, choice_conf=0.3)
        assert classify_feedback("q", provider=p, min_confidence=0.5) is None

    def test_unknown_label_returns_none(self):
        assert classify_feedback("q", provider=_RawChoiceProvider("不存在的类别")) is None

    def test_provider_exception_degrades(self):
        assert classify_feedback("q", provider=_BoomProvider()) is None

    def test_build_feedback_text_omits_empties(self):
        assert build_feedback_text("问题甲") == "原问题：问题甲"
        text = build_feedback_text("问题甲", "答案乙", "应该是丙")
        assert "问题甲" in text and "答案乙" in text and "应该是丙" in text


# ======================================================================
#  2. 检索充分性决策
# ======================================================================
def _make_pipeline(config: Dict[str, Any], provider: Any, hits):
    """构造一个只够 answer() 跑通的 pipeline 替身（不走 __init__）。"""
    p = RAGPipeline.__new__(RAGPipeline)
    p.config = config
    p.decision_provider = provider
    p._decision_state_checked = True
    p.reranker = None
    p.context_assembler = SimpleNamespace(enabled=False)
    p.llm = SimpleNamespace(gen_cfg=None, chat=lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("检索不足时不应调用 LLM 生成")
    ))
    p.ensure_llm = lambda: None
    p._classify = lambda q: SimpleNamespace(intent="fact", reply=None)
    p._retrieve = lambda *a, **k: list(hits)
    p._verify = lambda *a, **k: None
    return p


_SUFF_CFG = {
    "rag": {"fallback_answer": FALLBACK_NO_HIT_ANSWER, "top_k": 4},
    "retrieval": {"top_k": 4},
    "decision": {
        "enabled": True,
        "retrieval_sufficiency": {"enabled": True, "min_probability": 0.25},
    },
}


class TestRetrievalSufficiency:
    def test_disabled_returns_none(self):
        p = _make_pipeline(
            {"decision": {"retrieval_sufficiency": {"enabled": False}}},
            _ChoiceProvider(noul_prob=0.0),
            [_hit("x")],
        )
        assert p._assess_retrieval_sufficiency("q", [_hit("x")]) is None

    def test_no_hits_returns_none(self):
        p = _make_pipeline(_SUFF_CFG, _ChoiceProvider(noul_prob=0.0), [])
        assert p._assess_retrieval_sufficiency("q", []) is None

    def test_provider_none_returns_none(self):
        p = _make_pipeline(_SUFF_CFG, None, [_hit("x")])
        assert p._assess_retrieval_sufficiency("q", [_hit("x")]) is None

    def test_insufficient_returns_false(self):
        p = _make_pipeline(_SUFF_CFG, _ChoiceProvider(noul_prob=0.05), [_hit("x")])
        assert p._assess_retrieval_sufficiency("q", [_hit("x")]) is False

    def test_sufficient_returns_true(self):
        p = _make_pipeline(_SUFF_CFG, _ChoiceProvider(noul_prob=0.9), [_hit("x")])
        assert p._assess_retrieval_sufficiency("q", [_hit("x")]) is True

    def test_answer_skips_generation_when_insufficient(self):
        """检索不足时应直接返回兜底文案，绝不调用 LLM。"""
        p = _make_pipeline(_SUFF_CFG, _ChoiceProvider(noul_prob=0.02), [_hit("沾边内容")])
        result = p.answer("一个知识库里没有的问题")
        assert result.answer == FALLBACK_NO_HIT_ANSWER


# ======================================================================
#  3. 入库质检门禁
# ======================================================================
def _quality_handler(low_text="乱码"):  # noqa: ARG001
    """低质文本给低分，其它给高分；合规一律判否。"""

    def handler(name, spec, text):
        if name == "quality":
            idx = 0 if "乱码" in text else len(QUALITY_LADDER) - 1
            return {"level": idx + 1}
        if name == "compliance":
            return {"probability": 0.01}
        return None

    return handler


def _bad_compliance_handler():
    def handler(name, spec, text):
        if name == "quality":
            return {"level": len(QUALITY_LADDER)}
        if name == "compliance":
            return {"probability": 0.95}
        return None

    return handler


_GUARD_CFG_FLAG = {"enabled": True, "action": "flag", "min_quality_score": 0.3}
_GUARD_CFG_DROP = {"enabled": True, "action": "drop", "min_quality_score": 0.3}


class TestIngestGuard:
    def test_disabled_keeps_all(self):
        chunks = [_chunk("乱码内容")]
        res = guard_chunks(chunks, _FakeProvider(_quality_handler()), {"enabled": False})
        assert isinstance(res, IngestGuardResult)
        assert res.kept == chunks and not res.flagged and not res.dropped

    def test_provider_none_keeps_all(self):
        chunks = [_chunk("乱码内容")]
        res = guard_chunks(chunks, None, _GUARD_CFG_FLAG)
        assert res.kept == chunks and not res.flagged

    def test_flag_mode_marks_but_keeps(self):
        chunks = [_chunk("乱码内容", source="a.md"), _chunk("正常知识内容", source="b.md")]
        res = guard_chunks(chunks, _FakeProvider(_quality_handler()), _GUARD_CFG_FLAG)
        assert len(res.kept) == 2 and not res.dropped
        assert res.report["low_quality"] == 1
        assert chunks[0].metadata["_guard_low_quality"] is True
        assert chunks[1].metadata.get("_guard_low_quality") is None

    def test_drop_mode_removes_low_quality(self):
        chunks = [_chunk("乱码内容", source="a.md"), _chunk("正常知识内容", source="b.md")]
        res = guard_chunks(chunks, _FakeProvider(_quality_handler()), _GUARD_CFG_DROP)
        assert len(res.kept) == 1 and len(res.dropped) == 1
        assert res.report["dropped"] == 1

    def test_suspicious_content_flagged(self):
        chunks = [_chunk("正常内容", source="a.md")]
        res = guard_chunks(chunks, _FakeProvider(_bad_compliance_handler()), _GUARD_CFG_FLAG)
        assert res.report["suspicious"] == 1
        assert chunks[0].metadata["_guard_suspicious"] is True

    def test_provider_exception_keeps_all(self):
        chunks = [_chunk("乱码内容")]
        res = guard_chunks(chunks, _BoomProvider(), _GUARD_CFG_FLAG)
        assert res.kept == chunks and not res.flagged

    def test_max_chunks_budget(self):
        chunks = [_chunk(f"乱码{i}") for i in range(5)]
        p = _FakeProvider(_quality_handler())
        res = guard_chunks(chunks, p, {"enabled": True, "action": "flag", "max_chunks": 2})
        assert res.report["checked"] == 2
        # 超预算的 3 块直接保留、不被检查
        assert p.decide_calls == 2
        assert len(res.kept) == 5

    def test_single_decide_call_per_chunk(self):
        """两个原语必须在同一次 decide 调用里问完（对齐 d1 共享一次前向）。"""
        p = _FakeProvider(_quality_handler())
        guard_chunks([_chunk("正常知识内容")], p, _GUARD_CFG_FLAG)
        assert p.decide_calls == 1
