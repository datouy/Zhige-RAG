"""d1 决策层测试：原语归一化 / 重排 / 路由 / 工具预选 / 校验增强。

策略：用一个可编程的 fake provider 驱动所有集成点，避免依赖真实模型权重。
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.decision_provider import (
    BaseDecisionProvider,
    ChoiceResult,
    LLMDecisionProvider,
    NoulResult,
    ScoreResult,
    _coerce_result,
    _validate_spec,
    create_decision_provider,
)
from src.query_router import classify_query, classify_query_with_decision
from src.reranker import D1Reranker, create_reranker
from src.verifier import verify_answer, verify_answer_with_decision
from src.vector_store import Hit


# ======================================================================
#  测试替身
# ======================================================================
class _FakeProvider(BaseDecisionProvider):
    """可编程 provider：handler(name, spec, text) -> payload(dict|float|None)。"""

    name = "fake"

    def __init__(self, handler):
        self.handler = handler
        self.calls = 0

    def decide(self, text: str, questions: Dict[str, Dict[str, Any]]):
        out = {}
        for name, spec in questions.items():
            payload = self.handler(name, spec, text)
            if payload is None:
                continue
            self.calls += 1
            out[name] = _coerce_result(name, _validate_spec(name, spec), payload)
        return out


class _FakeLLM:
    """最小 LLM 替身：chat 恒返回预设文本。"""

    gen_cfg = None

    def __init__(self, response: str):
        self.response = response

    def chat(self, messages, generation=None, stream=False):
        return self.response


def _hit(text: str, score: float = 0.5, idx: int = 0) -> Hit:
    return Hit(id=f"h{idx}", text=text, score=score, metadata={"source": f"doc{idx}.md"})


# ======================================================================
#  原语归一化
# ======================================================================
class TestPrimitiveCoercion:
    def test_validate_spec_choice_list_to_dict(self):
        spec = _validate_spec("q", {"type": "choice", "criteria": ["a", "b"]})
        assert spec["criteria"] == {"a": "", "b": ""}

    def test_validate_spec_rejects_bad_type(self):
        with pytest.raises(ValueError):
            _validate_spec("q", {"type": "bogus"})

    def test_validate_spec_score_level_bounds(self):
        with pytest.raises(ValueError):
            _validate_spec("q", {"type": "score", "criteria": ["only-one"]})
        with pytest.raises(ValueError):
            _validate_spec("q", {"type": "score", "criteria": [str(i) for i in range(11)]})

    def test_coerce_noul_from_probability(self):
        spec = _validate_spec("q", {"type": "noul"})
        res = _coerce_result("q", spec, {"probability": 0.82})
        assert isinstance(res, NoulResult) and res.label == "yes"
        assert abs(res.probability - 0.82) < 1e-9

    def test_coerce_noul_clamps(self):
        spec = _validate_spec("q", {"type": "noul"})
        assert _coerce_result("q", spec, {"probability": 5.0}).probability == 1.0
        assert _coerce_result("q", spec, {"probability": -3}).probability == 0.0

    def test_coerce_choice_normalizes_distribution(self):
        spec = _validate_spec("q", {"type": "choice", "criteria": ["a", "b", "c"]})
        res = _coerce_result("q", spec, {"probabilities": {"a": 2, "b": 2, "c": 0}})
        assert isinstance(res, ChoiceResult)
        assert res.label in ("a", "b")
        assert abs(sum(res.probabilities.values()) - 1.0) < 1e-9
        assert res.probabilities["c"] == 0.0

    def test_coerce_choice_from_label(self):
        spec = _validate_spec("q", {"type": "choice", "criteria": ["a", "b"]})
        res = _coerce_result("q", spec, {"label": "b", "confidence": 0.8})
        assert res.label == "b"
        assert abs(res.probabilities["b"] - 0.8) < 1e-9

    def test_coerce_score_expected_value(self):
        spec = _validate_spec("q", {"type": "score", "criteria": ["l1", "l2", "l3", "l4", "l5"]})
        res = _coerce_result("q", spec, {"scores": {"l1": 0, "l2": 0, "l3": 0, "l4": 1, "l5": 0}})
        assert isinstance(res, ScoreResult)
        assert abs(res.score - 4.0) < 1e-9  # 命中第 4 档
        assert abs(res.normalized - 0.75) < 1e-9
        assert res.confidence == 1.0

    def test_coerce_score_numeric_index(self):
        spec = _validate_spec("q", {"type": "score", "criteria": ["a", "b", "c"]})
        # 数值 3 落在 1..3 视为档位序号 → 第 3 档
        res = _coerce_result("q", spec, {"score": 3})
        assert res.score > 2.0

    def test_coerce_unwraps_single_value_wrapper(self):
        spec = _validate_spec("q", {"type": "noul"})
        assert _coerce_result("q", spec, {"value": 0.9}).probability == 0.9


# ======================================================================
#  便捷方法与工厂
# ======================================================================
class TestProviderApi:
    def test_wrappers_return_typed_results(self):
        def handler(name, spec, text):
            if spec["type"] == "noul":
                return {"probability": 0.9}
            if spec["type"] == "choice":
                return {"probabilities": {"x": 0.9, "y": 0.1}}
            return {"scores": {"a": 1, "b": 0}}

        p = _FakeProvider(handler)
        assert isinstance(p.noul("t", "ins"), NoulResult)
        assert isinstance(p.choice("t", "ins", ["x", "y"]), ChoiceResult)
        assert isinstance(p.score("t", "ins", ["a", "b"]), ScoreResult)

    def test_provider_never_raises(self):
        def boom(name, spec, text):
            raise RuntimeError("boom")

        p = _FakeProvider(boom)
        assert p.noul("t", "ins") is None
        assert p.choice("t", "ins", ["x", "y"]) is None

    def test_empty_input_returns_none(self):
        p = _FakeProvider(lambda n, s, t: {"probability": 1.0})
        assert p.noul("", "ins") is None

    def test_factory_disabled_returns_none(self):
        assert create_decision_provider({"enabled": False}) is None
        assert create_decision_provider({"enabled": True, "backend": "none"}) is None
        assert create_decision_provider(None) is None

    def test_factory_llm_backend_with_injected_llm(self):
        p = create_decision_provider({"enabled": True, "backend": "llm"}, llm=_FakeLLM("{}"))
        assert isinstance(p, LLMDecisionProvider)


# ======================================================================
#  LLM 模拟后端
# ======================================================================
class TestLLMDecisionProvider:
    def test_parses_choice_and_noul(self):
        resp = (
            '{"intent": {"probabilities": {"fact": 0.9, "chitchat": 0.05, '
            '"aggregate": 0.05}}, "grounded": {"probability": 0.88}}'
        )
        p = LLMDecisionProvider(_FakeLLM(resp))
        out = p.decide(
            "text",
            {
                "intent": {"type": "choice", "criteria": ["fact", "chitchat", "aggregate"]},
                "grounded": {"type": "noul"},
            },
        )
        assert out["intent"].label == "fact"
        assert abs(out["intent"].confidence - 0.9) < 1e-6
        assert out["grounded"].label == "yes"

    def test_tolerates_code_fence_and_noise(self):
        resp = "好的，结果如下：\n```json\n{\"grounded\": {\"probability\": 0.1}}\n```\n以上。"
        p = LLMDecisionProvider(_FakeLLM(resp))
        out = p.decide("t", {"grounded": {"type": "noul"}})
        assert out["grounded"].label == "no"

    def test_garbage_returns_empty(self):
        p = LLMDecisionProvider(_FakeLLM("抱歉我不知道"))
        assert p.decide("t", {"g": {"type": "noul"}}) == {}


# ======================================================================
#  重排（score 原语）
# ======================================================================
class TestD1Reranker:
    def test_orders_by_decision_score(self):
        hits = [_hit("苹果价格行情", 0.5, 0), _hit("完全无关的内容", 0.5, 1), _hit("苹果价格与产地", 0.5, 2)]

        def handler(name, spec, text):
            # 只看「片段」部分（问题里也含"苹果价格"，不能据此判定相关）
            fragment = text.split("片段：")[-1]
            if "苹果价格" in fragment:
                return {"scores": {c: (1 if c == spec["criteria"][-1] else 0) for c in spec["criteria"]}}
            return {"scores": {c: (1 if c == spec["criteria"][0] else 0) for c in spec["criteria"]}}

        r = D1Reranker(_FakeProvider(handler), criteria=["低", "中", "高"], max_candidates=10)
        out = r.rerank("苹果价格", hits, top_n=3)
        assert out[0].text in ("苹果价格行情", "苹果价格与产地")
        assert out[-1].text == "完全无关的内容"
        assert out[0].metadata["_rerank_backend"] == "d1"
        assert 0.0 <= out[0].score <= 1.0

    def test_empty_hits(self):
        r = D1Reranker(_FakeProvider(lambda n, s, t: {"score": 1}), criteria=["a", "b"])
        assert r.rerank("q", [], top_n=3) == []

    def test_degrades_to_initial_order_on_failure(self):
        def boom(name, spec, text):
            raise RuntimeError("boom")

        hits = [_hit("A", 0.9, 0), _hit("B", 0.1, 1)]
        r = D1Reranker(_FakeProvider(boom), criteria=["a", "b"], max_candidates=10)
        out = r.rerank("q", hits, top_n=2)
        # 单条决策失败时沿用初检分 → 原顺序保持
        assert [h.text for h in out] == ["A", "B"]

    def test_requires_provider(self):
        with pytest.raises(ValueError):
            D1Reranker(None)


class TestRerankerFactory:
    def test_disabled_returns_none(self):
        assert create_reranker({"reranker": {"enabled": False}}) is None

    def test_d1_backend_uses_injected_provider(self):
        provider = _FakeProvider(lambda n, s, t: {"score": 1})
        r = create_reranker(
            {"reranker": {"enabled": True, "backend": "d1"}, "decision": {"enabled": True}},
            decision_provider=provider,
        )
        assert isinstance(r, D1Reranker)

    def test_d1_without_provider_falls_back_to_bge_or_none(self):
        # provider 不可用且未配置 model_name → 返回 None（不抛异常）
        r = create_reranker(
            {
                "reranker": {"enabled": True, "backend": "d1", "model_name": ""},
                "decision": {"enabled": False},
            }
        )
        assert r is None


# ======================================================================
#  查询路由（choice 原语）
# ======================================================================
class TestDecisionRouting:
    def test_none_provider_is_identical_to_rules(self):
        for q in ("你好", "参考文献有多少条", "RAG 是什么"):
            assert classify_query_with_decision(q, None).intent == classify_query(q).intent

    def test_decision_can_upgrade_fact_to_aggregate(self):
        def handler(name, spec, text):
            return {"probabilities": {"fact": 0.05, "aggregate": 0.9, "chitchat": 0.05}}

        d = classify_query_with_decision("总共收录了几份材料", _FakeProvider(handler))
        assert d.intent == "aggregate"

    def test_low_confidence_chitchat_is_rejected(self):
        def handler(name, spec, text):
            return {"probabilities": {"chitchat": 0.55, "fact": 0.45, "aggregate": 0.0}}

        # 0.55 < chitchat_min_confidence(0.75) → 保守回落 fact
        d = classify_query_with_decision("年假怎么申请", _FakeProvider(handler))
        assert d.intent == "fact"

    def test_high_confidence_chitchat_applies(self):
        def handler(name, spec, text):
            return {"probabilities": {"chitchat": 0.95, "fact": 0.05, "aggregate": 0.0}}

        d = classify_query_with_decision("随便聊聊", _FakeProvider(handler))
        assert d.intent == "chitchat"
        assert d.reply  # 有模板回应

    def test_rule_chitchat_short_circuits(self):
        calls = {"n": 0}

        def handler(name, spec, text):
            calls["n"] += 1
            return {"probabilities": {"fact": 1.0}}

        d = classify_query_with_decision("你好", _FakeProvider(handler))
        assert d.intent == "chitchat"
        assert calls["n"] == 0  # 规则已判定，未花一次决策


# ======================================================================
#  答案校验增强（noul + score）
# ======================================================================
CHUNKS = [
    {"index": 1, "title": "考勤", "content": "公司实行每天 8 小时工作制，午休 1 小时。"},
]


class TestDecisionVerification:
    def test_none_provider_matches_rules(self):
        answer = "公司实行每天 8 小时工作制 [1]。"
        a = verify_answer(answer, CHUNKS)
        b = verify_answer_with_decision(answer, CHUNKS, None)
        assert a.to_dict() == b.to_dict()
        assert "decision" not in b.to_dict()

    def test_decision_downgrades_when_not_grounded(self):
        answer = "公司实行每天 8 小时工作制 [1]。"
        assert verify_answer(answer, CHUNKS).status == "verified"

        def handler(name, spec, text):
            if spec["type"] == "noul":
                return {"probability": 0.1}  # 低概率 = 判定"无依据"
            return {"scores": {c: (1 if c == spec["criteria"][0] else 0) for c in spec["criteria"]}}

        res = verify_answer_with_decision(answer, CHUNKS, _FakeProvider(handler))
        assert res.status == "partial"
        assert res.decision["grounded"] is False
        assert res.decision["status_rule"] == "verified"
        assert res.decision["status_final"] == "partial"

    def test_to_dict_gains_decision_key_only_when_present(self):
        answer = "公司实行每天 8 小时工作制 [1]。"

        def handler(name, spec, text):
            return {"probability": 0.95} if spec["type"] == "noul" else {"score": 5}

        res = verify_answer_with_decision(answer, CHUNKS, _FakeProvider(handler))
        d = res.to_dict()
        assert "decision" in d
        assert {"grounded", "quality_score", "status_final"} <= set(d["decision"].keys())

    def test_refused_answer_skips_decision(self):
        fb = "抱歉，知识库中未找到相关信息，无法回答该问题。"
        calls = {"n": 0}

        def handler(name, spec, text):
            calls["n"] += 1
            return {"probability": 0.5}

        res = verify_answer_with_decision(fb, CHUNKS, _FakeProvider(handler), fallback_answer=fb)
        assert res.refused
        assert calls["n"] == 0


# ======================================================================
#  Agent 工具调度（choice 原语）
# ======================================================================
class TestAgentToolSelection:
    def _agent(self, provider, scripted):
        from src.agent import BuiltinTools, ReActAgent

        class _ScriptedLLM:
            gen_cfg = None

            def __init__(self, items):
                self.items = list(items)

            def chat(self, messages, generation=None, stream=False):
                return self.items.pop(0) if self.items else "Final Answer: 结束"

        return ReActAgent(
            llm=_ScriptedLLM(scripted),
            tools=BuiltinTools.create(),
            max_steps=3,
            decision_provider=provider,
        )

    def test_emits_decision_event_and_falls_back_to_tool(self):
        def handler(name, spec, text):
            return {"probabilities": {"get_current_time": 0.9, "calculator": 0.05, "text_stats": 0.05}}

        agent = self._agent(
            _FakeProvider(handler),
            [
                "Thought: 我不确定\nAction: None\nActionInput: {}\n",
                "Thought: 好了\nFinal Answer: 完成",
            ],
        )
        # 注意：查询不能命中 ReAct 的日期/时间确定性短路（"现在几点"会走本地时钟）
        events = list(agent.run("请告诉我当前的准确时间", stream=False))
        assert any(e["event"] == "decision" and e["data"]["tool"] == "get_current_time" for e in events)
        obs = [e for e in events if e["event"] == "observation"]
        assert obs and obs[0]["data"]["ok"] is True
        done = next(e for e in events if e["event"] == "done")
        assert "get_current_time" in done["data"]["tools_used"]

    def test_no_provider_emits_no_decision_event(self):
        agent = self._agent(
            None,
            [
                "Thought: t\nAction: calculator\nActionInput: {\"expression\": \"1+1\"}\n",
                "Thought: ok\nFinal Answer: 2",
            ],
        )
        events = list(agent.run("1+1", stream=False))
        assert not any(e["event"] == "decision" for e in events)

    def test_low_confidence_suggestion_ignored(self):
        def handler(name, spec, text):
            return {"probabilities": {"get_current_time": 0.3, "calculator": 0.3, "text_stats": 0.4}}

        agent = self._agent(
            _FakeProvider(handler),
            [
                "Thought: t\nAction: None\nActionInput: {}\n",
                "Thought: ok\nFinal Answer: 直接回答",
            ],
        )
        events = list(agent.run("随便问问", stream=False))
        # 置信度 0.4 < 0.5 → 不触发兜底工具执行
        assert not any(e["event"] == "observation" for e in events)
