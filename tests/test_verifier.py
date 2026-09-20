"""verifier 模块测试：引用合法性 / 句子覆盖率 / 事实支持度 / 综合评级。"""

from __future__ import annotations

from src.verifier import verify_answer


CHUNKS = [
    {"index": 1, "title": "考勤", "content": "公司实行每天 8 小时工作制，午休 1 小时。"},
    {"index": 2, "title": "报销", "content": "差旅费须在 7 天内提交报销单，金额上限 5000 元。"},
    {"index": 3, "title": "假期", "content": "年假为 5 天起，司龄每满一年增加一天。"},
]

FALLBACK = "抱歉，知识库中未找到相关信息，无法回答该问题。"


class TestRefused:
    def test_fallback_detected(self):
        res = verify_answer(FALLBACK, CHUNKS, fallback_answer=FALLBACK)
        assert res.refused
        assert res.status == "refused"

    def test_empty_answer(self):
        res = verify_answer("", CHUNKS)
        assert res.status == "refused"


class TestCitations:
    def test_valid_citations_verified(self):
        answer = "公司实行每天 8 小时工作制 [1]。差旅费须在 7 天内提交报销单 [2]。"
        res = verify_answer(answer, CHUNKS)
        assert res.invalid_citations == []
        assert res.cited_sentence_ratio == 1.0
        assert res.status == "verified"
        assert res.verified

    def test_out_of_range_citation(self):
        answer = "公司实行每天 8 小时工作制 [9]。"
        res = verify_answer(answer, CHUNKS)
        assert res.invalid_citations == [9]
        assert res.status == "unverified"

    def test_reference_section_excluded(self):
        answer = "公司实行每天 8 小时工作制 [1]。\n\n参考资料：\n[1] 考勤（来源：a.md，第 1 页）"
        res = verify_answer(answer, CHUNKS)
        # 参考资料中的编号不算正文引用
        assert res.invalid_citations == []
        assert res.cited_sentence_ratio == 1.0


class TestFacts:
    def test_supported_fact(self):
        answer = "差旅报销金额上限 5000 元 [2]。"
        res = verify_answer(answer, CHUNKS)
        assert res.unsupported_facts == []
        assert res.status == "verified"

    def test_unsupported_fact_flags_partial(self):
        answer = "差旅报销金额上限 9999 元 [2]。"
        res = verify_answer(answer, CHUNKS)
        assert res.unsupported_facts and "9999" in res.unsupported_facts[0]
        assert res.status == "partial"

    def test_many_unsupported_facts_unverified(self):
        answer = "上班 12 小时，年假 99 天，报销 8888 元，罚金 666 元 [1]。"
        res = verify_answer(answer, CHUNKS)
        assert len(res.unsupported_facts) >= 3
        assert res.status == "unverified"


class TestCoverage:
    def test_low_coverage_partial(self):
        # 两句都不带引用 → 覆盖率 0，且无编造事实 → partial
        answer = "公司实行每天 8 小时工作制。午休时间是 1 小时。"
        res = verify_answer(answer, CHUNKS)
        assert res.cited_sentence_ratio == 0.0
        assert res.status == "partial"


class TestGrounding:
    def test_grounded_citation_verified(self):
        # 引用句的词面全部来自所引上下文 → 不计 unsupported_sentences
        answer = "公司实行每天 8 小时工作制 [1]。"
        res = verify_answer(answer, CHUNKS)
        assert res.unsupported_sentences == []
        assert res.status == "verified"

    def test_fabricated_content_flagged(self):
        # 引用 [1] 但句子核心词在上下文里完全不存在 → 接地失败 → partial
        answer = "量子纠缠门禁系统采用区块链共识算法 [1]。"
        res = verify_answer(answer, CHUNKS)
        assert res.unsupported_sentences
        assert res.status == "partial"

    def test_two_ungrounded_sentences_unverified(self):
        answer = "量子纠缠门禁系统采用区块链共识算法 [1]。星舰引擎用炼金术驱动 [2]。"
        res = verify_answer(answer, CHUNKS)
        assert len(res.unsupported_sentences) >= 2
        assert res.status == "unverified"


class TestDict:
    def test_to_dict_fields(self):
        res = verify_answer("内容 [1]。", CHUNKS)
        d = res.to_dict()
        assert set(d.keys()) == {
            "refused",
            "invalid_citations",
            "cited_sentence_ratio",
            "unsupported_facts",
            "unsupported_sentences",
            "status",
            "verified",
        }
