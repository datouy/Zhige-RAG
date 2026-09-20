"""查询意图路由测试：寒暄拦截必须保守，真问题绝不能被误拦。"""

from src.query_router import RouteDecision, classify_query, is_aggregate_query


def test_chitchat_greeting():
    for q in ("你好", "您好！", "hello", "Hi~", "哈喽。"):
        d = classify_query(q)
        assert d.intent == "chitchat", f"{q!r} 应为寒暄，得到 {d.intent}"
        assert d.reply  # 模板含 {assistant_name} 占位符，由管线用运行时变量渲染


def test_chitchat_identity_thanks_farewell():
    assert classify_query("你是谁").intent == "chitchat"
    assert classify_query("你能做什么？").intent == "chitchat"
    assert classify_query("谢谢！").intent == "chitchat"
    assert classify_query("再见").intent == "chitchat"


def test_real_question_with_greeting_prefix_not_swallowed():
    """带寒暄前缀的真问题必须正常走检索，不允许被路由吞掉。"""
    for q in ("你好，请问年假怎么申请？", "您好，知识库的检索流程是什么", "嗨，参考文献有多少条"):
        d = classify_query(q)
        assert d.intent != "chitchat", f"{q!r} 被误判为寒暄"


def test_aggregate_intent():
    assert is_aggregate_query("这篇论文引用了多少文献？")
    assert is_aggregate_query("参考文献有多少条")
    assert not is_aggregate_query("文献综述写了多少字")  # 间隔超限，防误触发
    assert not is_aggregate_query("RAG 的核心思想是什么")


def test_fact_default_and_empty():
    assert classify_query("知识库的检索流程是什么").intent == "fact"
    assert classify_query("").intent == "fact"


def test_route_decision_fields():
    d = classify_query("你好")
    assert isinstance(d, RouteDecision) and d.reply
