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
    assert classify_query("你可以做什么？").intent == "chitchat"
    assert classify_query("你能帮我做什么？").intent == "chitchat"
    assert classify_query("你支持哪些功能？").intent == "chitchat"
    assert classify_query("谢谢！").intent == "chitchat"
    assert classify_query("再见").intent == "chitchat"


def test_capability_boundary_questions_are_answered_directly():
    """无知识库依据时如何处理，属于助手能力边界，不应检索文档后强行总结。"""
    for q in (
        "你会回答不存在知识库中的问题吗？",
        "你能回答知识库以外的问题吗",
        "如果知识库中没有答案，你会怎么办？",
        "你会不会编造知识库里没有的内容？",
        "没有检索到相关资料时你会怎么处理",
    ):
        d = classify_query(q)
        assert d.intent == "chitchat", f"{q!r} 应走能力边界直答，得到 {d.intent}"
        assert d.reply and "会分情况" in d.reply and "不会补造答案" in d.reply


def test_document_questions_about_missing_answer_behavior_still_use_rag():
    """询问论文/设计如何实现无答案处理时，仍应依据文档回答。"""
    for q in (
        "论文里如何设计知识库没有答案时的处理机制？",
        "文档中提到不存在知识库中的问题时怎么处理？",
        "这份需求如何描述没有检索到结果时的系统行为？",
    ):
        assert classify_query(q).intent == "fact", q


def test_datetime_questions_are_answered_without_retrieval():
    for q in (
        "现在几点了",
        "你知道现在几点吗？",
        "现在是什么时间",
        "当前日期是什么",
        "今天几号",
        "今天是星期几？",
    ):
        d = classify_query(q)
        assert d.intent == "chitchat", q
        assert d.reply and "{current_time}" in d.reply


def test_document_question_about_time_still_uses_rag():
    for q in (
        "论文中这个时间点发生了什么？",
        "文档里描述的系统时间机制是什么？",
    ):
        assert classify_query(q).intent == "fact", q


def test_live_information_questions_are_direct_refusals():
    for q in (
        "北京明天天气怎么样？",
        "现在黄金价格是多少？",
        "最新股价怎么样",
        "查一下今天的热搜",
    ):
        d = classify_query(q)
        assert d.intent == "chitchat", q
        assert d.reply and "无法获取实时" in d.reply


def test_document_weather_question_still_uses_rag():
    assert classify_query("论文中今天天气的描写是什么？").intent == "fact"


def test_general_generation_requests_are_directly_scoped():
    for q in (
        "帮我写一首关于秋天的诗",
        "写一个产品发布文案",
        "翻译成英文",
        "润色一下这段话",
    ):
        d = classify_query(q)
        assert d.intent == "chitchat", q
        assert d.reply and "不能直接进行脱离资料的创作" in d.reply


def test_generation_based_on_documents_still_uses_rag():
    for q in (
        "根据这篇论文写一段摘要",
        "基于上传材料润色一段结论",
        "按照文档内容写一份周报",
        "写一篇介绍这篇论文的文章",
    ):
        assert classify_query(q).intent == "fact", q


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
