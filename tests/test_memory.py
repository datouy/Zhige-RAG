"""memory 模块测试：即时记忆 / 长期记忆 / 查询改写 / 上下文组装。"""

from __future__ import annotations

import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.models import Base
from src.memory import (
    ContextAssembler,
    LongTermMemoryStore,
    SessionMemory,
    rewrite_query_for_retrieval,
)


# ----------------------------------------------------------------------
#  即时记忆
# ----------------------------------------------------------------------
class TestSessionMemory:
    def test_window_keeps_last_turns(self):
        mem = SessionMemory(max_turns=2, ttl_minutes=5)
        for i in range(4):
            mem.append("u1", "s1", "user", f"问题{i}")
            mem.append("u1", "s1", "assistant", f"回答{i}")
        hist = mem.get_history("u1", "s1")
        assert len(hist) == 4  # 2 轮 = 4 条
        assert hist[0]["content"] == "问题2"
        assert hist[-1]["content"] == "回答3"

    def test_isolation_by_session_and_user(self):
        mem = SessionMemory(max_turns=3)
        mem.append("u1", "s1", "user", "A")
        mem.append("u1", "s2", "user", "B")
        mem.append("u2", "s1", "user", "C")
        assert [h["content"] for h in mem.get_history("u1", "s1")] == ["A"]
        assert [h["content"] for h in mem.get_history("u1", "s2")] == ["B"]
        assert [h["content"] for h in mem.get_history("u2", "s1")] == ["C"]

    def test_ttl_expiry(self):
        mem = SessionMemory(max_turns=3, ttl_minutes=0.01)  # 0.6s
        mem.append("u1", "s1", "user", "早期消息")
        # 直接篡改时间戳模拟过期
        with mem._lock:
            for t in mem._sessions[mem._key("u1", "s1")]:
                t.ts -= 120
        assert mem.get_history("u1", "s1") == []

    def test_clear(self):
        mem = SessionMemory(max_turns=3)
        mem.append("u1", "s1", "user", "x")
        assert mem.clear("u1", "s1") == 1
        assert mem.get_history("u1", "s1") == []

    def test_disabled(self):
        mem = SessionMemory(max_turns=0)
        mem.append("u1", "s1", "user", "x")
        assert mem.get_history("u1", "s1") == []


# ----------------------------------------------------------------------
#  长期记忆（内存 SQLite）
# ----------------------------------------------------------------------
@pytest.fixture()
def ltm():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return LongTermMemoryStore(session_factory=sessionmaker(bind=engine))


class TestLongTermMemory:
    def test_remember_and_recall(self, ltm):
        assert ltm.remember("u1", "报销上限", "5000 元")
        assert ltm.remember("u1", "所属部门", "研发部")
        items = ltm.recall("u1")
        assert len(items) == 2
        assert {i["key"] for i in items} == {"报销上限", "所属部门"}

    def test_upsert_same_key(self, ltm):
        ltm.remember("u1", "部门", "研发部")
        ltm.remember("u1", "部门", "市场部")
        items = ltm.recall("u1")
        assert len(items) == 1
        assert items[0]["value"] == "市场部"

    def test_user_isolation(self, ltm):
        ltm.remember("u1", "k", "v1")
        ltm.remember("u2", "k", "v2")
        assert ltm.recall("u1")[0]["value"] == "v1"

    def test_forget(self, ltm):
        ltm.remember("u1", "k", "v")
        assert ltm.forget("u1", "k")
        assert ltm.recall("u1") == []

    def test_invalid_input_rejected(self, ltm):
        assert not ltm.remember("", "k", "v")
        assert not ltm.remember("u1", "", "v")
        assert not ltm.remember("u1", "k", "")

    def test_extract_remember_instruction(self):
        facts = LongTermMemoryStore.extract_from_text("请记住：我是研发部的工程师")
        assert ("用户叮嘱", "我是研发部的工程师") in facts

    def test_extract_profile(self):
        facts = LongTermMemoryStore.extract_from_text("我们的报销上限是每月5000元")
        assert ("报销上限", "每月5000元") in facts

    def test_maybe_remember_writes(self, ltm):
        written = ltm.maybe_remember_from_message("u1", "请记住：周五例会下午三点")
        assert written == [("用户叮嘱", "周五例会下午三点")]
        assert ltm.recall("u1")[0]["value"] == "周五例会下午三点"
        # 普通消息不写入
        assert ltm.maybe_remember_from_message("u1", "今天天气怎么样") == []


# ----------------------------------------------------------------------
#  检索查询改写
# ----------------------------------------------------------------------
class TestRewriteQuery:
    def test_deictic_query_concatenates_history(self):
        history = [{"role": "user", "content": "公司的年假政策是什么"}, {"role": "assistant", "content": "年假 5 天起"}]
        assert rewrite_query_for_retrieval("它怎么申请？", history) == "公司的年假政策是什么；它怎么申请？"

    def test_plain_query_unchanged(self):
        history = [{"role": "user", "content": "上一问"}]
        assert rewrite_query_for_retrieval("报销制度的具体流程", history) == "报销制度的具体流程"

    def test_no_history_unchanged(self):
        assert rewrite_query_for_retrieval("它是什么", []) == "它是什么"


# ----------------------------------------------------------------------
#  上下文组装器
# ----------------------------------------------------------------------
class _FakePrompts:
    def __init__(self):
        self.prompts = {
            "system": "你是 {assistant_name}，{org_name} 的助手。今天是 {current_date}。",
            "user_template": "【已知上下文】\n{context}\n\n【用户问题】\n{question}",
            "context_chunk": "[{index}] {title}\n{content}",
            "empty_context": "没有上下文。",
        }

    def system(self) -> str:
        return self.prompts["system"].strip()


class TestContextAssembler:
    def test_runtime_vars_injected(self):
        asm = ContextAssembler(_FakePrompts(), {})
        msgs = asm.build(
            "问题",
            [{"index": 1, "title": "T", "content": "C"}],
            runtime={"assistant_name": "小识", "org_name": "ACME", "current_date": "2026年09月05日"},
        )
        assert msgs[0]["role"] == "system"
        assert "小识" in msgs[0]["content"]
        assert "ACME" in msgs[0]["content"]
        assert "{assistant_name}" not in msgs[0]["content"]

    def test_missing_runtime_vars_left_alone(self):
        asm = ContextAssembler(_FakePrompts(), {})
        msgs = asm.build("问题", [{"index": 1, "title": "T", "content": "C"}], runtime={})
        assert "{assistant_name}" in msgs[0]["content"]

    def test_history_between_system_and_question(self):
        asm = ContextAssembler(_FakePrompts(), {"session_turns": 3})
        msgs = asm.build(
            "新问题",
            [{"index": 1, "title": "T", "content": "C"}],
            history=[
                {"role": "user", "content": "老问题"},
                {"role": "assistant", "content": "老回答"},
            ],
            runtime={"assistant_name": "A", "org_name": "O", "current_date": "D"},
        )
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]
        assert msgs[-1]["content"].endswith("新问题")

    def test_long_term_memory_block(self):
        asm = ContextAssembler(_FakePrompts(), {})
        msgs = asm.build(
            "问题",
            [{"index": 1, "title": "T", "content": "C"}],
            long_term=[{"key": "部门", "value": "研发部"}],
            runtime={"assistant_name": "A", "org_name": "O", "current_date": "D"},
        )
        assert "用户长期记忆" in msgs[0]["content"]
        assert "部门：研发部" in msgs[0]["content"]

    def test_memory_disabled(self):
        asm = ContextAssembler(_FakePrompts(), {"enabled": False})
        msgs = asm.build(
            "问题",
            [{"index": 1, "title": "T", "content": "C"}],
            history=[{"role": "user", "content": "老问题"}],
            long_term=[{"key": "k", "value": "v"}],
        )
        assert len(msgs) == 2  # system + 当前问题（历史与长期记忆均不注入）
        assert "长期记忆" not in msgs[0]["content"]

    def test_empty_context_path(self):
        asm = ContextAssembler(_FakePrompts(), {})
        msgs = asm.build("问题", [], runtime={"assistant_name": "A", "org_name": "O", "current_date": "D"})
        assert "没有上下文" in msgs[-1]["content"]
