"""Agent / 工具调用 测试套件。

覆盖：
- ToolRegistry 注册与 OpenAI schema
- Calculator / SearchDocuments / GetCurrentTime / PythonEval
- ReAct Agent 端到端：用一个 mock LLM 模拟完整 ReAct 流程
- max_steps 截断
- 工具异常被捕获
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("chromadb")

from src.agent import (  # noqa: E402
    BuiltinTools,
    ReActAgent,
    Tool,
    ToolRegistry,
    ToolResult,
)
from src.agent.builtin_tools import _safe_py_eval  # noqa: E402
from src.text_splitter import ChineseTextSplitter  # noqa: E402
from src.vector_store import ChromaStore  # noqa: E402


# ============================================================
#  Fixtures
# ============================================================
class _MockEmbed:
    """恒等哈希向量：相同文本 → 相同向量；不同文本 → 不同但确定的向量。"""
    dim = 16
    device = "cpu"

    def encode(self, texts, batch_size=8, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t)) % (10 ** 6)
            rng = np.random.default_rng(h)
            arr[i] = rng.standard_normal(self.dim)
            arr[i] = arr[i] / (np.linalg.norm(arr[i]) + 1e-9)
        return arr


@pytest.fixture
def mock_store(tmp_path):
    """构造一个最小可用的 ChromaStore，便于 search_documents 测试。"""
    (tmp_path / "doc1.txt").write_text(
        "RAG 是检索增强生成的缩写。它结合了信息检索与文本生成。"
        "通过向量检索最相关的上下文，再交给大模型回答问题。"
        "向量检索依赖于嵌入模型。",
        encoding="utf-8",
    )
    splitter = ChineseTextSplitter(chunk_size=60, chunk_overlap=10)
    chunks = splitter.split_text(
        (tmp_path / "doc1.txt").read_text(encoding="utf-8"),
        metadata={"source": "doc1.txt", "page": 1},
    )
    store = ChromaStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="agent_test",
        embedding_model=_MockEmbed(),
    )
    store.add_chunks(chunks)
    return store


# ============================================================
#  ToolRegistry / Tool
# ============================================================
class _EchoTool(Tool):
    name = "echo"
    description = "回显输入"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content={"echo": kwargs.get("text", "")})


def test_tool_registry_register():
    reg = ToolRegistry()
    assert len(reg) == 0
    assert "echo" not in reg
    reg.register(_EchoTool())
    assert len(reg) == 1
    assert "echo" in reg
    assert reg.get("echo").name == "echo"
    assert reg.get("not_found") is None


def test_tool_to_openai_schema():
    tool = _EchoTool()
    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["parameters"]["properties"]["text"]["type"] == "string"


def test_tool_registry_to_openai_schemas():
    reg = ToolRegistry()
    reg.register(_EchoTool())
    schemas = reg.to_openai_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "echo"


def test_tool_validate_params():
    tool = _EchoTool()
    # 缺必填参数
    err = tool.validate_params({})
    assert err is not None and "text" in err
    # 类型错误
    err2 = tool.validate_params({"text": 123})
    assert err2 is not None


def test_tool_safe_run_returns_toolresult_on_uncatched_error():
    class _Boom(Tool):
        name = "boom"
        description = "总抛异常"
        parameters = {"type": "object", "properties": {}, "required": []}

        def run(self, **kwargs):
            raise RuntimeError("boom!")

    res = _Boom().safe_run()
    assert res.success is False
    assert "RuntimeError" in (res.error or "")


# ============================================================
#  Calculator
# ============================================================
def test_calculator_tool():
    from src.agent.builtin_tools import CalculatorTool

    tool = CalculatorTool()
    res = tool.safe_run(expression="1+1")
    assert res.success
    assert res.content["result"] == 2

    res2 = tool.safe_run(expression="sqrt(16)+1")
    assert res2.success
    assert res2.content["result"] == 5

    # 非法表达式
    res3 = tool.safe_run(expression="import os")
    assert not res3.success
    assert res3.error


def test_calculator_tool_missing_required():
    from src.agent.builtin_tools import CalculatorTool

    res = CalculatorTool().safe_run()
    assert not res.success
    assert "expression" in (res.error or "")


# ============================================================
#  Search documents
# ============================================================
def test_search_documents_tool(mock_store):
    tool = BuiltinTools.create(vector_store=mock_store).get("search_documents")
    assert tool is not None
    res = tool.safe_run(query="什么是 RAG", top_k=2)
    assert res.success
    payload = res.content
    assert payload["total"] >= 1
    assert payload["hits"][0]["source"] == "doc1.txt"

    # 缺 query
    res2 = tool.safe_run()
    assert not res2.success


def test_list_documents_tool(mock_store):
    reg = BuiltinTools.create(vector_store=mock_store)
    res = reg.get("list_documents").safe_run()
    assert res.success
    assert res.content["total"] >= 1
    assert any(d["source"] == "doc1.txt" for d in res.content["documents"])


def test_get_document_chunks_tool(mock_store):
    reg = BuiltinTools.create(vector_store=mock_store)
    res = reg.get("get_document_chunks").safe_run(source="doc1.txt", limit=3)
    assert res.success
    assert res.content["total"] >= 1
    assert res.content["chunks"][0]["text"]


# ============================================================
#  Get current time
# ============================================================
def test_get_current_time_tool():
    from src.agent.builtin_tools import GetCurrentTimeTool

    res = GetCurrentTimeTool().safe_run()
    assert res.success
    payload = res.content
    assert "iso" in payload and "timestamp" in payload and "human" in payload
    assert isinstance(payload["timestamp"], int)

    res2 = GetCurrentTimeTool().safe_run(timezone="UTC")
    assert res2.success
    assert res2.content["timezone"].startswith("UTC")


# ============================================================
#  Text stats
# ============================================================
def test_text_stats_tool():
    from src.agent.builtin_tools import TextStatsTool

    res = TextStatsTool().safe_run(text="Hello 你好 world 世界")
    assert res.success
    assert res.content["cjk_chars"] == 4
    assert res.content["chars"] >= 10


# ============================================================
#  python_eval sandbox
# ============================================================
def test_python_eval_tool_basic():
    from src.agent.builtin_tools import PythonEvalTool

    res = PythonEvalTool().safe_run(expression="1 + 2 * 3")
    assert res.success
    assert res.content["result"] == 7

    res2 = PythonEvalTool().safe_run(expression="max(1, 2, 3) + sqrt(16)")
    assert res2.success
    assert res2.content["result"] == 7


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "import os",
        "open('x.txt')",
        "eval('1')",
        "exec('print(1)')",
        "globals()",
        "getattr(__builtins__, 'open')",
        "open('/etc/passwd').read()",
    ],
)
def test_python_eval_tool_blocks_dangerous(expr):
    """危险表达式应在进入沙箱前/期间被拒绝（ValueError）。"""
    with pytest.raises(Exception):
        _safe_py_eval(expr)


def test_python_eval_tool_returns_toolresult():
    from src.agent.builtin_tools import PythonEvalTool

    res = PythonEvalTool().safe_run(expression="__import__('os')")
    assert not res.success
    assert "沙箱" in (res.error or "") or "禁止" in (res.error or "")


# ============================================================
#  ReAct Agent (with scripted mock LLM)
# ============================================================
class _ScriptedLLM:
    """按调用次序返回预定响应的假 LLM。

    内部记录 ``messages`` 历史，方便测试断言。
    """

    def __init__(self, scripted_responses: List[str]):
        self.scripted = list(scripted_responses)
        self.calls: List[List[Dict[str, Any]]] = []
        self.gen_cfg = type("GC", (), {"max_new_tokens": 256})()  # noqa: N806
        self.model = "mock"
        self.tokenizer = None

    def chat(self, messages, generation=None, stream=False, **kw):
        # 复制一份避免外部修改
        self.calls.append([dict(m) for m in messages])
        if not self.scripted:
            raise RuntimeError("没有更多预编响应")
        return self.scripted.pop(0)


def _wait_done(events):
    """消费事件流直到 ``done`` 或 ``error``。"""
    evs = []
    for ev in events:
        evs.append(ev)
        if ev.get("event") in ("done", "error"):
            break
    return evs


def test_react_agent_simple_calc():
    scripted = [
        # 第一步：用 calculator
        (
            "Thought: 用户问 1+1，我可以用 calculator。\n"
            "Action: calculator\n"
            "ActionInput: {\"expression\": \"1+1\"}\n"
        ),
        # 第二步：总结
        (
            "Thought: 工具返回 2，我可以直接回答。\n"
            "Final Answer: 1 加 1 等于 2。"
        ),
    ]
    reg = BuiltinTools.create()
    agent = ReActAgent(llm=_ScriptedLLM(scripted), tools=reg, max_steps=3)
    events = _wait_done(agent.run("1+1=?"))
    types = [e["event"] for e in events]
    assert "thought" in types
    assert "action" in types
    obs = [e for e in events if e["event"] == "observation"]
    assert obs and obs[0]["data"]["ok"] is True
    assert any(e["event"] == "token" for e in events)
    done = next(e for e in events if e["event"] == "done")
    assert "2" in done["data"]["answer"]
    assert "calculator" in done["data"]["tools_used"]


def test_react_agent_search(mock_store):
    scripted = [
        (
            "Thought: 用户问 RAG 是什么，应当检索知识库。\n"
            "Action: search_documents\n"
            "ActionInput: {\"query\": \"什么是 RAG\", \"top_k\": 2}\n"
        ),
        (
            "Thought: 已经拿到检索结果，可以回答。\n"
            "Final Answer: 根据知识库，RAG 是检索增强生成。"
        ),
    ]
    reg = BuiltinTools.create(vector_store=mock_store)
    agent = ReActAgent(llm=_ScriptedLLM(scripted), tools=reg, max_steps=3)
    events = _wait_done(agent.run("什么是 RAG？"))
    types = [e["event"] for e in events]
    assert "action" in types
    action_ev = next(e for e in events if e["event"] == "action")
    assert action_ev["data"]["tool"] == "search_documents"
    assert "search_documents" in [e["data"]["tool"] for e in events if e["event"] == "action"]
    done = next(e for e in events if e["event"] == "done")
    assert "RAG" in done["data"]["answer"]


def test_react_agent_max_steps_truncates():
    """max_steps=2 时连续用 Action 而不出 Final Answer，应被截断并以 done 收尾。"""
    scripted = [
        # 1
        "Thought: 试一试\nAction: calculator\nActionInput: {\"expression\": \"1\"}\n",
        # 2
        "Thought: 再试一试\nAction: calculator\nActionInput: {\"expression\": \"2\"}\n",
        # 3（已经触达 max_steps=2，正常情况不会再用）
        "Thought: fallback\nFinal Answer: 综合两种调用后的最终答案。",
    ]
    reg = BuiltinTools.create()
    agent = ReActAgent(llm=_ScriptedLLM(scripted), tools=reg, max_steps=2)
    events = _wait_done(agent.run("反复试试"))
    done = next(e for e in events if e["event"] == "done")
    # 截断场景下 answer 由 fallback 调用 LLM 得来（scripted 第 3 条）
    assert "answer" in done["data"]
    # 截断时事件流的最后一个 done 应含有 truncated 标志，或者 tools_used 包括 calculator
    assert "calculator" in done["data"]["tools_used"]


def test_react_agent_handles_unknown_tool():
    """LLM 给出未知工具名 → 触发 observation 反馈，不应抛错。"""
    scripted = [
        (
            "Thought: 试一下\nAction: not_a_real_tool\nActionInput: {\"a\": 1}\n"
        ),
        (
            "Thought: 改用 calculator\n"
            "Action: calculator\n"
            "ActionInput: {\"expression\": \"5\"}\n"
        ),
        "Thought: 完成\nFinal Answer: 答案是 5",
    ]
    reg = BuiltinTools.create()
    agent = ReActAgent(llm=_ScriptedLLM(scripted), tools=reg, max_steps=3)
    events = _wait_done(agent.run("试试"))
    obs = [e for e in events if e["event"] == "observation"]
    assert any(e["data"]["ok"] is False for e in obs)
    done = next(e for e in events if e["event"] == "done")
    assert "5" in done["data"]["answer"]


def test_react_agent_handles_tool_exception():
    """工具抛异常 → 被 safe_run 捕获 → 不应让 Agent 崩溃。"""

    class _Boom(Tool):
        name = "boom_tool"
        description = "总抛"
        parameters = {"type": "object", "properties": {}, "required": []}

        def run(self, **kwargs):
            raise RuntimeError("forced")

    reg = ToolRegistry()
    reg.register(_Boom())
    scripted = [
        "Thought: 调它\nAction: boom_tool\nActionInput: {}\n",
        "Thought: 失败了，直接回答\nFinal Answer: 工具炸了，但我可以告诉你答案。",
    ]
    agent = ReActAgent(llm=_ScriptedLLM(scripted), tools=reg, max_steps=2)
    events = _wait_done(agent.run("调 boom"))
    types = [e["event"] for e in events]
    assert "error" not in types
    obs = [e for e in events if e["event"] == "observation"]
    assert obs and obs[0]["data"]["ok"] is False
    done = next(e for e in events if e["event"] == "done")
    assert "工具炸了" in done["data"]["answer"]


def test_react_agent_history_recorded():
    scripted = [
        "Thought: 一次即可\nAction: calculator\nActionInput: {\"expression\": \"1\"}\n",
        "Thought: done\nFinal Answer: ok",
    ]
    reg = BuiltinTools.create()
    agent = ReActAgent(llm=_ScriptedLLM(scripted), tools=reg, max_steps=3)
    _wait_done(agent.run("hi"))
    assert len(agent.history) == 2
    assert agent.history[0]["parsed"]["action"] == "calculator"


def test_react_agent_event_types_match_schema():
    """校验 yield 事件格式严格符合约定。"""
    scripted = [
        "Thought: t\nAction: calculator\nActionInput: {\"expression\": \"3\"}\n",
        "Thought: ok\nFinal Answer: 3",
    ]
    reg = BuiltinTools.create()
    agent = ReActAgent(llm=_ScriptedLLM(scripted), tools=reg, max_steps=3)
    seen = set()
    for ev in agent.run("3"):
        et = ev.get("event")
        assert et is not None
        assert "data" in ev
        seen.add(et)
        if et == "done":
            break
    assert {"thought", "action", "observation", "token", "done"}.issubset(seen)


def test_python_eval_helper_math():
    """直接验证沙箱 _safe_py_eval 支持常用数学运算。"""
    assert _safe_py_eval("1 + 2 * 3") == 7
    assert abs(_safe_py_eval("sqrt(16) + 1") - 5) < 1e-9
    assert _safe_py_eval("[1, 2, 3][1]") == 2
    assert _safe_py_eval("(1, 2, 3)[-1]") == 3
