"""ReAct 风格的 Agent 主类。

设计要点
--------

- **不依赖 OpenAI / function-calling 协议**：当前 ``LocalLLM`` 仅暴露标准的
  ``chat(messages, generation, stream)``。因此本实现走"文本 ReAct"路线——
  在 prompt 中描述工具，让 LLM 输出 ``Thought/Action/ActionInput/Observation``
  标记，再用正则解析。优点：对任何 chat 兼容模型都可用；缺点：要靠 LLM 严格
  按照指定格式输出。

- **事件流**：ReActAgent.run(...) 是生成器（也支持 stream 标记），逐 yield
  ``{"event": ..., "data": ...}`` 事件，便于 UI / WebSocket 实时展示。

- **异常吸收**：工具执行失败时只把错误作为 Observation 反馈给 LLM，并不
  直接抛给调用方；只有当 ``max_steps`` 次循环内仍然没有产出 Final Answer，
  才以 ``done`` 事件给出截断收尾。

- **可注入的 LLM**：构造时传入任意 duck-typed ``llm``，仅需具备 ``chat``、
  ``gen_cfg``、可选 ``tokenizer``。这让我们在测试中可以用 fake LLM。

事件格式
========

+-----------+---------------------------+---------------------------+
| event     | data 类型                  | 说明                      |
+===========+===========================+===========================+
| thought   | str                        | LLM 思考文本（Thought）   |
+-----------+---------------------------+---------------------------+
| action    | {"tool": str, "input": dict}| 工具调用（Action+Input）  |
+-----------+---------------------------+---------------------------+
| observation | {"result": str, "ok": bool}| 工具返回结果              |
+-----------+---------------------------+---------------------------+
| token     | str                        | 最终回答的 token          |
+-----------+---------------------------+---------------------------+
| done      | {"steps": int, "tools_used": [str]} | 循环结束             |
+-----------+---------------------------+---------------------------+
| error     | str                        | 异常信息                  |
+-----------+---------------------------+---------------------------+
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from src.utils import get_logger

from .tool import Tool, ToolRegistry, ToolResult

logger = get_logger("agent.react")


# =====================================================================
#  ReAct 提示模板
# =====================================================================
# 日期/时间意图：命中即走本地时钟确定性回答，不进 ReAct 循环
_DATETIME_INTENT_RE = re.compile(r"今天|现在.*(?:时间|日期)|几号|几点|日期|星期")

REACT_SYSTEM_PROMPT = """你是一个中文知识库助手。当前日期：{current_date}。可以使用以下工具：

{tools}

请严格按照以下格式回答（每轮输出一段，直到给出 Final Answer）：
Thought: 你的思考，说明接下来要做什么
Action: 工具名（必须是以下之一：[{tool_names}]，若无需工具则填 None）
ActionInput: 一个 JSON object，例如 {{"query": "你好"}}；若 Action 是 None 则留空 JSON {{}}
Observation: 工具返回结果（自动填充，无需手写）

可以重复多轮 Thought/Action/Observation，直到你有足够信息给出最终回答：
Thought: 我现在可以回答了
Final Answer: 你的最终中文回答

约束：
1. 只使用列出的工具，不允许编造工具名；
2. ActionInput 必须是合法 JSON；
3. 涉及"今天/现在/当前时间/日期"的问题，必须调用 get_current_time 工具获取，禁止凭记忆猜测日期；
4. 最终回答只依据工具返回结果；工具没有提供的信息，明确说明无法得知；
3. 若工具出错，按 Observation 中的错误提示继续推理或改用其它工具；
4. 当不需要工具也能直接回答时，Action 写 None 并在下一行直接给 Final Answer。
"""


@dataclass
class ReActConfig:
    """ReAct Agent 配置容器。"""

    max_steps: int = 5
    temperature: float = 0.2
    max_new_tokens: int = 512
    system_prompt: Optional[str] = None  # 覆盖默认 ReAct 提示模板


# =====================================================================
#  工具描述块构造
# =====================================================================
def _format_tools_block(tools: List[Tool]) -> str:
    """将工具列表格式化为 prompt 中"工具说明"那段文本。"""
    if not tools:
        return "（暂无可用工具）"
    lines: List[str] = []
    for t in tools:
        params = t.parameters or {"type": "object", "properties": {}}
        lines.append(f"- {t.name}: {t.description}")
        lines.append(f"  参数 JSON Schema: {json.dumps(params, ensure_ascii=False)}")
    return "\n".join(lines)


# =====================================================================
#  ReAct 输出解析
# =====================================================================
#  抓取 Thought / Action / ActionInput / Observation / Final Answer
#  说明：Observation 应由程序填充，因此 LLM 输出中即便出现也会被忽略。
_ACTION_RE = re.compile(r"Action\s*:\s*([^\n]+)")
_ACTION_INPUT_RE = re.compile(r"ActionInput\s*:\s*(\{.*?\})", re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought\s*:\s*([^\n]*)")
_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(.+?)(?:\n\s*$|\Z)", re.DOTALL)


@dataclass
class _ParsedStep:
    thought: str = ""
    action: Optional[str] = None
    action_input: Dict[str, Any] = field(default_factory=dict)
    final_answer: Optional[str] = None


def _parse_llm_output(text: str) -> _ParsedStep:
    """从 LLM 输出中抽取结构化字段。

    Args:
        text: LLM 单轮输出文本（可能包含多行）。

    Returns:
        :class:`_ParsedStep`（仅含 LLM 实际输出的字段；缺失则默认）。
    """
    step = _ParsedStep()

    thought_m = _THOUGHT_RE.search(text)
    if thought_m:
        step.thought = thought_m.group(1).strip()

    action_m = _ACTION_RE.search(text)
    if action_m:
        raw = action_m.group(1).strip()
        # 容错：去掉常见的尾部逗号、句号；遇到 None / 无 也视作空
        raw = raw.rstrip(",.; ")
        if raw.lower() in ("none", "null", "", "无", "不需要"):
            step.action = None
        else:
            step.action = raw

    input_m = _ACTION_INPUT_RE.search(text)
    if input_m:
        raw_json = input_m.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                step.action_input = parsed
            else:
                step.action_input = {"value": parsed}
        except Exception:
            # 尝试简单修复：去掉末尾多余逗号
            fixed = re.sub(r",\s*([\]}])", r"\1", raw_json)
            try:
                parsed = json.loads(fixed)
                step.action_input = parsed if isinstance(parsed, dict) else {"value": parsed}
            except Exception:
                step.action_input = {"_raw": raw_json}

    final_m = _FINAL_RE.search(text)
    if final_m:
        step.final_answer = final_m.group(1).strip()

    return step


# =====================================================================
#  ReAct Agent 主类
# =====================================================================
class ReActAgent:
    """ReAct 风格 Agent。

    典型用法::

        agent = ReActAgent(llm=my_llm, tools=BuiltinTools.create(vector_store=my_store))
        for ev in agent.run("2+3=?"):
            print(ev)
    """

    def __init__(
        self,
        llm: Any,
        tools: ToolRegistry,
        max_steps: int = 5,
    ) -> None:
        if llm is None:
            raise ValueError("llm 不能为空")
        if not isinstance(tools, ToolRegistry):
            raise TypeError("tools 必须是 ToolRegistry 实例")
        self.llm = llm
        self.tools = tools
        self.max_steps = max(1, int(max_steps))
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        tool_list = self.tools.list_tools()
        names = self.tools.list_names()
        from datetime import datetime

        return REACT_SYSTEM_PROMPT.format(
            tools=_format_tools_block(tool_list),
            tool_names=", ".join(names),
            current_date=datetime.now().strftime("%Y年%m月%d日"),
        )

    def _ensure_llm_loaded(self) -> None:
        """若 LLM 是 :class:`LocalLLM` 风格懒加载的，触发 ensure。"""
        if hasattr(self.llm, "ensure_llm"):
            try:
                self.llm.ensure_llm()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        if getattr(self.llm, "model", None) is None and hasattr(self.llm, "_load_model"):
            try:
                self.llm._load_model()  # type: ignore[attr-defined]
                self.llm._load_tokenizer()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    def run(self, query: str, stream: bool = True) -> Iterator[Dict[str, Any]]:
        """执行一次 ReAct 循环，yield 事件字典。

        Args:
            query: 用户原始问题。
            stream: 是否使用流式 token 输出最终答案。

        Yields:
            ``{"event": str, "data": Any}``
        """
        self.history = []
        tools_used: List[str] = []
        steps = 0

        if not query or not isinstance(query, str):
            yield {"event": "error", "data": "query 不能为空"}
            return

        # A3 确定性短路：日期/时间类问题直接用本地时钟回答。
        # 1.5B 模型在 ReAct 协议下经常不调用 get_current_time 而凭训练记忆
        # 编造日期（实测答"2023年4月1日"），这类确定性事实不交给模型。
        if _DATETIME_INTENT_RE.search(query):
            from datetime import datetime

            now = datetime.now()
            answer = f"今天是 {now.strftime('%Y年%m月%d日')}，当前时间 {now.strftime('%H:%M:%S')}。"
            steps = 1
            yield {
                "event": "done",
                "data": {"steps": steps, "tools_used": ["local_clock"], "answer": answer},
            }
            return

        self._ensure_llm_loaded()

        tool_names = self.tools.list_names()
        sys_prompt = self._build_system_prompt()
        scratchpad = (
            "用户问题："
            + query.strip()
            + "\n\n"
            "Thought: 我需要先看看有哪些工具，并选择合适的步骤回答这个问题。\n"
            "Action: None\n"
            "ActionInput: {}\n"
        )

        final_answer: Optional[str] = None

        for step_idx in range(self.max_steps):
            steps = step_idx + 1
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": scratchpad},
            ]

            try:
                llm_out = self.llm.chat(messages, stream=False)
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM 调用失败")
                yield {"event": "error", "data": f"LLM 调用失败: {exc}"}
                return

            if not isinstance(llm_out, str):
                llm_out = str(llm_out)

            parsed = _parse_llm_output(llm_out)

            # 1) Thought
            if parsed.thought:
                yield {"event": "thought", "data": parsed.thought}

            self.history.append(
                {
                    "step": steps,
                    "raw_llm": llm_out,
                    "parsed": {
                        "thought": parsed.thought,
                        "action": parsed.action,
                        "action_input": parsed.action_input,
                        "final_answer": parsed.final_answer,
                    },
                }
            )

            # 2) Final Answer 优先返回
            if parsed.final_answer is not None:
                final_answer = parsed.final_answer
                # 流式 yield 每个字符（也支持非字符 token）
                if stream:
                    for piece in self._yield_tokens(final_answer):
                        yield {"event": "token", "data": piece}
                yield {
                    "event": "done",
                    "data": {"steps": steps, "tools_used": tools_used, "answer": final_answer},
                }
                return

            # 3) Action —— 验证是否在白名单
            action = parsed.action
            if action is None:
                # Agent 自评：没想用工具也没产出 Final Answer，提示它产出结论
                scratchpad += (
                    f"\nThought: 我暂时不需要工具，但还没给出最终回答。"
                    f"请直接给出 Final Answer。\n"
                )
                continue

            if action not in tool_names:
                obs_text = f"未知工具 '{action}'，可用工具：{tool_names}"
                yield {"event": "observation", "data": {"result": obs_text, "ok": False}}
                scratchpad += (
                    f"\nThought: {parsed.thought or '选择工具'}\n"
                    f"Action: {action}\nActionInput: {json.dumps(parsed.action_input, ensure_ascii=False)}\n"
                    f"Observation: {obs_text}\n"
                )
                continue

            # 4) 执行工具（try/except 保护；safe_run 已吸收大部分异常）
            tool = self.tools.get(action)
            params = parsed.action_input or {}
            yield {
                "event": "action",
                "data": {"tool": action, "input": params},
            }
            t0 = time.perf_counter()
            try:
                result = tool.safe_run(**params)
            except Exception as exc:  # noqa: BLE001 - 最后一层保险
                logger.exception("工具 %s 执行崩溃", action)
                result = ToolResult(success=False, content=None, error=f"{type(exc).__name__}: {exc}")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if result.success and action not in tools_used:
                tools_used.append(action)

            obs_text = self._format_observation(result)
            yield {
                "event": "observation",
                "data": {"result": obs_text, "ok": result.success, "latency_ms": round(elapsed_ms, 2)},
            }

            scratchpad += (
                f"\nThought: {parsed.thought or '执行工具'}\n"
                f"Action: {action}\nActionInput: {json.dumps(params, ensure_ascii=False)}\n"
                f"Observation: {obs_text}\n"
            )

        # 5) 超过 max_steps 仍未给出 Final Answer → 截断
        try:
            fallback = self._summarize_fallback(scratchpad, tool_names)
        except Exception as exc:  # noqa: BLE001
            logger.warning("生成截断 fallback 失败：%s", exc)
            fallback = "抱歉，Agent 达到最大步数仍未得出最终结论。"

        if stream:
            for piece in self._yield_tokens(fallback):
                yield {"event": "token", "data": piece}
        yield {
            "event": "done",
            "data": {
                "steps": steps,
                "tools_used": tools_used,
                "answer": fallback,
                "truncated": True,
            },
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _yield_tokens(text: str):
        """尽量按"感知上"的 token 切分文本，便于流式 UI 逐字显示。"""
        # 简单按字符切分（中文友好），一次 2-4 个字符
        if not text:
            return
        i = 0
        n = len(text)
        while i < n:
            chunk = text[i : i + 4]
            i += 4
            yield chunk

    @staticmethod
    def _format_observation(result: ToolResult) -> str:
        """把 ToolResult 序列化成 Observation 文本。"""
        if result.success:
            content = result.content
            try:
                if isinstance(content, (dict, list)):
                    return json.dumps(content, ensure_ascii=False)
                return str(content)
            except Exception:  # noqa: BLE001
                return str(content)
        return f"[error] {result.error or '未知错误'}"

    def _summarize_fallback(self, scratchpad: str, tool_names: List[str]) -> str:
        """超过 max_steps 时的兜底：让 LLM 综合已用工具结果给一个简洁回答。"""
        summary_messages = [
            {
                "role": "system",
                "content": (
                    "你是中文知识库助手。下面是一段 ReAct 推理过程的全部历史。"
                    "请综合信息给出对用户问题的简洁回答；如无足够信息请直接说明。"
                ),
            },
            {
                "role": "user",
                "content": scratchpad + "\n\n请仅输出最终中文回答，不要写 Thought/Action 标记。",
            },
        ]
        try:
            text = self.llm.chat(summary_messages, stream=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fallback LLM 调用失败: %s", exc)
            return "Agent 达到最大步数仍未得出最终结论。"
        text = (text or "").strip()
        # 去掉可能残留的标记
        text = re.sub(r"Thought:.*?(?=\n|$)", "", text, flags=re.DOTALL)
        text = re.sub(r"Action:.*?(?=\n|$)", "", text)
        text = re.sub(r"ActionInput:.*?(?=\n|$)", "", text)
        text = re.sub(r"Observation:.*?(?=\n|$)", "", text)
        return text.strip() or "Agent 达到最大步数仍未得出最终结论。"


__all__ = ["ReActAgent", "ReActConfig"]
