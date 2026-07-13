"""🤖 Agent (ReAct 工具调用) 页面。

UI 行为：
- 顶部显示当前 Agent 可用工具（名称 + 描述 + 参数）。
- 聊天输入框 → ReAct 循环 → 实时渲染 Thought/Action/Observation/Final Answer。
- 同会话共享 ReActAgent 实例（避免每次重新构造）。
- 兼容"无 LLM"情况：tools/calculator 等内置工具可独立测试。

设计原则：
- 不依赖 RAG 之外的额外组件；只用现有的 ``RAGPipeline`` 提供的 ``llm``。
- 与其它 page_modules 风格一致：``render_page(cfg)`` 单入口。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.utils import get_logger

logger = get_logger("ui.agent")


def _get_agent_for_ui(cfg: dict):
    """在 Streamlit ``@st.cache_resource`` 之外的简易单例：模块级字典缓存。"""
    if "_REACT_AGENT_SINGLETON" not in globals():
        globals()["_REACT_AGENT_SINGLETON"] = None
    if globals()["_REACT_AGENT_SINGLETON"] is not None:
        return globals()["_REACT_AGENT_SINGLETON"]

    from src.agent import BuiltinTools, ReActAgent
    from src.embeddings import EmbeddingModel
    from src.utils import apply_env_overrides, load_config
    from src.vector_store import ChromaStore

    base_cfg = load_config("config/config.yaml")
    base_cfg = apply_env_overrides(base_cfg)
    if cfg:
        base_cfg.update(cfg or {})
    # embedding + vector store
    emb_cfg = base_cfg.get("embedding", {})
    embedding = EmbeddingModel(
        model_name=emb_cfg.get("model_name", "BAAI/bge-small-zh-v1.5"),
        device=emb_cfg.get("device", "auto"),
        batch_size=emb_cfg.get("batch_size", 16),
        max_seq_length=emb_cfg.get("max_seq_length", 512),
        normalize=emb_cfg.get("normalize_embeddings", True),
        cache_dir=emb_cfg.get("cache_dir"),
        local_files_only=emb_cfg.get("local_files_only", False),
    )
    vs_cfg = base_cfg.get("vector_store", {})
    vs = ChromaStore(
        persist_directory=vs_cfg.get("persist_directory", "data/chroma_db"),
        collection_name=vs_cfg.get("collection_name", "chinese_rag_kb"),
        embedding_model=embedding,
        distance_fn=vs_cfg.get("distance_fn", "cosine"),
    )
    tools = BuiltinTools.create(vector_store=vs)

    max_steps = int(base_cfg.get("agent", {}).get("max_steps", 5))

    # LLM：与 RAG 流水线共享，避免内存翻倍
    from ui.app import load_pipeline  # 复用现有 cache_resource

    pipeline = load_pipeline("config/config.yaml")
    try:
        pipeline.ensure_llm()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure_llm 失败：%s", exc)
    llm = getattr(pipeline, "llm", None)
    if llm is None:
        raise RuntimeError("LLM 尚未加载，无法启动 Agent")

    agent = ReActAgent(llm=llm, tools=tools, max_steps=max_steps)
    globals()["_REACT_AGENT_SINGLETON"] = agent
    return agent


def _list_tools_ui(agent) -> List[Dict[str, Any]]:
    """格式化工具列表用于 st.dataframe / st.json 展示。"""
    rows: List[Dict[str, Any]] = []
    for t in agent.tools.list_tools():
        params = t.parameters or {}
        required = params.get("required") or []
        rows.append(
            {
                "name": t.name,
                "description": t.description,
                "required": ", ".join(required) if required else "—",
            }
        )
    return rows


# ===================================================================
#  页面入口
# ===================================================================
def render_page(cfg: dict) -> None:
    """渲染 Agent 页面（Streamlit ``page_modules`` 风格入口）。"""
    import streamlit as st  # type: ignore

    st.header("🤖 Agent（ReAct 工具调用）")
    st.caption(
        "LLM 通过 Thought → Action → Observation 循环决定调用哪些工具来回答问题。"
        "内置工具：search_documents / list_documents / calculator / get_current_time / "
        "text_stats / python_eval 等。"
    )

    # 加载 / 显示 Agent
    try:
        agent = _get_agent_for_ui(cfg)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Agent 初始化失败：{exc}")
        st.info("请确认配置 (embedding / llm / vector_store) 正确，且 LLM 已下载。")
        return

    # ---- 工具清单 ----
    with st.expander("🛠 可用工具", expanded=False):
        tool_rows = _list_tools_ui(agent)
        try:
            import pandas as pd  # type: ignore

            st.dataframe(pd.DataFrame(tool_rows), use_container_width=True, hide_index=True)
        except ImportError:
            st.json(tool_rows)

    # ---- 聊天历史 ----
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []  # [{role, content, steps, tools_used, ...}]

    for turn in st.session_state.agent_history:
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                st.markdown(turn["content"])
            else:
                # Assistant turn：先展示 trace，再给 final answer
                trace = turn.get("trace", [])
                final = turn.get("content", "")
                if trace:
                    with st.expander(
                        f"🧭 推理过程（{len(trace)} 步，工具：{', '.join(turn.get('tools_used', [])) or '无'}）",
                        expanded=False,
                    ):
                        for i, step in enumerate(trace, 1):
                            if step.get("thought"):
                                st.markdown(f"**Thought {i}**: {step['thought']}")
                            if step.get("action"):
                                st.markdown(
                                    f"**Action {i}**: `{step['action']}` → "
                                    f"```json\n{step.get('input', {})}\n```"
                                )
                            if step.get("observation") is not None:
                                ok = step.get("observation_ok", True)
                                icon = "✅" if ok else "⚠️"
                                st.markdown(f"**Observation {i}** {icon}:")
                                st.code(str(step["observation"])[:800])
                if turn.get("truncated"):
                    st.warning("超过最大步数，已被截断。")
                st.markdown(final or "（无回答）")

    # ---- 输入 ----
    cols = st.columns([6, 1])
    user_query = cols[0].chat_input("向 Agent 提问…")
    reset = cols[1].button("🧹 清空", use_container_width=True)

    if reset:
        st.session_state.agent_history = []
        st.rerun()

    if user_query:
        st.session_state.agent_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            trace: List[Dict[str, Any]] = []
            tools_used: List[str] = []
            final = ""
            steps = 0
            truncated = False
            t0 = time.perf_counter()
            try:
                for ev in agent.run(user_query, stream=False):
                    et = ev.get("event")
                    data = ev.get("data")
                    if et == "thought":
                        trace.append({"thought": str(data)})
                    elif et == "action":
                        info = data or {}
                        trace.append(
                            {
                                "action": info.get("tool"),
                                "input": info.get("input", {}),
                            }
                        )
                    elif et == "observation":
                        info = data or {}
                        trace.append(
                            {
                                "observation": info.get("result"),
                                "observation_ok": bool(info.get("ok", True)),
                            }
                        )
                    elif et == "token":
                        final += str(data)
                        placeholder.markdown(final + "▌")
                    elif et == "done":
                        info = data or {}
                        steps = int(info.get("steps", 0))
                        tools_used = list(info.get("tools_used", []))
                        truncated = bool(info.get("truncated", False))
                        if "answer" in info and not final:
                            final = str(info.get("answer") or "")
                    elif et == "error":
                        st.error(str(data))
                        final = "（错误）"
                elapsed = (time.perf_counter() - t0) * 1000
                placeholder.markdown(final or "（无回答）")
                st.caption(f"耗时 {elapsed:.0f} ms · 步数 {steps} · 工具 {len(tools_used)}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Agent 执行失败")
                placeholder.error(f"Agent 执行失败：{exc}")
                final = "（执行失败）"

            st.session_state.agent_history.append(
                {
                    "role": "assistant",
                    "content": final,
                    "trace": trace,
                    "tools_used": tools_used,
                    "truncated": truncated,
                    "steps": steps,
                }
            )
