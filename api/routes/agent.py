"""Agent 路由。

端点：
- GET  /api/v1/agent/tools
- POST /api/v1/agent/chat
- WS   /api/v1/ws/agent

P1.4：``get_agent`` 不再每次请求都 ``RAGPipeline.from_config(...)``，复用
``api.deps.get_runtime_pipeline`` 单例，避免 LLM 重复加载。

安全（P5 审计修复）：agent / 工具集必须**按 user_id 缓存**。之前的模块级
全局单例会永久绑定第一个调用者的租户向量库——之后任何用户调用 Agent，
检索到的都是第一个用户的私有知识库（跨租户数据泄露）。
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api.deps import get_llm_executor, get_runtime_config, get_runtime_pipeline
from src.db.models import User
from src.middleware.auth import get_current_user
from src.utils import get_logger

logger = get_logger("api.routes.agent")

router = APIRouter()


# user_id -> 该用户的工具集 / Agent 实例（工具绑定租户向量库，必须按用户隔离）
_agent_tools_by_user: Dict[str, Any] = {}
_agent_by_user: Dict[str, Any] = {}
# RLock：_get_agent 持锁期间会调用 _get_agent_tools（内部再次加锁），
# threading.Lock 不可重入，同线程二次加锁会永久死锁。
_agent_lock = threading.RLock()


def _get_agent_tools(user_id: Optional[str] = None):
    """获取（或创建）指定用户的 Agent 内置工具集。"""
    cache_key = user_id or "__global__"
    if cache_key not in _agent_tools_by_user:
        with _agent_lock:
            if cache_key not in _agent_tools_by_user:
                from src.agent import BuiltinTools
                from src.factories import TenantAwareFactory

                cfg = get_runtime_config()
                if user_id:
                    vs = TenantAwareFactory.get_chroma_store(user_id, cfg)
                else:
                    from api.deps import get_vector_store

                    vs = get_vector_store()
                _agent_tools_by_user[cache_key] = BuiltinTools.create(vector_store=vs)
    return _agent_tools_by_user[cache_key]


def _get_agent(user_id: Optional[str] = None):
    """获取（或创建）指定用户的 ReActAgent 实例（LLM 为全局共享单例）。"""
    global _agent_by_user
    cache_key = user_id or "__global__"
    if cache_key not in _agent_by_user:
        with _agent_lock:
            if cache_key not in _agent_by_user:
                from src.agent import ReActAgent

                pipeline = get_runtime_pipeline(user_id=user_id)
                try:
                    pipeline.ensure_llm()
                except Exception as exc:
                    logger.warning("ensure_llm 失败：%s", exc)
                llm = getattr(pipeline, "llm", None)
                if llm is None:
                    raise RuntimeError("Pipeline 未加载 LLM，无法启动 Agent")
                cfg = get_runtime_config()
                max_steps = int(cfg.get("agent", {}).get("max_steps", 5))
                tools = _get_agent_tools(user_id)
                _agent_by_user[cache_key] = ReActAgent(llm=llm, tools=tools, max_steps=max_steps)
    return _agent_by_user[cache_key]


def reset_agent_cache(user_id: Optional[str] = None) -> None:
    """清空 agent 缓存（用户注销 / 测试用）。user_id 为 None 时全量清空。"""
    with _agent_lock:
        if user_id is None:
            _agent_by_user.clear()
            _agent_tools_by_user.clear()
        else:
            _agent_by_user.pop(user_id, None)
            _agent_tools_by_user.pop(user_id, None)


@router.get("/api/v1/agent/tools", tags=["Agent"])
async def list_agent_tools(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """列出 Agent 可用工具（名称 + 描述 + 参数 schema）。"""
    user: User = current_user
    try:
        tools = await run_in_threadpool(_get_agent_tools, user.id)
        items = []
        for t in tools.list_tools():
            items.append(
                {"name": t.name, "description": t.description, "parameters": t.parameters}
            )
        return {"tools": items, "total": len(items)}
    except Exception as exc:
        logger.error("获取工具列表失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class AgentChatRequest(BaseModel):
    """POST /api/v1/agent/chat 请求体。"""

    query: str = Field(..., min_length=1, description="用户问题")
    max_steps: Optional[int] = Field(None, ge=1, le=20, description="覆盖默认 max_steps")


@router.post("/api/v1/agent/chat", tags=["Agent"])
def agent_chat_sync(
    body: AgentChatRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """同步 Agent 调用：收集完整事件流后一次性返回。

    说明：必须是 ``def`` ——Agent 循环内含多次阻塞 LLM 推理，async 端点
    里直接执行会卡死事件循环；FastAPI 自动把 ``def`` 端点调度到线程池。
    """
    user: User = current_user
    try:
        agent = _get_agent(user.id)
        if body.max_steps is not None and body.max_steps != agent.max_steps:
            from src.agent.react_agent import ReActAgent

            agent = ReActAgent(llm=agent.llm, tools=agent.tools, max_steps=body.max_steps)
        events: List[Dict[str, Any]] = []
        steps = 0
        tools_used: List[str] = []
        answer = ""
        truncated = False
        for ev in agent.run(body.query, stream=False):
            et = ev.get("event")
            data = ev.get("data")
            events.append({"event": et, "data": data})
            if et == "done":
                steps = int((data or {}).get("steps", 0))
                tools_used = list((data or {}).get("tools_used", []))
                answer = (data or {}).get("answer", "")
                truncated = bool((data or {}).get("truncated", False))
            elif et == "error":
                return {"error": str(data), "events": events}
        return {
            "answer": answer,
            "steps": steps,
            "tools_used": tools_used,
            "truncated": truncated,
            "events": events,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Agent 调用失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.websocket("/api/v1/ws/agent")
async def websocket_agent(websocket: WebSocket, token: Optional[str] = None):
    """WebSocket 流式 Agent 调用。必须携带有效 JWT（``?token=``）。"""
    from src.auth.jwt_handler import decode_token

    payload = decode_token(token) if token else None
    if not payload or not payload.get("sub"):
        await websocket.close(code=4401)  # 未认证
        return
    user_id = payload["sub"]

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            max_steps = data.get("max_steps")
            if not query:
                await websocket.send_json({"event": "error", "data": "query is required"})
                continue
            executor = get_llm_executor()

            async def _run():
                # Agent 循环内含多次阻塞 LLM 推理，放到独立线程生产事件，
                # 经 asyncio.Queue 转交事件循环逐条发送——实现真正的流式。
                # 之前的"线程池里收集全部事件再发送"让用户在整个推理完成前
                # 收不到任何 token，长任务期间还可能被空闲超时断连。
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue(maxsize=256)
                _SENTINEL = object()

                def _produce() -> None:
                    try:
                        agent0 = _get_agent(user_id)
                        if (
                            isinstance(max_steps, int)
                            and max_steps > 0
                            and max_steps != agent0.max_steps
                        ):
                            from src.agent.react_agent import ReActAgent

                            a = ReActAgent(llm=agent0.llm, tools=agent0.tools, max_steps=max_steps)
                        else:
                            a = agent0
                        for ev in a.run(query, stream=True):
                            asyncio.run_coroutine_threadsafe(queue.put(ev), loop).result()
                    except Exception as exc:  # noqa: BLE001
                        asyncio.run_coroutine_threadsafe(
                            queue.put({"event": "error", "data": str(exc)}), loop
                        ).result()
                    finally:
                        asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop).result()

                producer = threading.Thread(target=_produce, daemon=True)
                producer.start()
                while True:
                    ev = await queue.get()
                    if ev is _SENTINEL:
                        break
                    await websocket.send_json(ev)

            try:
                sem = getattr(executor, "semaphore", None)
                if sem is not None:
                    async with sem:
                        await _run()
                else:
                    await _run()
            except Exception as exc:
                logger.error("Agent WebSocket 处理失败: %s", exc)
                await websocket.send_json({"event": "error", "data": str(exc)})
    except WebSocketDisconnect:
        logger.info("Agent WebSocket 客户端断开")
    except Exception as exc:
        logger.error("Agent WebSocket 异常: %s", exc)