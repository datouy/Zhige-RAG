"""Agent 路由：保存配置、REST 调用与 WebSocket 流式调用。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps import get_llm_gate, get_runtime_config, get_runtime_pipeline
from src.agent_profiles import SavedAgentError, saved_agent_runtime
from src.db.database import get_db
from src.db.models import MCPServer, User
from src.kb_service import KBError, get_kb
from src.mcp_stdio import MCPTool, discover_mcp_tools, mcp_tool_name
from src.middleware.identity import get_current_user
from src.utils import get_logger

logger = get_logger("api.routes.agent")
router = APIRouter()


# 缓存键始终包含用户、知识库集合与工具权限；保存 Agent 还包含版本。
_agent_tools_cache: Dict[Any, Any] = {}
_agent_cache: Dict[Any, Any] = {}
_agent_lock = threading.RLock()


def _server_args(server: MCPServer) -> List[str]:
    try:
        value = json.loads(server.args_json or "[]")
    except (TypeError, ValueError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _load_mcp_tools(user_id: str, allowed_tools: List[str]):
    """只发现白名单中实际选中的 MCP 工具；单个服务失败不拖垮 Agent。"""
    from src.agent import ToolRegistry
    from src.db.database import SessionLocal

    registry = ToolRegistry()
    allowed = set(allowed_tools)
    if not any(name.startswith("mcp_") for name in allowed):
        return registry

    with SessionLocal() as db:
        servers = (
            db.query(MCPServer)
            .filter(
                MCPServer.user_id == user_id,
                MCPServer.enabled.is_(True),
                MCPServer.approval_status == "approved",
            )
            .all()
        )
    timeout = float(os.getenv("MCP_TIMEOUT_SECONDS", "20"))
    for server in servers:
        try:
            remote_tools = discover_mcp_tools(
                server.command, _server_args(server), timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001 - 离线连接器不应让整个 Agent 无法启动
            logger.warning("加载 MCP 服务 %s 失败: %s", server.id, exc)
            continue
        for item in remote_tools:
            remote_name = str(item.get("name") or "")
            local_name = mcp_tool_name(server.id, remote_name)
            if local_name not in allowed:
                continue
            registry.register(
                MCPTool(
                    server_id=server.id,
                    server_name=server.name,
                    remote_name=remote_name,
                    description=str(item.get("description") or ""),
                    parameters=item.get("inputSchema") or {},
                    command=server.command,
                    args=_server_args(server),
                    timeout=timeout,
                )
            )
    return registry


def _get_agent_tools(
    user_id: Optional[str] = None,
    kb: Optional[Dict[str, str]] = None,
    allowed_tools: Optional[List[str]] = None,
):
    """构造与用户、知识库和工具白名单严格绑定的工具集。"""
    kb_key = kb.get("collection_name") if kb else None
    allowed_key = None if allowed_tools is None else tuple(sorted(allowed_tools))
    cache_key = (user_id or "__global__", kb_key, allowed_key)
    if cache_key in _agent_tools_cache:
        return _agent_tools_cache[cache_key]

    with _agent_lock:
        if cache_key in _agent_tools_cache:
            return _agent_tools_cache[cache_key]
        from src.agent import BuiltinTools
        from src.factories import TenantAwareFactory

        cfg = get_runtime_config()
        if kb and kb.get("collection_name"):
            vector_store = TenantAwareFactory.get_chroma_store_by_name(
                kb["collection_name"], cfg
            )
        elif user_id:
            vector_store = TenantAwareFactory.get_chroma_store(user_id, cfg)
        else:
            from api.deps import get_vector_store

            vector_store = get_vector_store()

        builtin_names = None
        if allowed_tools is not None:
            builtin_names = [name for name in allowed_tools if not name.startswith("mcp_")]
        tools = BuiltinTools.create(vector_store=vector_store, include=builtin_names)
        if user_id and allowed_tools is not None:
            for tool in _load_mcp_tools(user_id, allowed_tools).list_tools():
                tools.register(tool)
        _agent_tools_cache[cache_key] = tools
        return tools


def _get_agent(
    user_id: Optional[str] = None,
    kb: Optional[Dict[str, str]] = None,
    saved_agent: Optional[Dict[str, Any]] = None,
):
    """按完整配置缓存 ReActAgent；LLM 仍由全局 Pipeline 共享。"""
    if saved_agent:
        kb = saved_agent.get("kb") or kb
        allowed_tools = list(saved_agent.get("allowed_tools") or [])
        instructions = str(saved_agent.get("instructions") or "")
        prompt_hash = hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:16]
        cache_key = (
            user_id or "__global__",
            (kb or {}).get("collection_name"),
            saved_agent.get("id"),
            saved_agent.get("updated_at"),
            prompt_hash,
            tuple(sorted(allowed_tools)),
            int(saved_agent.get("max_steps") or 5),
        )
    else:
        allowed_tools = None
        instructions = ""
        cache_key = (
            user_id or "__global__",
            (kb or {}).get("collection_name"),
            None,
        )

    if cache_key in _agent_cache:
        return _agent_cache[cache_key]
    with _agent_lock:
        if cache_key in _agent_cache:
            return _agent_cache[cache_key]
        from src.agent import ReActAgent

        pipeline = get_runtime_pipeline(user_id=user_id, kb=kb)
        try:
            pipeline.ensure_llm()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "ReAct Agent 依赖生成式 LLM；当前为无 LLM 模式。"
                    "请使用 /api/v1/chat 获取带引用的检索摘录。"
                ),
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("ensure_llm 失败：%s", exc)
        llm = getattr(pipeline, "llm", None)
        if llm is None:
            raise RuntimeError("Pipeline 未加载 LLM，无法启动 Agent")

        cfg = get_runtime_config()
        default_steps = int((cfg.get("agent") or {}).get("max_steps", 5))
        max_steps = int((saved_agent or {}).get("max_steps") or default_steps)
        tools = _get_agent_tools(user_id, kb=kb, allowed_tools=allowed_tools)
        decision_provider = None
        if ((cfg.get("decision", {}) or {}).get("tool_select", {}) or {}).get("enabled"):
            decision_provider = pipeline._get_decision_provider()
        _agent_cache[cache_key] = ReActAgent(
            llm=llm,
            tools=tools,
            max_steps=max_steps,
            decision_provider=decision_provider,
            system_prompt=instructions,
        )
        return _agent_cache[cache_key]


def _with_steps(agent, max_steps: Optional[int]):
    if max_steps is None or max_steps == agent.max_steps:
        return agent
    from src.agent import ReActAgent

    return ReActAgent(
        llm=agent.llm,
        tools=agent.tools,
        max_steps=max_steps,
        decision_provider=getattr(agent, "decision_provider", None),
        system_prompt=getattr(agent, "system_prompt", None),
    )


def reset_agent_cache(user_id: Optional[str] = None) -> None:
    """清空 Agent 与工具缓存；保存 Agent 或 MCP 变更后调用。"""
    with _agent_lock:
        if user_id is None:
            _agent_cache.clear()
            _agent_tools_cache.clear()
            return
        for cache in (_agent_cache, _agent_tools_cache):
            for key in list(cache):
                owner = key[0] if isinstance(key, tuple) else key
                if owner == user_id:
                    cache.pop(key, None)


def _resolve_saved_agent(user_id: str, agent_id: Optional[str]):
    if not agent_id:
        return None
    from src.db.database import SessionLocal

    with SessionLocal() as db:
        try:
            return saved_agent_runtime(db, user_id, agent_id)
        except (SavedAgentError, KBError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/v1/agent/tools", tags=["Agent"])
async def list_agent_tools(
    kb_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """列出当前知识库可用的内置工具。MCP 工具由对应服务单独发现。"""
    try:
        kb = None
        if kb_id:
            from src.kb_service import get_accessible_kb

            row, access_role = get_accessible_kb(db, current_user, kb_id)
            kb = {
                "id": row.id,
                "collection_name": row.collection_name,
                "owner_id": row.user_id,
                "access_role": access_role,
            }
        tools = await run_in_threadpool(_get_agent_tools, current_user.id, kb, None)
        items = [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in tools.list_tools()
        ]
        return {"tools": items, "total": len(items)}
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("获取工具列表失败: %s", exc)
        raise HTTPException(status_code=500, detail="获取工具列表失败") from exc


class AgentChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_steps: Optional[int] = Field(None, ge=1, le=20)
    agent_id: Optional[str] = Field(None, max_length=36)


@router.post("/api/v1/agent/chat", tags=["Agent"])
def agent_chat_sync(
    body: AgentChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """同步 Agent 调用；保存 Agent 的知识范围和工具白名单会覆盖请求范围。"""
    try:
        saved_agent = _resolve_saved_agent(current_user.id, body.agent_id)
        if saved_agent:
            agent = _get_agent(
                current_user.id, kb=saved_agent["kb"], saved_agent=saved_agent
            )
        else:
            agent = _get_agent(current_user.id)
        agent = _with_steps(agent, body.max_steps)

        events: List[Dict[str, Any]] = []
        steps = 0
        tools_used: List[str] = []
        answer = ""
        truncated = False
        for event in agent.run(body.query, stream=False):
            event_type = event.get("event")
            data = event.get("data")
            events.append({"event": event_type, "data": data})
            if event_type == "done":
                steps = int((data or {}).get("steps", 0))
                tools_used = list((data or {}).get("tools_used", []))
                answer = (data or {}).get("answer", "")
                truncated = bool((data or {}).get("truncated", False))
            elif event_type == "error":
                return {"error": str(data), "events": events}
        return {
            "answer": answer,
            "steps": steps,
            "tools_used": tools_used,
            "truncated": truncated,
            "agent_id": body.agent_id,
            "events": events,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent 调用失败: %s", exc)
        raise HTTPException(status_code=500, detail="Agent 调用失败") from exc


@router.websocket("/api/v1/ws/agent")
async def websocket_agent(websocket: WebSocket):
    """Agent WebSocket：支持 request_id 隔离和同连接取消。"""
    from src.db.database import SessionLocal
    from src.team_auth import websocket_user

    with SessionLocal() as db:
        try:
            user_id = websocket_user(websocket, db).id
        except PermissionError as exc:
            await websocket.close(code=1008, reason=str(exc))
            return
    await websocket.accept()

    tasks: set[asyncio.Task] = set()
    cancellations: Dict[str, threading.Event] = {}

    async def send_event(event: Dict[str, Any], request_id: str) -> None:
        payload = dict(event)
        if request_id:
            payload["request_id"] = request_id
        await websocket.send_json(payload)

    async def drain(queue: asyncio.Queue, sentinel: object) -> None:
        while True:
            item = await queue.get()
            if item is sentinel:
                return

    async def serve(data: Dict[str, Any]) -> None:
        request_id = str(data.get("request_id") or uuid.uuid4())
        cancel_event = cancellations[request_id]
        query = str(data.get("query") or "").strip()
        max_steps = data.get("max_steps")
        if max_steps is not None and (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or not 1 <= max_steps <= 20
        ):
            await send_event(
                {"event": "error", "data": "max_steps 必须在 1 到 20 之间"}, request_id
            )
            return
        try:
            saved_agent = _resolve_saved_agent(user_id, data.get("agent_id"))
        except HTTPException as exc:
            await send_event({"event": "error", "data": exc.detail}, request_id)
            return
        if cancel_event.is_set():
            await send_event({"event": "cancelled", "data": None}, request_id)
            return

        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        sentinel = object()
        loop = asyncio.get_running_loop()

        def produce() -> None:
            try:
                if saved_agent:
                    agent = _get_agent(
                        user_id, kb=saved_agent["kb"], saved_agent=saved_agent
                    )
                else:
                    agent = _get_agent(user_id)
                agent = _with_steps(agent, max_steps)
                for event in agent.run(query, stream=True):
                    if cancel_event.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
            except Exception as exc:  # noqa: BLE001
                asyncio.run_coroutine_threadsafe(
                    queue.put({"event": "error", "data": str(exc)}), loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

        threading.Thread(target=produce, daemon=True).start()
        gate = get_llm_gate()
        try:
            if gate is not None:
                await gate.__aenter__()
            while True:
                event = await queue.get()
                if event is sentinel:
                    if cancel_event.is_set():
                        await send_event(
                            {"event": "cancelled", "data": None}, request_id
                        )
                    break
                if cancel_event.is_set():
                    asyncio.create_task(drain(queue, sentinel))
                    await send_event({"event": "cancelled", "data": None}, request_id)
                    return
                await send_event(event, request_id)
        finally:
            if gate is not None:
                await gate.__aexit__(None, None, None)

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("event") == "cancel":
                request_id = str(data.get("request_id") or "")
                event = cancellations.get(request_id)
                if event is not None:
                    event.set()
                continue
            query = str(data.get("query") or "").strip()
            if not query:
                await send_event(
                    {"event": "error", "data": "query is required"},
                    str(data.get("request_id") or ""),
                )
                continue
            request_id = str(data.get("request_id") or uuid.uuid4())
            data["request_id"] = request_id
            cancellations[request_id] = threading.Event()
            task = asyncio.create_task(serve(data))
            tasks.add(task)
            task.add_done_callback(
                lambda done, rid=request_id: (
                    tasks.discard(done), cancellations.pop(rid, None)
                )
            )
    except WebSocketDisconnect:
        logger.info("Agent WebSocket 客户端断开")
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent WebSocket 异常: %s", exc, exc_info=True)
        try:
            await websocket.send_json({"event": "error", "data": "Agent 连接异常"})
        except Exception:  # noqa: BLE001
            pass
    finally:
        for event in cancellations.values():
            event.set()
        for task in tasks:
            task.cancel()


__all__ = ["reset_agent_cache", "router"]
