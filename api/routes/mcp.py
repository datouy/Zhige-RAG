"""本机 MCP stdio 服务配置与工具发现。"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.db.models import MCPServer, User
from src.mcp_stdio import MCPError, discover_mcp_tools, mcp_tool_name, resolve_mcp_command
from src.middleware.identity import get_current_user
from src.utils import get_logger

logger = get_logger("api.routes.mcp")
router = APIRouter()


class MCPServerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    command: str = Field(..., min_length=1, max_length=512)
    args: List[str] = Field(default_factory=list, max_length=50)
    enabled: bool = True


class MCPServerUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=80)
    command: Optional[str] = Field(None, min_length=1, max_length=512)
    args: Optional[List[str]] = Field(None, max_length=50)
    enabled: Optional[bool] = None


def _server_args(server: MCPServer) -> List[str]:
    try:
        value = json.loads(server.args_json or "[]")
    except (TypeError, ValueError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _server_dict(server: MCPServer) -> Dict[str, Any]:
    return {
        "id": server.id,
        "name": server.name,
        "command": server.command,
        "args": _server_args(server),
        "enabled": bool(server.enabled),
        "tool_count": int(server.tool_count or 0),
        "last_error": server.last_error or "",
        "last_checked_at": server.last_checked_at.isoformat()
        if server.last_checked_at
        else None,
        "created_at": server.created_at.isoformat() if server.created_at else None,
        "updated_at": server.updated_at.isoformat() if server.updated_at else None,
    }


def _owned_server(db: Session, user_id: str, server_id: str) -> MCPServer:
    server = (
        db.query(MCPServer)
        .filter(MCPServer.id == server_id, MCPServer.user_id == user_id)
        .first()
    )
    if server is None:
        raise HTTPException(status_code=404, detail="MCP 服务不存在或无权访问")
    return server


def _normalize_args(values: List[str]) -> List[str]:
    result = []
    for value in values:
        item = str(value)
        if "\x00" in item or "\r" in item or "\n" in item or len(item) > 2000:
            raise HTTPException(status_code=400, detail="MCP 参数包含无效字符或过长")
        result.append(item)
    return result


def _discover(server: MCPServer) -> List[Dict[str, Any]]:
    timeout = float(os.getenv("MCP_TIMEOUT_SECONDS", "20"))
    return discover_mcp_tools(server.command, _server_args(server), timeout=timeout)


def _public_tools(server: MCPServer, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "name": mcp_tool_name(server.id, str(tool.get("name"))),
            "remote_name": tool.get("name"),
            "description": tool.get("description") or "",
            "parameters": tool.get("inputSchema") or {
                "type": "object",
                "properties": {},
            },
        }
        for tool in tools
    ]


@router.get("/api/v1/mcp/servers", tags=["MCP"])
def list_mcp_servers(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(MCPServer)
        .filter(MCPServer.user_id == current_user.id)
        .order_by(MCPServer.created_at.asc())
        .all()
    )
    return {"servers": [_server_dict(server) for server in rows], "total": len(rows)}


@router.post("/api/v1/mcp/servers", status_code=201, tags=["MCP"])
def create_mcp_server(
    body: MCPServerCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="名称不能为空")
    try:
        resolve_mcp_command(body.command)
    except MCPError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    server = MCPServer(
        user_id=current_user.id,
        name=name,
        command=body.command.strip(),
        args_json=json.dumps(_normalize_args(body.args), ensure_ascii=False),
        enabled=body.enabled,
    )
    db.add(server)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="MCP 服务名称已存在") from exc
    db.refresh(server)
    from api.routes.agent import reset_agent_cache

    reset_agent_cache(current_user.id)
    return _server_dict(server)


@router.patch("/api/v1/mcp/servers/{server_id}", tags=["MCP"])
def update_mcp_server(
    server_id: str,
    body: MCPServerUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    server = _owned_server(db, current_user.id, server_id)
    if body.command is not None:
        try:
            resolve_mcp_command(body.command)
        except MCPError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        server.command = body.command.strip()
    if body.args is not None:
        server.args_json = json.dumps(_normalize_args(body.args), ensure_ascii=False)
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="名称不能为空")
        server.name = name
    if body.enabled is not None:
        server.enabled = body.enabled
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="MCP 服务名称已存在") from exc
    db.refresh(server)
    from api.routes.agent import reset_agent_cache

    reset_agent_cache(current_user.id)
    return _server_dict(server)


@router.delete("/api/v1/mcp/servers/{server_id}", tags=["MCP"])
def delete_mcp_server(
    server_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    server = _owned_server(db, current_user.id, server_id)
    db.delete(server)
    db.commit()
    from api.routes.agent import reset_agent_cache

    reset_agent_cache(current_user.id)
    return {"deleted": server_id}


@router.get("/api/v1/mcp/servers/{server_id}/tools", tags=["MCP"])
async def get_mcp_server_tools(
    server_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    server = _owned_server(db, current_user.id, server_id)
    try:
        tools = await run_in_threadpool(_discover, server)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"MCP 连接失败：{exc}") from exc
    return {"server": _server_dict(server), "tools": _public_tools(server, tools)}


@router.post("/api/v1/mcp/servers/{server_id}/test", tags=["MCP"])
async def test_mcp_server(
    server_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    server = _owned_server(db, current_user.id, server_id)
    server.last_checked_at = datetime.utcnow()
    try:
        tools = await run_in_threadpool(_discover, server)
    except Exception as exc:  # noqa: BLE001 - 把服务错误转成可操作的界面状态
        server.tool_count = 0
        server.last_error = str(exc)[:2000]
        db.commit()
        from api.routes.agent import reset_agent_cache

        reset_agent_cache(current_user.id)
        logger.warning("MCP 服务 %s 检查失败: %s", server_id, exc)
        raise HTTPException(status_code=502, detail=f"MCP 连接失败：{exc}") from exc
    server.tool_count = len(tools)
    server.last_error = ""
    db.commit()
    from api.routes.agent import reset_agent_cache

    reset_agent_cache(current_user.id)
    return {
        "server": _server_dict(server),
        "tools": _public_tools(server, tools),
    }


__all__ = ["router"]
