"""保存 Agent 的规范化与所有权校验。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.db.models import SavedAgent
from src.kb_service import get_kb


class SavedAgentError(ValueError):
    """Agent 不存在、无权限或配置无效。"""


def normalize_tool_names(values: Optional[List[str]]) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        name = str(value or "").strip()
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", name) or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= 40:
            break
    return names


def agent_allowed_tools(agent: SavedAgent) -> List[str]:
    try:
        values = json.loads(agent.allowed_tools_json or "[]")
    except (TypeError, ValueError):
        return []
    return normalize_tool_names(values if isinstance(values, list) else [])


def saved_agent_dict(agent: SavedAgent) -> Dict[str, Any]:
    return {
        "id": agent.id,
        "name": agent.name,
        "kb_id": agent.kb_id,
        "instructions": agent.instructions or "",
        "allowed_tools": agent_allowed_tools(agent),
        "max_steps": int(agent.max_steps or 5),
        "created_at": agent.created_at.isoformat() if agent.created_at else None,
        "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
    }


def get_saved_agent(db: Session, user_id: str, agent_id: str) -> SavedAgent:
    agent = (
        db.query(SavedAgent)
        .filter(SavedAgent.id == agent_id, SavedAgent.user_id == user_id)
        .first()
    )
    if agent is None:
        raise SavedAgentError("Agent 不存在或无权访问")
    return agent


def saved_agent_runtime(db: Session, user_id: str, agent_id: str) -> Dict[str, Any]:
    """返回经过所有权校验、可安全跨线程传递的 Agent 配置。"""
    agent = get_saved_agent(db, user_id, agent_id)
    kb = get_kb(db, user_id, agent.kb_id)
    return {
        "id": agent.id,
        "name": agent.name,
        "kb": {"id": kb.id, "collection_name": kb.collection_name},
        "instructions": agent.instructions or "",
        "allowed_tools": agent_allowed_tools(agent),
        "max_steps": int(agent.max_steps or 5),
        "updated_at": agent.updated_at.isoformat() if agent.updated_at else "",
    }


__all__ = [
    "SavedAgentError",
    "agent_allowed_tools",
    "get_saved_agent",
    "normalize_tool_names",
    "saved_agent_dict",
    "saved_agent_runtime",
]
