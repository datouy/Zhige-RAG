"""保存 Agent 的 CRUD 路由。"""
from __future__ import annotations

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.agent_profiles import (
    SavedAgentError,
    get_saved_agent,
    normalize_tool_names,
    saved_agent_dict,
)
from src.db.database import get_db
from src.db.models import SavedAgent, User
from src.kb_service import KBError, get_kb
from src.middleware.identity import get_current_user

router = APIRouter()


class SavedAgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    kb_id: str = Field(..., min_length=1, max_length=36)
    instructions: str = Field("", max_length=4000)
    allowed_tools: List[str] = Field(default_factory=list, max_length=40)
    max_steps: int = Field(5, ge=1, le=20)


class SavedAgentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=80)
    kb_id: Optional[str] = Field(None, min_length=1, max_length=36)
    instructions: Optional[str] = Field(None, max_length=4000)
    allowed_tools: Optional[List[str]] = Field(None, max_length=40)
    max_steps: Optional[int] = Field(None, ge=1, le=20)


@router.get("/api/v1/agents", tags=["Agent"])
def list_saved_agents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(SavedAgent)
        .filter(SavedAgent.user_id == current_user.id)
        .order_by(SavedAgent.updated_at.desc(), SavedAgent.created_at.desc())
        .all()
    )
    return {"agents": [saved_agent_dict(agent) for agent in rows], "total": len(rows)}


@router.post("/api/v1/agents", status_code=201, tags=["Agent"])
def create_saved_agent(
    body: SavedAgentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="名称不能为空")
        get_kb(db, current_user.id, body.kb_id)
        agent = SavedAgent(
            user_id=current_user.id,
            kb_id=body.kb_id,
            name=name,
            instructions=body.instructions.strip(),
            allowed_tools_json=json.dumps(
                normalize_tool_names(body.allowed_tools), ensure_ascii=False
            ),
            max_steps=body.max_steps,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        from api.routes.agent import reset_agent_cache

        reset_agent_cache(current_user.id)
        return saved_agent_dict(agent)
    except (KBError, SavedAgentError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Agent 名称已存在") from exc


@router.get("/api/v1/agents/{agent_id}", tags=["Agent"])
def get_saved_agent_detail(
    agent_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        return saved_agent_dict(get_saved_agent(db, current_user.id, agent_id))
    except SavedAgentError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/api/v1/agents/{agent_id}", tags=["Agent"])
def update_saved_agent(
    agent_id: str,
    body: SavedAgentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        agent = get_saved_agent(db, current_user.id, agent_id)
        if body.kb_id is not None:
            get_kb(db, current_user.id, body.kb_id)
            agent.kb_id = body.kb_id
        if body.name is not None:
            name = body.name.strip()
            if not name:
                raise HTTPException(status_code=400, detail="名称不能为空")
            agent.name = name
        if body.instructions is not None:
            agent.instructions = body.instructions.strip()
        if body.allowed_tools is not None:
            agent.allowed_tools_json = json.dumps(
                normalize_tool_names(body.allowed_tools), ensure_ascii=False
            )
        if body.max_steps is not None:
            agent.max_steps = body.max_steps
        db.commit()
        db.refresh(agent)
        from api.routes.agent import reset_agent_cache

        reset_agent_cache(current_user.id)
        return saved_agent_dict(agent)
    except SavedAgentError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KBError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Agent 名称已存在") from exc


@router.delete("/api/v1/agents/{agent_id}", tags=["Agent"])
def delete_saved_agent(
    agent_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        agent = get_saved_agent(db, current_user.id, agent_id)
    except SavedAgentError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    db.delete(agent)
    db.commit()
    from api.routes.agent import reset_agent_cache

    reset_agent_cache(current_user.id)
    return {"deleted": agent_id}


__all__ = ["router"]
