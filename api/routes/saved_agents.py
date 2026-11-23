"""保存 Agent 的 CRUD 路由。"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

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
from src.kb_service import KBError, get_accessible_kb
from src.middleware.identity import get_current_user

router = APIRouter()


AGENT_TEMPLATES = (
    {
        "id": "policy-expert",
        "name": "制度问答专家",
        "description": "只依据当前知识库回答制度问题，事实必须带引用，资料不足时明确拒答。",
        "instructions": (
            "你是企业制度问答专家。先检索知识库，再逐条回答；每个关键事实附引用编号。"
            "发现版本冲突时优先采用最新且状态为 active 的正式制度，并说明冲突。"
            "资料不足时直接说明缺少什么资料，禁止凭常识补全。"
        ),
        "allowed_tools": ["search_documents", "get_document_chunks", "list_documents"],
        "max_steps": 5,
    },
    {
        "id": "contract-reviewer",
        "name": "合同审阅助手",
        "description": "按审批权限、付款、验收、违约和隐私风险检查合同，并标出原文依据。",
        "instructions": (
            "你是企业合同审阅助手。围绕金额与审批链、付款条件、交付验收、违约责任、"
            "保密与个人信息条款逐项检查。每项结论必须引用知识库中的规则或合同原文；"
            "区分高风险、待确认和建议改进，不替代法务最终意见。"
        ),
        "allowed_tools": ["search_documents", "get_document_chunks", "calculator", "text_stats"],
        "max_steps": 7,
    },
    {
        "id": "document-comparator",
        "name": "多文档对比助手",
        "description": "对比多份制度、合同或报告，先列差异，再指出生效版本与冲突。",
        "instructions": (
            "你是多文档对比助手。先确认各文档的标题、版本与状态，再用表格比较关键字段；"
            "指出新增、删除、冲突和仍待确认项。任何差异都要给出文档与分块依据，"
            "不得把旧版内容当作当前规则。"
        ),
        "allowed_tools": ["search_documents", "get_document_chunks", "list_documents", "text_stats"],
        "max_steps": 8,
    },
)


@router.get("/api/v1/agent/templates", tags=["Agent"])
def list_agent_templates() -> Dict[str, Any]:
    """内置 Agent 模板；前端填入表单后仍走正常的权限校验与保存流程。"""
    return {"templates": list(AGENT_TEMPLATES), "total": len(AGENT_TEMPLATES)}


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
        get_accessible_kb(db, current_user, body.kb_id, "editor")
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
            get_accessible_kb(db, current_user, body.kb_id, "editor")
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
