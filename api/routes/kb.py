"""知识库与会话路由。

端点：

知识库
- GET    /api/v1/kb                 列表（默认库在最前）
- POST   /api/v1/kb                 新建
- GET    /api/v1/kb/{kb_id}         详情
- PATCH  /api/v1/kb/{kb_id}         改名 / 改描述
- DELETE /api/v1/kb/{kb_id}         删除（连带向量库与文档台账）

会话
- GET    /api/v1/sessions                    列表（最近活跃优先）
- POST   /api/v1/sessions                    新建
- GET    /api/v1/sessions/{session_id}       详情
- PATCH  /api/v1/sessions/{session_id}       重命名
- DELETE /api/v1/sessions/{session_id}       删除
- GET    /api/v1/sessions/{session_id}/messages  历史消息

安全：所有端点都要求认证，且**一律以 token 中的 user_id 作为过滤条件**，
路径里的 id 只用于定位资源，不能用于越权（越权返回 404/403 而不是泄露数据）。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps import get_runtime_config
from src.db.database import get_db
from src.db.models import ChatMessage, User
from src.kb_service import (
    KBError,
    append_message,
    create_kb,
    create_session,
    delete_kb,
    delete_session,
    ensure_default_kb,
    get_kb,
    get_session,
    kb_quota,
    list_kbs,
    list_messages,
    list_sessions,
    refresh_kb_stats,
    update_kb,
    update_session,
)
from src.middleware.auth import get_current_user
from src.utils import get_logger

logger = get_logger("api.routes.kb")

router = APIRouter()


# ======================================================================
#  请求体
# ======================================================================
class KBCreateBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: str = Field("", max_length=500, description="描述")


class KBUpdateBody(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class SessionCreateBody(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    kb_id: Optional[str] = Field(None, description="关联的知识库，缺省为默认库")


class SessionUpdateBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class MessageCreateBody(BaseModel):
    """供前端在流式回答结束后落库助手消息（用户消息一般由 /chat 自动写入）。"""
    role: str = Field("assistant", pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)
    sources: Optional[List[Dict[str, Any]]] = None


# ======================================================================
#  序列化
# ======================================================================
def _kb_dict(kb) -> Dict[str, Any]:
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description or "",
        "collection_name": kb.collection_name,
        "doc_count": kb.doc_count or 0,
        "chunk_count": kb.chunk_count or 0,
        "is_default": bool(kb.is_default),
        "created_at": kb.created_at.isoformat() if kb.created_at else None,
        "updated_at": kb.updated_at.isoformat() if kb.updated_at else None,
    }


def _session_dict(session, preview: Optional[str] = None) -> Dict[str, Any]:
    data = {
        "id": session.id,
        "kb_id": session.kb_id,
        "title": session.title,
        "message_count": session.message_count or 0,
        "last_message_at": session.last_message_at.isoformat()
        if session.last_message_at
        else None,
        "created_at": session.created_at.isoformat() if session.created_at else None,
    }
    if preview is not None:
        data["preview"] = preview
    return data


# ======================================================================
#  知识库
# ======================================================================
@router.get("/api/v1/kb", tags=["知识库"])
async def api_list_kb(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """列出当前用户的知识库（保证至少有一个默认库）。"""
    user: User = current_user
    try:
        ensure_default_kb(db, user.id, get_runtime_config())
        items = list_kbs(db, user.id)
        return {
            "knowledge_bases": [_kb_dict(kb) for kb in items],
            "quota": kb_quota(user.subscription_tier or "free"),
            "total": len(items),
        }
    except Exception as exc:
        logger.error("列出知识库失败: %s", exc)
        raise HTTPException(status_code=500, detail="获取知识库列表失败") from exc


@router.post("/api/v1/kb", status_code=201, tags=["知识库"])
async def api_create_kb(
    body: KBCreateBody,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """新建知识库（受套餐 max_kb 限制）。"""
    user: User = current_user
    try:
        kb = create_kb(db, user, body.name, body.description, get_runtime_config())
        return _kb_dict(kb)
    except KBError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("创建知识库失败: %s", exc)
        raise HTTPException(status_code=500, detail="创建知识库失败") from exc


@router.get("/api/v1/kb/{kb_id}", tags=["知识库"])
async def api_get_kb(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """知识库详情（统计会实时重算）。"""
    user: User = current_user
    try:
        kb = refresh_kb_stats(db, user.id, kb_id)
        return _kb_dict(kb)
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/api/v1/kb/{kb_id}", tags=["知识库"])
async def api_update_kb(
    kb_id: str,
    body: KBUpdateBody,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """改名 / 改描述。"""
    user: User = current_user
    try:
        kb = update_kb(db, user.id, kb_id, body.name, body.description)
        return _kb_dict(kb)
    except KBError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/api/v1/kb/{kb_id}", tags=["知识库"])
async def api_delete_kb(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """删除知识库（连带其向量 collection、文档台账与会话）。默认库不可删。"""
    user: User = current_user
    try:
        return delete_kb(db, user.id, kb_id, get_runtime_config())
    except KBError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("删除知识库失败: %s", exc)
        raise HTTPException(status_code=500, detail="删除知识库失败") from exc


# ======================================================================
#  会话
# ======================================================================
@router.get("/api/v1/sessions", tags=["会话"])
async def api_list_sessions(
    kb_id: Optional[str] = Query(None, description="按知识库过滤"),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """会话列表，最近活跃的在前，并带最后一条消息预览。"""
    user: User = current_user
    try:
        sessions = list_sessions(db, user.id, kb_id=kb_id, limit=limit)
        out = []
        for s in sessions:
            preview = None
            last = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == s.id)
                .order_by(ChatMessage.created_at.desc())
                .first()
            )
            if last is not None:
                preview = (last.content or "")[:80]
            out.append(_session_dict(s, preview))
        return {"sessions": out, "total": len(out)}
    except Exception as exc:
        logger.error("列出会话失败: %s", exc)
        raise HTTPException(status_code=500, detail="获取会话列表失败") from exc


@router.post("/api/v1/sessions", status_code=201, tags=["会话"])
async def api_create_session(
    body: SessionCreateBody,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """新建会话；不传 kb_id 时关联默认知识库。"""
    user: User = current_user
    try:
        kb_id = body.kb_id
        if not kb_id:
            kb_id = ensure_default_kb(db, user.id, get_runtime_config()).id
        session = create_session(db, user.id, body.title, kb_id)
        return _session_dict(session)
    except Exception as exc:
        logger.error("创建会话失败: %s", exc)
        raise HTTPException(status_code=500, detail="创建会话失败") from exc


@router.get("/api/v1/sessions/{session_id}", tags=["会话"])
async def api_get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """会话详情。"""
    try:
        return _session_dict(get_session(db, current_user.id, session_id))
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/api/v1/sessions/{session_id}", tags=["会话"])
async def api_update_session(
    session_id: str,
    body: SessionUpdateBody,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """重命名会话。"""
    try:
        return _session_dict(update_session(db, current_user.id, session_id, body.title))
    except KBError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/api/v1/sessions/{session_id}", tags=["会话"])
async def api_delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """删除会话及其消息。"""
    try:
        return delete_session(db, current_user.id, session_id)
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/v1/sessions/{session_id}/messages", tags=["会话"])
async def api_list_messages(
    session_id: str,
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """会话内的历史消息（时间正序）。"""
    user: User = current_user
    try:
        rows = list_messages(db, user.id, session_id, limit=limit)
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": json.loads(m.sources_json) if m.sources_json else [],
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in rows
        ],
        "total": len(rows),
    }


@router.post("/api/v1/sessions/{session_id}/messages", status_code=201, tags=["会话"])
async def api_append_message(
    session_id: str,
    body: MessageCreateBody,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """追加消息。

    主要用于前端在**流式回答结束后**把助手消息落库
    （用户提问由 ``/api/v1/chat`` 自动记录）。
    """
    user: User = current_user
    try:
        get_session(db, user.id, session_id)  # 越权校验
        msg = append_message(db, user.id, session_id, body.role, body.content, body.sources)
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"id": msg.id, "role": msg.role, "created_at": msg.created_at.isoformat()}
