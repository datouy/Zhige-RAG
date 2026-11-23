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
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func
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
    get_accessible_kb,
    get_session,
    list_accessible_kbs,
    list_kbs,
    list_messages,
    list_sessions,
    refresh_kb_stats,
    update_kb,
    update_session,
)
from src.middleware.identity import get_current_user
from src.team_auth import require_editor
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
    verification: Optional[Dict[str, Any]] = None
    timings: Optional[Dict[str, Any]] = None


# ======================================================================
#  序列化
# ======================================================================
def _kb_dict(kb, access_role: str = "owner", current_user_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description or "",
        "collection_name": kb.collection_name,
        "doc_count": kb.doc_count or 0,
        "chunk_count": kb.chunk_count or 0,
        "is_default": bool(kb.is_default),
        "owner_id": kb.user_id,
        "is_owned": current_user_id is None or kb.user_id == current_user_id,
        "access_role": access_role,
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


def _markdown_inline(value: Any) -> str:
    """把任意来源字段压成安全、可读的一行 Markdown 文本。"""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.replace("\\", "\\\\").replace("`", "\\`")


def _markdown_quote(value: Any) -> str:
    lines = str(value or "").strip().splitlines() or [""]
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def _export_markdown(session, messages: List[ChatMessage]) -> str:
    """生成个人信息库可长期保存的纯 Markdown 会话档案。"""
    exported_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    title = _markdown_inline(session.title or "对话") or "对话"
    lines = [
        f"# {title}",
        "",
        f"- 创建：{(session.created_at.isoformat(timespec='seconds') if session.created_at else '未知')}",
        f"- 导出：{exported_at}",
        "",
    ]
    if not messages:
        lines.extend(["暂无消息。", ""])
        return "\n".join(lines)

    for message in messages:
        role = "用户" if message.role == "user" else "助手"
        lines.extend([f"## {role}", "", (message.content or "").strip(), ""])
        if message.role != "assistant" or not message.sources_json:
            continue
        try:
            sources = json.loads(message.sources_json)
        except (TypeError, ValueError):
            sources = []
        if not isinstance(sources, list) or not sources:
            continue
        lines.extend(["### 引用", ""])
        for index, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                continue
            source_title = _markdown_inline(
                source.get("title") or source.get("source") or f"资料 {index}"
            )
            source_path = _markdown_inline(source.get("source") or "")
            snippet = source.get("snippet") or source.get("text") or ""
            lines.append(f"{index}. **{source_title}**")
            if source_path:
                lines.extend(["", f"   `{source_path}`"])
            if snippet:
                quoted = _markdown_quote(snippet)
                lines.extend(["", "\n".join(f"   {line}" for line in quoted.splitlines())])
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ======================================================================
#  知识库
# ======================================================================
@router.get("/api/v1/kb", tags=["知识库"])
async def api_list_kb(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """列出本人与共享知识库（保证当前用户至少有一个自己的默认库）。"""
    user: User = current_user
    try:
        ensure_default_kb(db, user.id, get_runtime_config())
        list_kbs(db, user.id)
        items = list_accessible_kbs(db, user)
        return {
            "knowledge_bases": [
                _kb_dict(kb, access, user.id) for kb, access in items
            ],
            "total": len(items),
        }
    except Exception as exc:
        logger.error("列出知识库失败: %s", exc)
        raise HTTPException(status_code=500, detail="获取知识库列表失败") from exc


@router.post("/api/v1/kb", status_code=201, tags=["知识库"])
async def api_create_kb(
    body: KBCreateBody,
    current_user: User = Depends(require_editor),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """新建知识库。"""
    user: User = current_user
    try:
        kb = create_kb(db, user, body.name, body.description, get_runtime_config())
        return _kb_dict(kb, "owner", user.id)
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
        kb, access = get_accessible_kb(db, user, kb_id)
        kb = refresh_kb_stats(db, kb.user_id, kb_id)
        return _kb_dict(kb, access, user.id)
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/api/v1/kb/{kb_id}", tags=["知识库"])
async def api_update_kb(
    kb_id: str,
    body: KBUpdateBody,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """改名 / 改描述（所有者或团队管理员）。"""
    user: User = current_user
    try:
        target, access = get_accessible_kb(db, user, kb_id)
        if access not in {"owner", "admin"}:
            raise KBError("只有知识库所有者或团队管理员可以修改信息")
        kb = update_kb(db, target.user_id, kb_id, body.name, body.description)
        return _kb_dict(kb, access, user.id)
    except KBError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/api/v1/kb/{kb_id}", tags=["知识库"])
async def api_delete_kb(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """删除知识库（所有者或团队管理员；默认库不可删）。"""
    user: User = current_user
    try:
        target, access = get_accessible_kb(db, user, kb_id)
        if access not in {"owner", "admin"}:
            raise KBError("只有知识库所有者或团队管理员可以删除知识库")
        owner_id = target.user_id
        result = delete_kb(db, owner_id, kb_id, get_runtime_config())
        from api.routes.agent import reset_agent_cache

        reset_agent_cache(owner_id)
        return result
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
        if kb_id:
            get_accessible_kb(db, user, kb_id)
        sessions = list_sessions(db, user.id, kb_id=kb_id, limit=limit)
        session_ids = [session.id for session in sessions]
        previews: Dict[str, str] = {}
        if session_ids:
            ranked = (
                db.query(
                    ChatMessage.session_id.label("session_id"),
                    ChatMessage.content.label("content"),
                    func.row_number()
                    .over(
                        partition_by=ChatMessage.session_id,
                        order_by=ChatMessage.created_at.desc(),
                    )
                    .label("row_number"),
                )
                .filter(ChatMessage.session_id.in_(session_ids))
                .subquery()
            )
            previews = {
                row.session_id: (row.content or "")[:80]
                for row in db.query(ranked.c.session_id, ranked.c.content)
                .filter(ranked.c.row_number == 1)
                .all()
            }
        out = []
        for s in sessions:
            out.append(_session_dict(s, previews.get(s.id)))
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
        else:
            get_accessible_kb(db, user, kb_id)
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
                "verification": json.loads(m.verification_json) if m.verification_json else None,
                "timings": json.loads(m.timings_json) if m.timings_json else None,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in rows
        ],
        "total": len(rows),
    }


@router.get("/api/v1/sessions/{session_id}/export.md", tags=["会话"])
async def api_export_session_markdown(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """下载完整会话，正文和引用按消息时间顺序导出为 Markdown。"""
    try:
        session = get_session(db, current_user.id, session_id)
        messages = list_messages(db, current_user.id, session_id, limit=500)
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    filename_base = re.sub(r'[\\/:*?"<>|\r\n]+', "-", session.title or "对话").strip(" .-")
    filename_base = filename_base[:80] or f"conversation-{session.id[:8]}"
    fallback = f"conversation-{session.id[:8]}.md"
    disposition = (
        f'attachment; filename="{fallback}"; '
        f"filename*=UTF-8''{quote(filename_base + '.md', safe='')}"
    )
    return Response(
        content=_export_markdown(session, messages),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": disposition, "Cache-Control": "no-store"},
    )


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
        msg = append_message(
            db,
            user.id,
            session_id,
            body.role,
            body.content,
            body.sources,
            verification=body.verification,
            timings=body.timings,
        )
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"id": msg.id, "role": msg.role, "created_at": msg.created_at.isoformat()}
