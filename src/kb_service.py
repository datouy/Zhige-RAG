"""知识库与会话的业务逻辑。

为什么需要这一层
----------------
此前系统只有"用户 → 底层 Chroma collection"这一层映射：用户看到的是一堆文档，
没有"知识库"这个概念 —— 无法按主题分开管理（人事制度 / 产品手册 / 客户合同），
也没法单独删除或统计某个主题的资料。会话同理，只存在于进程内存里，
重启即失、没有列表、不能重命名。

本模块把两者提升为**用户可见的一等实体**，并负责：

- collection 命名与隔离（保证升级不破坏已有向量数据）；
- 会话与消息的读写，供 API 与前端使用。
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.db.models import (
    ChatMessage,
    ChatSession,
    DocumentRecord,
    KnowledgeBase,
    KnowledgeBaseMember,
    User,
)
from src.factories import _sanitize_user_id
from src.security import effective_role
from src.utils import get_logger

logger = get_logger("kb_service")

DEFAULT_KB_NAME = "默认知识库"
_MAX_TITLE_LEN = 200
_MAX_NAME_LEN = 100


class KBError(ValueError):
    """知识库操作失败（消息面向用户，可直接回传）。"""


# ----------------------------------------------------------------------
#  collection 命名
# ----------------------------------------------------------------------
def collection_name_for(
    user_id: str,
    kb_id: Optional[str],
    base_collection: str,
    emb_cfg: Optional[Dict[str, Any]] = None,
    is_default: bool = False,
) -> str:
    """计算知识库对应的 Chroma collection 名。

    命名规则（**刻意保证向后兼容**）：

    - 默认库：``{user}_{base}`` —— 与引入多知识库之前完全一致，
      因此升级后老用户的向量数据仍然命中，不会"知识库突然变空"；
    - 其它库：``{user}_{kb短id}_{base}`` —— 与默认库隔离。

    最后再叠加 Embedding 指纹（非 local 后端时），避免换 embedding 模型后混库。
    """
    from src.embeddings_provider import embedding_collection_name

    safe_uid = _sanitize_user_id(user_id)
    if is_default or not kb_id:
        raw = f"{safe_uid}_{base_collection}"
    else:
        raw = f"{safe_uid}_{str(kb_id)[:8]}_{base_collection}"
    return embedding_collection_name(raw, emb_cfg or {})


# ----------------------------------------------------------------------
#  知识库 CRUD
# ----------------------------------------------------------------------
def list_kbs(db: Session, user_id: str) -> List[KnowledgeBase]:
    """列出用户的知识库（默认库排在最前）。"""
    return (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.user_id == user_id)
        .order_by(KnowledgeBase.is_default.desc(), KnowledgeBase.created_at.asc())
        .all()
    )


def get_kb(db: Session, user_id: str, kb_id: str) -> KnowledgeBase:
    """取指定知识库；不属于该用户时抛 :class:`KBError`（防越权）。"""
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.user_id == user_id)
        .first()
    )
    if kb is None:
        raise KBError("知识库不存在或无权访问")
    return kb


def _member_permission(db: Session, user_id: str, kb_id: str) -> Optional[str]:
    row = (
        db.query(KnowledgeBaseMember)
        .filter(
            KnowledgeBaseMember.kb_id == kb_id,
            KnowledgeBaseMember.user_id == user_id,
        )
        .first()
    )
    return str(row.permission or "read") if row is not None else None


def can_access_kb(
    db: Session,
    user: User,
    kb: KnowledgeBase,
    required_permission: str = "read",
) -> bool:
    """对象级知识库授权；默认拒绝，admin 与所有者拥有完整权限。"""
    required = str(required_permission or "read").lower()
    if required not in {"read", "write"}:
        return False
    if effective_role(user) == "admin" or kb.user_id == user.id:
        return True
    member = _member_permission(db, user.id, kb.id)
    if member == "write":
        return True
    if member == "read":
        return required == "read"
    visibility = str(kb.visibility or "private").lower()
    if required == "read" and visibility == "company":
        return True
    dept = str(getattr(user, "department", "") or "").strip()
    if (
        required == "read"
        and visibility == "department"
        and dept
        and dept == str(kb.department or "").strip()
    ):
        return True
    return False


def get_accessible_kb(
    db: Session,
    user: User,
    kb_id: str,
    required_permission: str = "read",
) -> KnowledgeBase:
    """按对象 ACL 取知识库；不存在与越权统一返回同一错误。"""
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None or not can_access_kb(db, user, kb, required_permission):
        raise KBError("知识库不存在或无权访问")
    return kb


def list_accessible_kbs(db: Session, user: User) -> List[KnowledgeBase]:
    """列出所有者、显式成员、本部门共享和全员共享的知识库。"""
    if effective_role(user) == "admin":
        return (
            db.query(KnowledgeBase)
            .order_by(KnowledgeBase.is_default.desc(), KnowledgeBase.created_at.asc())
            .all()
        )
    member_ids = [
        row[0]
        for row in db.query(KnowledgeBaseMember.kb_id)
        .filter(KnowledgeBaseMember.user_id == user.id)
        .all()
    ]
    conditions = [
        KnowledgeBase.user_id == user.id,
        KnowledgeBase.visibility == "company",
    ]
    if member_ids:
        conditions.append(KnowledgeBase.id.in_(member_ids))
    dept = str(getattr(user, "department", "") or "").strip()
    if dept:
        conditions.append(
            (KnowledgeBase.visibility == "department")
            & (KnowledgeBase.department == dept)
        )
    return (
        db.query(KnowledgeBase)
        .filter(or_(*conditions))
        .order_by(KnowledgeBase.is_default.desc(), KnowledgeBase.created_at.asc())
        .all()
    )


def set_kb_sharing(
    db: Session,
    user: User,
    kb_id: str,
    *,
    visibility: Optional[str] = None,
    department: Optional[str] = None,
    members: Optional[List[Dict[str, str]]] = None,
) -> KnowledgeBase:
    """所有者/admin 调整知识库共享范围与显式成员。"""
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None or not can_access_kb(db, user, kb, "write"):
        raise KBError("知识库不存在或无权访问")
    if effective_role(user) != "admin" and kb.user_id != user.id:
        raise KBError("只有知识库所有者或系统管理员可以调整共享权限")

    if visibility is not None:
        normalized = str(visibility).strip().lower()
        if normalized not in {"private", "department", "company"}:
            raise KBError("visibility 仅支持 private / department / company")
        kb.visibility = normalized
    if department is not None:
        kb.department = str(department).strip()[:100]
    if kb.visibility == "department" and not (kb.department or user.department):
        raise KBError("部门共享必须设置部门")
    if kb.visibility == "department" and not kb.department:
        kb.department = str(user.department or "").strip()

    if members is not None:
        db.query(KnowledgeBaseMember).filter(KnowledgeBaseMember.kb_id == kb.id).delete(
            synchronize_session=False
        )
        seen: set[str] = set()
        for item in members:
            target_id = str((item or {}).get("user_id") or "").strip()
            permission = str((item or {}).get("permission") or "read").strip().lower()
            if not target_id or target_id in seen:
                continue
            if target_id == kb.user_id:
                continue
            if permission not in {"read", "write"}:
                raise KBError("成员权限仅支持 read / write")
            target = db.get(User, target_id)
            if target is None or not bool(target.is_active):
                raise KBError(f"成员不存在或已停用：{target_id}")
            db.add(
                KnowledgeBaseMember(
                    kb_id=kb.id,
                    user_id=target.id,
                    permission=permission,
                    created_by=user.id,
                )
            )
            seen.add(target_id)
    db.commit()
    db.refresh(kb)
    return kb


def list_kb_members(db: Session, kb_id: str) -> List[KnowledgeBaseMember]:
    return (
        db.query(KnowledgeBaseMember)
        .filter(KnowledgeBaseMember.kb_id == kb_id)
        .order_by(KnowledgeBaseMember.created_at.asc())
        .all()
    )


def ensure_default_kb(
    db: Session, user_id: str, cfg: Dict[str, Any]
) -> KnowledgeBase:
    """确保用户至少有一个默认知识库，返回它。

    首次调用时创建。默认库的存在保证了两件事：

    1. 不传 ``kb_id`` 的老客户端/老数据仍能正常工作；
    2. 用户在界面上"总有一个可以往里传东西的库"。
    """
    existing = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.user_id == user_id, KnowledgeBase.is_default.is_(True))
        .first()
    )
    if existing is not None:
        return existing

    base_collection = (cfg.get("vector_store") or {}).get("collection_name", "chinese_rag_kb")
    kb = KnowledgeBase(
        user_id=user_id,
        name=DEFAULT_KB_NAME,
        description="系统自动创建，未指定知识库时的默认归属",
        collection_name=collection_name_for(
            user_id, None, base_collection, cfg.get("embedding"), is_default=True
        ),
        is_default=True,
        visibility="private",
        department="",
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)

    # 老数据回填：把没有 kb_id 的文档记录挂到默认库上，避免它们"无家可归"
    try:
        updated = (
            db.query(DocumentRecord)
            .filter(DocumentRecord.user_id == user_id, DocumentRecord.kb_id.is_(None))
            .update({DocumentRecord.kb_id: kb.id}, synchronize_session=False)
        )
        if updated:
            db.commit()
            logger.info("已将 %d 条历史文档记录归入默认知识库", updated)
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.warning("历史文档回填默认知识库失败（不影响使用）: %s", exc)

    logger.info("为用户创建默认知识库：%s", kb.id)
    return kb


def create_kb(
    db: Session, user: User, name: str, description: str, cfg: Dict[str, Any]
) -> KnowledgeBase:
    """新建知识库。"""
    name = (name or "").strip()
    if not name:
        raise KBError("知识库名称不能为空")
    if len(name) > _MAX_NAME_LEN:
        raise KBError(f"知识库名称过长（最多 {_MAX_NAME_LEN} 字）")
    if len(description or "") > 500:
        raise KBError("描述过长（最多 500 字）")

    if (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.user_id == user.id, KnowledgeBase.name == name)
        .first()
        is not None
    ):
        raise KBError(f"已存在同名知识库：{name}")

    # 确保默认库存在后再创建新库
    ensure_default_kb(db, user.id, cfg)

    base_collection = (cfg.get("vector_store") or {}).get("collection_name", "chinese_rag_kb")
    kb = KnowledgeBase(
        user_id=user.id,
        name=name,
        description=(description or "").strip(),
        collection_name="",  # 先占位，拿到 id 后再算（命名里含 kb 短 id）
        is_default=False,
        visibility="private",
        department=str(user.department or ""),
    )
    db.add(kb)
    db.flush()  # 需要 id 才能算 collection 名
    kb.collection_name = collection_name_for(
        user.id, kb.id, base_collection, cfg.get("embedding"), is_default=False
    )
    db.commit()
    db.refresh(kb)
    logger.info("用户 %s 创建知识库 %s（%s）", user.id[:8], kb.name, kb.collection_name)
    return kb


def update_kb(
    db: Session, user_id: str, kb_id: str, name: Optional[str], description: Optional[str]
) -> KnowledgeBase:
    """改名 / 改描述。"""
    kb = get_kb(db, user_id, kb_id)
    if name is not None:
        name = name.strip()
        if not name:
            raise KBError("知识库名称不能为空")
        if len(name) > _MAX_NAME_LEN:
            raise KBError(f"知识库名称过长（最多 {_MAX_NAME_LEN} 字）")
        dup = (
            db.query(KnowledgeBase)
            .filter(
                KnowledgeBase.user_id == user_id,
                KnowledgeBase.name == name,
                KnowledgeBase.id != kb_id,
            )
            .first()
        )
        if dup is not None:
            raise KBError(f"已存在同名知识库：{name}")
        kb.name = name
    if description is not None:
        if len(description) > 500:
            raise KBError("描述过长（最多 500 字）")
        kb.description = description.strip()
    db.commit()
    db.refresh(kb)
    return kb


def delete_kb(db: Session, user_id: str, kb_id: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """删除知识库：连带删除其向量 collection、文档台账与会话。

    默认知识库**不允许删除** —— 它是不指定 kb_id 时的兜底归属，
    删掉会让老客户端与新数据同时失去落脚点。
    """
    kb = get_kb(db, user_id, kb_id)
    if kb.is_default:
        raise KBError("默认知识库不能删除（可在其中清空文档）")

    removed_docs = (
        db.query(DocumentRecord)
        .filter(DocumentRecord.user_id == user_id, DocumentRecord.kb_id == kb_id)
        .delete(synchronize_session=False)
    )
    session_ids = [
        row[0]
        for row in db.query(ChatSession.id)
        .filter(ChatSession.user_id == user_id, ChatSession.kb_id == kb_id)
        .all()
    ]
    if session_ids:
        db.query(ChatMessage).filter(ChatMessage.session_id.in_(session_ids)).delete(
            synchronize_session=False
        )
        db.query(ChatSession).filter(ChatSession.id.in_(session_ids)).delete(
            synchronize_session=False
        )
    db.delete(kb)
    db.commit()

    collection_removed = _drop_collection(kb.collection_name, cfg)
    logger.info(
        "删除知识库 %s：文档 %d 条、会话 %d 个、collection=%s(%s)",
        kb.name,
        removed_docs,
        len(session_ids),
        kb.collection_name,
        "已删除" if collection_removed else "本就不存在",
    )
    return {
        "deleted": kb.name,
        "documents": int(removed_docs),
        "sessions": len(session_ids),
        "collection_dropped": collection_removed,
    }


def _drop_collection(collection_name: str, cfg: Dict[str, Any]) -> bool:
    """删除当前向量后端的 collection/table；失败返回 False。"""
    if not collection_name:
        return False

    from src.factories import TenantAwareFactory

    try:
        store = TenantAwareFactory._chroma_stores.pop(collection_name, None)
        if store is None:
            store = TenantAwareFactory.get_chroma_store_by_name(collection_name, cfg)
        delete = getattr(store, "delete_collection", None)
        if not callable(delete):
            raise RuntimeError("当前向量后端不支持删除 collection")
        delete()
        logger.info("已删除 collection：%s", collection_name)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("删除 collection %s 失败（可能本就不存在）: %s", collection_name, exc)
        return False


def refresh_kb_stats(
    db: Session, user_id: str, kb_id: str, *, commit: bool = True
) -> KnowledgeBase:
    """重算知识库的文档数/分块数统计。

    ``commit=False`` 供跨向量/数据库补偿事务使用，由调用方统一提交。
    """
    kb = get_kb(db, user_id, kb_id)
    rows = (
        db.query(DocumentRecord)
        .filter(DocumentRecord.user_id == user_id, DocumentRecord.kb_id == kb_id)
        .all()
    )
    kb.doc_count = len(rows)
    kb.chunk_count = int(sum((r.chunk_count or 0) for r in rows))
    if commit:
        db.commit()
    else:
        db.flush()
    db.refresh(kb)
    return kb


# ----------------------------------------------------------------------
#  会话 CRUD
# ----------------------------------------------------------------------
def list_sessions(
    db: Session, user_id: str, kb_id: Optional[str] = None, limit: int = 50
) -> List[ChatSession]:
    """列出会话（最近活跃的在前）。"""
    query = db.query(ChatSession).filter(ChatSession.user_id == user_id)
    if kb_id:
        query = query.filter(ChatSession.kb_id == kb_id)
    return (
        query.order_by(
            ChatSession.last_message_at.is_(None),
            ChatSession.last_message_at.desc(),
            ChatSession.created_at.desc(),
        )
        .limit(max(1, min(limit, 200)))
        .all()
    )


def create_session(
    db: Session, user_id: str, title: Optional[str] = None, kb_id: Optional[str] = None
) -> ChatSession:
    """新建会话。"""
    session = ChatSession(
        user_id=user_id,
        kb_id=kb_id,
        title=(title or "新会话").strip()[:_MAX_TITLE_LEN] or "新会话",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: Session, user_id: str, session_id: str) -> ChatSession:
    """取会话；越权或不存在时抛错。"""
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )
    if session is None:
        raise KBError("会话不存在或无权访问")
    return session


def ensure_session(
    db: Session, user_id: str, session_id: Optional[str], kb_id: Optional[str] = None
) -> ChatSession:
    """按 id 取会话，不存在则创建（便于前端"用固定 id 开聊"）。"""
    if session_id:
        found = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .first()
        )
        if found is not None:
            return found
        return create_session(db, user_id, kb_id=kb_id)
    return create_session(db, user_id, kb_id=kb_id)


def update_session(
    db: Session, user_id: str, session_id: str, title: Optional[str]
) -> ChatSession:
    """重命名会话。"""
    session = get_session(db, user_id, session_id)
    if title is not None:
        title = title.strip()
        if not title:
            raise KBError("会话标题不能为空")
        session.title = title[:_MAX_TITLE_LEN]
    db.commit()
    db.refresh(session)
    return session


def delete_session(db: Session, user_id: str, session_id: str) -> Dict[str, Any]:
    """删除会话及其消息。"""
    session = get_session(db, user_id, session_id)
    removed = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id, ChatMessage.user_id == user_id)
        .delete(synchronize_session=False)
    )
    db.delete(session)
    db.commit()
    return {"deleted": session_id, "messages": int(removed)}


def append_message(
    db: Session,
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Any]] = None,
) -> ChatMessage:
    """追加一条消息，并更新会话的统计与首条标题。"""
    import json

    message = ChatMessage(
        session_id=session_id,
        user_id=user_id,
        role=role,
        content=content or "",
        sources_json=json.dumps(sources, ensure_ascii=False) if sources else None,
    )
    db.add(message)

    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )
    if session is not None:
        session.message_count = (session.message_count or 0) + 1
        session.last_message_at = datetime.now(timezone.utc).replace(tzinfo=None)
        # 首条用户消息自动成为标题（用户体验：无需手动命名）
        if role == "user" and (session.title in ("", "新会话")):
            session.title = _auto_title(content)
    db.commit()
    db.refresh(message)
    return message


def _auto_title(text: str) -> str:
    """从首条提问生成会话标题。"""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    return (cleaned[:_MAX_TITLE_LEN] if cleaned else "新会话") or "新会话"


def list_messages(
    db: Session, user_id: str, session_id: str, limit: int = 100
) -> List[ChatMessage]:
    """取会话内的消息（时间正序）。"""
    get_session(db, user_id, session_id)  # 越权校验
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id, ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    if limit and len(rows) > limit:
        rows = rows[-limit:]
    return rows


__all__ = [
    "DEFAULT_KB_NAME",
    "KBError",
    "append_message",
    "can_access_kb",
    "collection_name_for",
    "create_kb",
    "create_session",
    "delete_kb",
    "delete_session",
    "ensure_default_kb",
    "ensure_session",
    "get_accessible_kb",
    "get_kb",
    "get_session",
    "list_kbs",
    "list_accessible_kbs",
    "list_kb_members",
    "list_messages",
    "list_sessions",
    "refresh_kb_stats",
    "set_kb_sharing",
    "update_kb",
    "update_session",
]
