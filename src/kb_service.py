"""知识库与会话的业务逻辑。

为什么需要这一层
----------------
此前系统只有"用户 → 底层 Chroma collection"这一层映射：用户看到的是一堆文档，
没有"知识库"这个概念 —— 无法按主题分开管理（人事制度 / 产品手册 / 客户合同），
也没法单独删除或统计某个主题的资料。会话同理，只存在于进程内存里，
重启即失、没有列表、不能重命名。

本模块把两者提升为**用户可见的一等实体**，并负责：

- collection 命名与隔离（保证升级不破坏已有向量数据）；
- 各套餐的知识库数量配额（复用 ``models.SUBSCRIPTION_TIERS`` 的 ``max_kb``）；
- 会话与消息的读写，供 API 与前端使用。
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.db.models import ChatMessage, ChatSession, DocumentRecord, KnowledgeBase, User, get_tier_limits
from src.factories import _sanitize_user_id
from src.utils import get_logger, resolve_path

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
def kb_quota(tier: str) -> int:
    """返回该套餐允许的知识库数量（-1 表示不限）。"""
    return int(get_tier_limits(tier).get("max_kb", 1))


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
    """新建知识库（受套餐 ``max_kb`` 限制）。"""
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

    # 确保默认库存在后再判断配额，避免"第一个库"也被算作超额
    ensure_default_kb(db, user.id, cfg)

    limit = kb_quota(user.subscription_tier or "free")
    current = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user.id).count()
    if limit != -1 and current >= limit:
        raise KBError(
            f"当前套餐（{user.subscription_tier}）最多创建 {limit} 个知识库，"
            f"请升级套餐或删除不再需要的知识库"
        )

    base_collection = (cfg.get("vector_store") or {}).get("collection_name", "chinese_rag_kb")
    kb = KnowledgeBase(
        user_id=user.id,
        name=name,
        description=(description or "").strip(),
        collection_name="",  # 先占位，拿到 id 后再算（命名里含 kb 短 id）
        is_default=False,
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
    """删除 Chroma collection（不存在或失败返回 False，不阻塞主流程）。

    关键：**必须复用进程内已打开的 Chroma client**。Chroma 的 ``PersistentClient``
    对同一 persist 目录在同一个进程里只允许存在一种配置实例，再建一个会直接抛
    ``An instance of Chroma already exists ... with different settings`` ——
    这正是"删除知识库后向量库仍在"的原因。
    """
    if not collection_name:
        return False

    from src.factories import TenantAwareFactory

    # 1) 优先取被删库自己的 store；否则退而取任意一个（它们共享同一 client）
    client = None
    cached = TenantAwareFactory._chroma_stores.pop(collection_name, None)
    if cached is not None:
        client = getattr(cached, "client", None)
    if client is None:
        for store in list(TenantAwareFactory._chroma_stores.values()):
            client = getattr(store, "client", None)
            if client is not None:
                break

    # 2) 进程内确实没有打开的 client 时，才新建（配置需与 vector_store 一致）
    if client is None:
        try:
            import chromadb
            from chromadb.config import Settings

            persist_dir = resolve_path(
                (cfg.get("vector_store") or {}).get(
                    "persist_directory", "data/chroma_db"
                )
            )
            client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False, allow_reset=False),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("无法获取 Chroma client，跳过 collection 删除: %s", exc)
            return False

    try:
        client.delete_collection(collection_name)
        logger.info("已删除 collection：%s", collection_name)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("删除 collection %s 失败（可能本就不存在）: %s", collection_name, exc)
        return False


def refresh_kb_stats(db: Session, user_id: str, kb_id: str) -> KnowledgeBase:
    """重算知识库的文档数/分块数统计。"""
    kb = get_kb(db, user_id, kb_id)
    rows = (
        db.query(DocumentRecord)
        .filter(DocumentRecord.user_id == user_id, DocumentRecord.kb_id == kb_id)
        .all()
    )
    kb.doc_count = len(rows)
    kb.chunk_count = int(sum((r.chunk_count or 0) for r in rows))
    db.commit()
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
        session.last_message_at = datetime.utcnow()
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
    "collection_name_for",
    "create_kb",
    "create_session",
    "delete_kb",
    "delete_session",
    "ensure_default_kb",
    "ensure_session",
    "get_kb",
    "get_session",
    "kb_quota",
    "list_kbs",
    "list_messages",
    "list_sessions",
    "refresh_kb_stats",
    "update_kb",
    "update_session",
]
