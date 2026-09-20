"""数据库模型。

包含 User、RefreshToken、UsageLog 模型及订阅层配置。

四层架构补充表：
- DocumentRecord  ：数据层文档登记表（来源/状态/权限/更新时间/内容哈希）
- FeedbackRecord  ：反馈层用户反馈（好评/差评 + 纠错文本，驱动持续迭代）
- LongTermMemory  ：长期记忆（用户级键值事实，与即时/检索上下文区分）
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, Any
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, UniqueConstraint, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# ======================== 订阅层配置 ========================

SUBSCRIPTION_TIERS: Dict[str, Dict[str, Any]] = {
    "free": {
        "max_chunks": 1000,
        "max_queries_per_day": 50,
        "available_models": ["Qwen2.5-0.5B-Instruct"],
        "reranker": False,
        "max_kb": 1,
    },
    "pro": {
        "max_chunks": 50000,
        "max_queries_per_day": 5000,
        "available_models": ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct-GPTQ-Int4"],
        "reranker": True,
        "max_kb": 5,
    },
    "enterprise": {
        "max_chunks": -1,  # 无限制
        "max_queries_per_day": -1,
        "available_models": ["*"],
        "reranker": True,
        "max_kb": -1,
    },
}


def get_tier_limits(tier: str) -> Dict[str, Any]:
    """获取指定订阅层的限制配置。"""
    return SUBSCRIPTION_TIERS.get(tier, SUBSCRIPTION_TIERS["free"])


# ======================== 数据模型 ========================

class User(Base):
    """用户模型。"""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    subscription_tier = Column(String(20), default="free")  # free / pro / enterprise
    max_chunks = Column(Integer, default=1000)
    max_queries_per_day = Column(Integer, default=50)
    # 所属部门/分组：与文档 metadata.acl（逗号分隔）配合实现数据层权限过滤；
    # 空串 = 仅可见 acl="*" 的全员文档
    department = Column(String(100), default="", nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RefreshToken(Base):
    """Refresh Token 模型。"""
    __tablename__ = "refresh_tokens"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    token_hash = Column(String(255), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    revoked = Column(Boolean, default=False)


class UsageLog(Base):
    """API 用量日志。"""
    __tablename__ = "usage_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=True)
    tokens_used = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    chunks_accessed = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# ======================== 数据层：文档登记表 ========================

class DocumentRecord(Base):
    """数据层文档登记表：入库文档的权威台账。

    每条记录对应向量库中的一个 ``source``，是"来源明确"的可审计依据：
    - status     : active / draft / expired（与向量库 metadata.doc_status 一致，
                   检索侧按 status 过滤，draft/expired 不参与召回）
    - acl        : 逗号分隔的可见范围（``*`` 表示全员）；为空即"权限不清"，
                   在 ``data_quality.require_explicit_acl`` 开启时会被准入门拒绝
    - content_hash: 清洗后全文 SHA256，用于文档级去重与变更检测
    """
    __tablename__ = "document_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    source = Column(String(512), nullable=False)  # 与向量库 metadata.source 一致
    title = Column(String(512), nullable=True)
    doc_status = Column(String(20), default="active", nullable=False)  # active/draft/expired
    acl = Column(String(512), default="*", nullable=False)
    department = Column(String(100), nullable=True)
    content_hash = Column(String(64), nullable=True, index=True)
    chunk_count = Column(Integer, default=0)
    file_ext = Column(String(16), nullable=True)
    updated_at_source = Column(DateTime, nullable=True)  # 源文件的最后修改时间
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "source", name="uq_docrecord_user_source"),
    )


# ======================== 反馈层 ========================

class FeedbackRecord(Base):
    """反馈层：用户对一次问答的评价。

    rating 取值 helpful / not_helpful；差评 + correction 文本是
    bad-case 闭环的源头——可一键导出为评估集（scripts/evaluate）驱动迭代。
    """
    __tablename__ = "feedback_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    sources_json = Column(Text, nullable=True)  # 引用来源 JSON 数组
    rating = Column(String(20), nullable=False, index=True)  # helpful / not_helpful
    correction = Column(Text, nullable=True)  # 用户给出的纠正/期望答案
    question_type = Column(String(50), nullable=True)  # 可选分类标签
    resolved = Column(Boolean, default=False, index=True)  # 差评是否已处理
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# ======================== 长期记忆 ========================

class LongTermMemory(Base):
    """长期记忆：跨会话持久的用户级键值事实。

    与即时记忆（本轮会话窗口）和检索记忆（向量召回）严格区分；
    由用户显式指令（"记住……"）或反馈层沉淀写入，检索时不进向量库，
    以结构化方式注入 system 上下文。
    """
    __tablename__ = "long_term_memories"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False)
    mem_key = Column(String(200), nullable=False)
    mem_value = Column(Text, nullable=False)
    source = Column(String(50), default="user")  # user / feedback / system
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "mem_key", name="uq_ltm_user_key"),
        Index("ix_ltm_user", "user_id"),
    )
