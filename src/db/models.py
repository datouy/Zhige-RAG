"""数据库模型。

默认仍以固定本地用户运行；设置 ``ZHIGE_AUTH_MODE=team`` 后启用可选的
团队账号、角色与知识库共享。两种模式共用同一套数据归属字段，因此从本地
单用户切换到团队模式不会搬库或复制知识。

治理相关表：
- DocumentRecord       ：文档台账与索引生命周期
- FeedbackRecord       ：问答质量反馈
- KnowledgeGap         ：拒答/缺资料形成的待补知识清单
- KnowledgeBaseMember  ：知识库共享成员
- AuthSession          ：不透明会话令牌（只保存哈希）
- Audit/UsageLog       ：操作与用量审计
- MCPServer            ：本机 MCP 工具连接及审批状态
"""
from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, UniqueConstraint, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# ======================== 数据模型 ========================

class User(Base):
    """用户模型。

    默认模式只有固定的本地管理员；团队模式才使用口令与会话认证。
    """
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    # 所属部门/分组：与文档 metadata.acl（逗号分隔）配合实现数据层权限过滤；
    # 空串 = 仅可见 acl="*" 的全员文档
    department = Column(String(100), default="", nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    # team 模式下为 admin/editor/viewer；本地模式固定按 admin 处理。
    role = Column(String(20), default="viewer", nullable=False)
    password_hash = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


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
    resource = Column(String(512), nullable=True)
    details_json = Column(Text, nullable=True)
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
    # 归属的知识库；老数据留空 = 默认库（迁移时回填）
    kb_id = Column(String(36), nullable=True, index=True)
    source = Column(String(512), nullable=False)  # 与向量库 metadata.source 一致
    title = Column(String(512), nullable=True)
    doc_status = Column(String(20), default="active", nullable=False)  # active/draft/expired
    acl = Column(String(512), default="*", nullable=False)
    department = Column(String(100), nullable=True)
    content_hash = Column(String(64), nullable=True, index=True)
    chunk_count = Column(Integer, default=0)
    file_ext = Column(String(16), nullable=True)
    updated_at_source = Column(DateTime, nullable=True)  # 源文件的最后修改时间
    # indexed / reindexing / failed；与业务状态 doc_status 分离，故障重试
    # 不应把一篇 active 文档伪装成 draft。
    index_status = Column(String(20), default="indexed", nullable=False)
    last_indexed_at = Column(DateTime, nullable=True)
    last_index_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "user_id", "kb_id", "source", name="uq_docrecord_user_kb_source"
        ),
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
    citation_issue = Column(Boolean, default=False, nullable=False)
    missing_source = Column(Boolean, default=False, nullable=False)
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


# ======================== 知识库（用户可见的一等实体）========================

class KnowledgeBase(Base):
    """知识库。

    此前用户直接面对底层的 Chroma collection，产品语义不清：一个用户实质
    只有一个"库"，无法按主题（如"人事制度""产品手册"）分开管理，也没法
    单独分享或删除。本表把"知识库"提升为用户可见的一等实体。

    collection 隔离策略（``src/kb_service.py::collection_name_for``）：

    - **默认库**沿用历史命名 ``{user}_{base}`` —— 保证升级不破坏已有向量数据；
    - 新建库使用 ``{user}_{kb短id}_{base}``，与默认库互不干扰。
    """
    __tablename__ = "knowledge_bases"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500), default="", nullable=False)
    collection_name = Column(String(255), nullable=False)
    doc_count = Column(Integer, default=0, nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    # 每个用户的第一个库标记为默认库：未显式指定 kb_id 的写入/检索都落到这里
    is_default = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_kb_user_name"),
    )


# ======================== 会话与消息 ========================

class ChatSession(Base):
    """对话会话（持久化）。

    此前只有进程内的 ``SESSION_MEMORY``：重启即失、没有会话列表、也无法
    重命名或删除。本表把会话变成可管理的实体。
    """
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    kb_id = Column(String(36), nullable=True, index=True)  # 关联的知识库
    title = Column(String(200), default="新会话", nullable=False)
    message_count = Column(Integer, default=0, nullable=False)
    last_message_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatMessage(Base):
    """会话消息。

    保留完整的问答对（含引用来源），用于：会话列表的最后一句话预览、
    重新打开会话时回看历史、以及把某轮对话一键反馈为 bad-case。
    """
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(36), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)
    verification_json = Column(Text, nullable=True)
    timings_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_msg_session_created", "session_id", "created_at"),
    )


# ======================== 可复用 Agent ========================

class SavedAgent(Base):
    """保存一个知识原型：检索范围、行为说明与工具权限。

    不复制知识库或建立新的检索存储，只保存引用和一组允许工具名。
    """

    __tablename__ = "saved_agents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    kb_id = Column(String(36), nullable=False, index=True)
    name = Column(String(80), nullable=False)
    instructions = Column(Text, default="", nullable=False)
    allowed_tools_json = Column(Text, nullable=True)
    max_steps = Column(Integer, default=5, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_saved_agent_user_name"),
    )


# ======================== MCP stdio 连接 ========================

class MCPServer(Base):
    """本机 MCP stdio 服务配置。

    只保存启动命令和参数，不保存进程环境、令牌或工具输出。命令始终以参数
    数组启动，不经过 shell；引用的命令由运行时白名单再次校验。
    """

    __tablename__ = "mcp_servers"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    name = Column(String(80), nullable=False)
    command = Column(String(512), nullable=False)
    args_json = Column(Text, default="[]", nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
    # pending / approved / rejected；加载给 Agent 时只读取 approved+enabled
    approval_status = Column(String(20), default="pending", nullable=False, index=True)
    approved_by = Column(String(36), nullable=True)
    approved_at = Column(DateTime, nullable=True)
    tool_count = Column(Integer, default=0, nullable=False)
    last_error = Column(Text, nullable=True)
    last_checked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_mcp_server_user_name"),
    )


# ======================== 团队会话与共享 ========================

class AuthSession(Base):
    """团队登录会话。数据库中只保存令牌哈希，原文令牌仅下发一次。"""

    __tablename__ = "auth_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    token_hash = Column(String(64), nullable=False, unique=True, index=True)
    user_agent = Column(String(255), nullable=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow)


class KnowledgeBaseMember(Base):
    """知识库共享成员；owner 关系仍保留在 KnowledgeBase.user_id。"""

    __tablename__ = "knowledge_base_members"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    kb_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(36), nullable=False, index=True)
    role = Column(String(20), default="viewer", nullable=False)  # viewer/editor
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("kb_id", "user_id", name="uq_kb_member_user"),
    )


class KnowledgeGap(Base):
    """未解决知识缺口：由拒答、缺资料反馈或管理员手工登记形成。"""

    __tablename__ = "knowledge_gaps"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    kb_id = Column(String(36), nullable=True, index=True)
    session_id = Column(String(36), nullable=True, index=True)
    feedback_id = Column(String(36), nullable=True, index=True)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    reason = Column(String(50), default="refused", nullable=False)
    detail = Column(Text, nullable=True)
    status = Column(String(20), default="open", nullable=False, index=True)
    resolution = Column(Text, nullable=True)
    assigned_to = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_gap_status_updated", "status", "updated_at"),
    )
