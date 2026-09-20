"""Add department to users; create feedback/memory/data-layer tables.

Revision ID: 002_feedback_memory
Revises: 001_initial
Create Date: 2026-09-05

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "002_feedback_memory"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """users 表补 department 列；新建四层架构所需的登记/反馈/记忆表。"""
    # 1. users.department（ACL 分组依据；已存在时跳过——SQLite 无 IF NOT EXISTS）
    conn = op.get_bind()
    cols = [row[1] for row in conn.execute(sa.text("PRAGMA table_info(users)")).fetchall()]
    if "department" not in cols:
        op.add_column("users", sa.Column("department", sa.String(100), nullable=False, server_default=""))

    # 2. 数据层：文档登记表
    op.create_table(
        "document_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False, index=True),
        sa.Column("source", sa.String(512), nullable=False),
        sa.Column("title", sa.String(512), nullable=True),
        sa.Column("doc_status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("acl", sa.String(512), nullable=False, server_default="*"),
        sa.Column("department", sa.String(100), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True, index=True),
        sa.Column("chunk_count", sa.Integer, server_default="0"),
        sa.Column("file_ext", sa.String(16), nullable=True),
        sa.Column("updated_at_source", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime),
        sa.Column("updated_at", sa.DateTime),
        sa.UniqueConstraint("user_id", "source", name="uq_docrecord_user_source"),
    )

    # 3. 反馈层
    op.create_table(
        "feedback_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False, index=True),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("answer", sa.Text, nullable=True),
        sa.Column("sources_json", sa.Text, nullable=True),
        sa.Column("rating", sa.String(20), nullable=False, index=True),
        sa.Column("correction", sa.Text, nullable=True),
        sa.Column("question_type", sa.String(50), nullable=True),
        sa.Column("resolved", sa.Boolean, server_default="0", index=True),
        sa.Column("created_at", sa.DateTime, index=True),
    )

    # 4. 长期记忆
    op.create_table(
        "long_term_memories",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("mem_key", sa.String(200), nullable=False),
        sa.Column("mem_value", sa.Text, nullable=False),
        sa.Column("source", sa.String(50), server_default="user"),
        sa.Column("created_at", sa.DateTime),
        sa.Column("updated_at", sa.DateTime),
        sa.UniqueConstraint("user_id", "mem_key", name="uq_ltm_user_key"),
    )
    op.create_index("ix_ltm_user", "long_term_memories", ["user_id"])


def downgrade() -> None:
    """回滚：删表 + 删列（SQLite 3.35+ 支持 DROP COLUMN）。"""
    op.drop_index("ix_ltm_user", table_name="long_term_memories")
    op.drop_table("long_term_memories")
    op.drop_table("feedback_records")
    op.drop_table("document_records")
    conn = op.get_bind()
    cols = [row[1] for row in conn.execute(sa.text("PRAGMA table_info(users)")).fetchall()]
    if "department" in cols:
        op.drop_column("users", "department")
