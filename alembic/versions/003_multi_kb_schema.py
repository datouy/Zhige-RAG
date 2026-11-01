"""Add multi-knowledge-base and persistent chat schema.

Revision ID: 003_multi_kb_schema
Revises: 002_feedback_memory
Create Date: 2026-10-19
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "003_multi_kb_schema"
down_revision: Union[str, None] = "002_feedback_memory"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_names(table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    columns = _column_names("document_records")
    if "kb_id" not in columns:
        op.add_column("document_records", sa.Column("kb_id", sa.String(36), nullable=True))
        op.create_index("ix_document_records_kb_id", "document_records", ["kb_id"])

    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False, index=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.String(500), nullable=False, server_default=""),
        sa.Column("collection_name", sa.String(255), nullable=False),
        sa.Column("doc_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("is_default", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime),
        sa.Column("updated_at", sa.DateTime),
        sa.UniqueConstraint("user_id", "name", name="uq_kb_user_name"),
    )

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False, index=True),
        sa.Column("kb_id", sa.String(36), nullable=True, index=True),
        sa.Column("title", sa.String(200), nullable=False, server_default="新会话"),
        sa.Column("message_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_message_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime),
        sa.Column("updated_at", sa.DateTime),
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(36), nullable=False, index=True),
        sa.Column("user_id", sa.String(36), nullable=False, index=True),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("sources_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, index=True),
    )
    op.create_index(
        "ix_msg_session_created", "chat_messages", ["session_id", "created_at"]
    )

    # SQLite 不支持直接删除约束，batch_alter_table 会自动重建表。
    with op.batch_alter_table("document_records") as batch:
        batch.drop_constraint("uq_docrecord_user_source", type_="unique")
        batch.create_unique_constraint(
            "uq_docrecord_user_kb_source", ["user_id", "kb_id", "source"]
        )


def downgrade() -> None:
    with op.batch_alter_table("document_records") as batch:
        batch.drop_constraint("uq_docrecord_user_kb_source", type_="unique")
        batch.create_unique_constraint(
            "uq_docrecord_user_source", ["user_id", "source"]
        )

    op.drop_index("ix_msg_session_created", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("knowledge_bases")
    if "kb_id" in _column_names("document_records"):
        op.drop_index("ix_document_records_kb_id", table_name="document_records")
        op.drop_column("document_records", "kb_id")
