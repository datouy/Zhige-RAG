"""Add knowledge-base visibility, department scope and member ACL.

Revision ID: 007_kb_object_acl
Revises: 006_enterprise_auth
Create Date: 2026-11-23
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007_kb_object_acl"
down_revision: Union[str, None] = "006_enterprise_auth"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.add_column(
            sa.Column("visibility", sa.String(20), nullable=False, server_default="private")
        )
        batch.add_column(
            sa.Column("department", sa.String(100), nullable=False, server_default="")
        )
        batch.create_index("ix_knowledge_bases_visibility", ["visibility"], unique=False)
        batch.create_index("ix_knowledge_bases_department", ["department"], unique=False)

    op.create_table(
        "knowledge_base_members",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("kb_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("permission", sa.String(20), nullable=False, server_default="read"),
        sa.Column("created_by", sa.String(36), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("kb_id", "user_id", name="uq_kb_member"),
    )
    op.create_index("ix_knowledge_base_members_kb_id", "knowledge_base_members", ["kb_id"])
    op.create_index("ix_knowledge_base_members_user_id", "knowledge_base_members", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_knowledge_base_members_user_id", table_name="knowledge_base_members")
    op.drop_index("ix_knowledge_base_members_kb_id", table_name="knowledge_base_members")
    op.drop_table("knowledge_base_members")
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.drop_index("ix_knowledge_bases_department")
        batch.drop_index("ix_knowledge_bases_visibility")
        batch.drop_column("department")
        batch.drop_column("visibility")
