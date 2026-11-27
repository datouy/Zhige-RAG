"""Reconcile migration head with the current enterprise data model.

Revision ID: 006_restore_enterprise_schema
Revises: 005_drop_auth_schema
Create Date: 2026-11-27
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from src.db.models import Base


revision: str = "006_restore_enterprise_schema"
down_revision: Union[str, None] = "005_drop_auth_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_ENTERPRISE_TABLES = (
    "saved_agents",
    "mcp_servers",
    "auth_sessions",
    "knowledge_base_members",
    "knowledge_gaps",
)


def _columns(table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(
        bind=bind,
        tables=[Base.metadata.tables[name] for name in _ENTERPRISE_TABLES],
    )

    additions = {
        "users": [
            (
                "role",
                sa.Column("role", sa.String(20), nullable=False, server_default="viewer"),
            ),
            ("password_hash", sa.Column("password_hash", sa.String(255), nullable=True)),
            (
                "failed_login_attempts",
                sa.Column(
                    "failed_login_attempts",
                    sa.Integer,
                    nullable=False,
                    server_default="0",
                ),
            ),
            ("locked_until", sa.Column("locked_until", sa.DateTime(), nullable=True)),
        ],
        "feedback_records": [
            (
                "citation_issue",
                sa.Column(
                    "citation_issue",
                    sa.Boolean,
                    nullable=False,
                    server_default=sa.false(),
                ),
            ),
            (
                "missing_source",
                sa.Column(
                    "missing_source",
                    sa.Boolean,
                    nullable=False,
                    server_default=sa.false(),
                ),
            ),
        ],
        "document_records": [
            (
                "index_status",
                sa.Column(
                    "index_status",
                    sa.String(20),
                    nullable=False,
                    server_default="indexed",
                ),
            ),
            ("last_indexed_at", sa.Column("last_indexed_at", sa.DateTime(), nullable=True)),
            ("last_index_error", sa.Column("last_index_error", sa.Text(), nullable=True)),
        ],
        "chat_messages": [
            ("verification_json", sa.Column("verification_json", sa.Text(), nullable=True)),
            ("timings_json", sa.Column("timings_json", sa.Text(), nullable=True)),
        ],
        "usage_logs": [
            ("resource", sa.Column("resource", sa.String(512), nullable=True)),
            ("details_json", sa.Column("details_json", sa.Text(), nullable=True)),
        ],
    }
    for table, columns in additions.items():
        existing = _columns(table)
        for name, column in columns:
            if name not in existing:
                op.add_column(table, column)


def downgrade() -> None:
    # The five enterprise tables were runtime-created by earlier builds and
    # may contain production data. Dropping them on downgrade is unsafe.
    # Column removals are likewise destructive; roll back by restoring the
    # pre-upgrade database snapshot described in the migration manifest.
    pass
