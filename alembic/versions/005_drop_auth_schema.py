"""Drop auth schema (password_hash / refresh_tokens) to match models.py.

Revision ID: 005_drop_auth_schema
Revises: 004_drop_subscription
Create Date: 2026-11-10

背景：0.3.0 起不再提供登录 / 注册，``src.db.models`` 已删除
``users.password_hash`` 与 ``refresh_tokens``。此前 alembic 链与
``init_db()`` 双轨漂移——alembic 001 建的 auth 结构，运行时由
``database._cleanup_legacy_auth_schema`` 删除，但 alembic 链里没有对应迁移。
本迁移把 alembic 链补齐，使 ``alembic upgrade head`` 的最终 schema 与
``models.py`` / ``init_db`` 完全一致。
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "005_drop_auth_schema"
down_revision: Union[str, None] = "004_drop_subscription"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """删除免登录后废弃的 auth 结构，与 models.py 现状对齐。"""
    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("password_hash")


def downgrade() -> None:
    """回滚：恢复登录时代的 password_hash 列与 refresh_tokens 表。"""
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("password_hash", sa.String(255), nullable=True))
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("token_hash", sa.String(255), nullable=False),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("created_at", sa.DateTime, default=sa.func.utcnow()),
        sa.Column("revoked", sa.Boolean, default=False),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
