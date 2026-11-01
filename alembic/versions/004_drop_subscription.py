"""Drop subscription / quota columns from users.

Revision ID: 004_drop_subscription
Revises: 003_multi_kb_schema
Create Date: 2026-11-01
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "004_drop_subscription"
down_revision: Union[str, None] = "003_multi_kb_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove the subscription tier and quota columns that are no longer used."""
    with op.batch_alter_table("users") as batch:
        batch.drop_column("subscription_tier")
        batch.drop_column("max_chunks")
        batch.drop_column("max_queries_per_day")


def downgrade() -> None:
    """Restore the removed columns (nullable) to undo the cleanup."""
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("subscription_tier", sa.String(20), nullable=True))
        batch.add_column(sa.Column("max_chunks", sa.Integer, nullable=True))
        batch.add_column(sa.Column("max_queries_per_day", sa.Integer, nullable=True))
