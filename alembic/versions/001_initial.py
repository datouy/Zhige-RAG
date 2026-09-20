"""Initial migration for User, RefreshToken, and UsageLog models.

Revision ID: 001_initial
Revises:
Create Date: 2026-07-18

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial tables for User, RefreshToken, and UsageLog."""
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("username", sa.String(50), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("subscription_tier", sa.String(20), default="free"),
        sa.Column("max_chunks", sa.Integer, default=1000),
        sa.Column("max_queries_per_day", sa.Integer, default=50),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("is_admin", sa.Boolean, default=False),
        sa.Column("created_at", sa.DateTime, default=sa.func.utcnow()),
        sa.Column("updated_at", sa.DateTime, default=sa.func.utcnow(), onupdate=sa.func.utcnow()),
    )
    
    # Create indexes for users
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    
    # Create refresh_tokens table
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("token_hash", sa.String(255), nullable=False),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("created_at", sa.DateTime, default=sa.func.utcnow()),
        sa.Column("revoked", sa.Boolean, default=False),
    )
    
    # Create index for refresh_tokens
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    
    # Create usage_logs table
    op.create_table(
        "usage_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("endpoint", sa.String(100), nullable=False),
        sa.Column("method", sa.String(10), nullable=False),
        sa.Column("status_code", sa.Integer, nullable=True),
        sa.Column("tokens_used", sa.Integer, default=0),
        sa.Column("latency_ms", sa.Integer, default=0),
        sa.Column("chunks_accessed", sa.Integer, default=0),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, default=sa.func.utcnow()),
    )
    
    # Create indexes for usage_logs
    op.create_index("ix_usage_logs_user_id", "usage_logs", ["user_id"])
    op.create_index("ix_usage_logs_created_at", "usage_logs", ["created_at"])


def downgrade() -> None:
    """Drop all initial tables."""
    op.drop_index("ix_usage_logs_created_at", "usage_logs")
    op.drop_index("ix_usage_logs_user_id", "usage_logs")
    op.drop_table("usage_logs")
    
    op.drop_index("ix_refresh_tokens_user_id", "refresh_tokens")
    op.drop_table("refresh_tokens")
    
    op.drop_index("ix_users_email", "users")
    op.drop_index("ix_users_username", "users")
    op.drop_table("users")
