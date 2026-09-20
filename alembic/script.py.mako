"""Alembic migration script template.

Revision ID: {rev_id}
Revises: {down_revision}
Create Date: {create_date}

"""
from __future__ import annotations
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
{imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = {repr(up_revision)}
down_revision: Union[str, None] = {repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    {upgrades if upgrades else "pass"}


def downgrade() -> None:
    {downgrades if downgrades else "pass"}
