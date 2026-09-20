"""Alembic migration environment configuration.

Loads configuration from config.yaml for database connection settings.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Project root and path setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import models and database setup
from src.db.models import Base
from src.utils import load_config, apply_env_overrides, resolve_path

# this is the Alembic Config object
config = context.config

# Load database URL from config.yaml
try:
    cfg = load_config("config/config.yaml")
    cfg = apply_env_overrides(cfg)
    
    # Support both PostgreSQL and SQLite
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        # Check if PostgreSQL is configured
        db_config = cfg.get("database", {})
        if db_config.get("url"):
            db_url = db_config["url"]
        else:
            # Fallback to SQLite
            db_path = os.getenv("DB_PATH", "data/users.db")
            db_dir = resolve_path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            db_url = f"sqlite:///{db_path}"
    
    # Override sqlalchemy.url in alembic config
    config.set_main_option("sqlalchemy.url", db_url)
except Exception as e:
    print(f"Warning: Could not load database URL from config: {e}")
    # Use default SQLite path
    db_path = resolve_path("data/users.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate support
target_metadata = Base.metadata


def get_url() -> str:
    """Get database URL from config."""
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine.
    Calls to context.execute() emit the given string to the script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a
    connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
