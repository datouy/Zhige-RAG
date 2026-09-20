"""数据库连接管理，支持连接池和数据库迁移。

支持 SQLite（开发）和 PostgreSQL（生产）。
PostgreSQL 连接池配置：
- pool_size: 基础连接数
- max_overflow: 允许的额外连接数
- pool_timeout: 获取连接超时（秒）
- pool_recycle: 连接回收时间（秒）
- pool_pre_ping: 使用前验证连接
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text, Engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import NullPool, QueuePool

from .models import Base

# 数据库 URL 配置
DB_PATH = os.getenv("DB_PATH", "data/users.db")
# 相对路径锚定项目根，避免从其它工作目录启动时新建空库、用户"突然全部登录失败"
if not Path(DB_PATH).is_absolute():
    DB_PATH = str(Path(__file__).resolve().parent.parent / DB_PATH)
DB_DIR = Path(DB_PATH).parent
DB_DIR.mkdir(parents=True, exist_ok=True)

# 检测是否为 PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "")
_is_postgresql = DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")


def _create_engine() -> Engine:
    """根据数据库类型创建优化的 Engine。

    SQLite: 使用 NullPool（每请求独立连接）。之前用 StaticPool 会让所有线程
    共享同一条 DBAPI 连接，FastAPI 线程池并发时游标状态互相串扰。
    PostgreSQL: 使用 QueuePool（连接池，生产环境推荐）
    """
    if _is_postgresql:
        # PostgreSQL 生产环境配置
        return create_engine(
            DATABASE_URL,
            pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "40")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
            pool_pre_ping=os.getenv("DB_POOL_PRE_PING", "true").lower() == "true",
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
            poolclass=QueuePool,
        )
    else:
        # SQLite 开发环境配置
        return create_engine(
            f"sqlite:///{DB_PATH}",
            connect_args={"check_same_thread": False, "timeout": 10},
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
            poolclass=NullPool,
        )


engine: Engine = _create_engine()

if not _is_postgresql:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
        """每个新连接启用 WAL 与写等待，降低并发下 database is locked 概率。"""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """初始化数据库表，并对老库做轻量列迁移。

    ``create_all`` 只建缺失的**表**，不会给已有表加列——新版本给
    ``users`` 加了 ``department``（数据层 ACL 分组依据），老库直接启动会
    在注册/登录时报 ``no such column: users.department``。这里按方言做
    存在性检查后 ``ALTER TABLE`` 补列；完整迁移仍走 alembic。
    """
    Base.metadata.create_all(bind=engine)
    try:
        with engine.connect() as conn:
            if engine.dialect.name == "sqlite":
                cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(users)")}
            else:
                cols = {
                    row[0]
                    for row in conn.exec_driver_sql(
                        "SELECT column_name FROM information_schema.columns WHERE table_name='users'"
                    )
                }
            if "department" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE users ADD COLUMN department VARCHAR(100) NOT NULL DEFAULT ''"
                )
                conn.commit()
    except Exception:
        import logging

        logging.getLogger(__name__).warning("users.department 列迁移失败（可能已存在）", exc_info=True)


def get_db() -> Generator[Session, None, None]:
    """获取数据库会话的依赖项（用于 FastAPI 依赖注入）。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def close_db() -> None:
    """关闭数据库引擎（用于优雅关闭）。"""
    engine.dispose()


def check_connection() -> bool:
    """检查数据库连接是否正常。"""
    try:
        # SQLAlchemy 2.0 要求裸 SQL 用 text() 包装，否则抛 ArgumentError
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
