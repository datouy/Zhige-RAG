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

import logging
import os
import shutil
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text, Engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import NullPool, QueuePool

from .models import Base

_log = logging.getLogger(__name__)

# 项目根目录：src/db/database.py 往上三级
# 注意：此前只上了两级，得到的是 ``src/`` 而不是项目根，导致 SQLite 库落到
# ``src/data/users.db``，与文档（``data/users.db``）以及 data/ 下的其他数据
# （chroma_db / raw / kg_*.db）分家。这里修正为真正的项目根。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 数据库 URL 配置
DB_PATH = os.getenv("DB_PATH", "data/users.db")
# 相对路径锚定项目根，避免从其它工作目录启动时新建空库、用户"突然全部登录失败"
if not Path(DB_PATH).is_absolute():
    DB_PATH = str(_PROJECT_ROOT / DB_PATH)

_PATH_FIXED = False
if os.getenv("DB_PATH") is None:
    # 兼容历史错误：老版本把 ``data/users.db`` 解析成了 ``src/data/users.db``。
    # 若新位置没有库、而老位置有，则自动搬过来 —— 否则老用户会"凭空登录失败"。
    _legacy = _PROJECT_ROOT / "src" / "data" / "users.db"
    _current = Path(DB_PATH)
    if _legacy.exists() and not _current.exists() and _current.name == "users.db":
        _current.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            old = Path(str(_legacy) + suffix)
            if old.exists():
                shutil.move(str(old), str(Path(str(_current) + suffix)))
        _PATH_FIXED = True

DB_DIR = Path(DB_PATH).parent
DB_DIR.mkdir(parents=True, exist_ok=True)

if _PATH_FIXED:
    _log.warning(
        "检测到旧版数据库位于 src/data/，已迁移到 %s（这是路径修正，不是数据丢失）", DB_PATH
    )

# 轻量列迁移表：{表名: [(列名, DDL 片段)]}
# 只用于给**已存在的表**补列（create_all 不会做这件事），完整迁移走 alembic。
_LIGHT_COLUMN_MIGRATIONS = {
    "users": [
        # 数据层 ACL 分组依据
        ("department", "VARCHAR(100) NOT NULL DEFAULT ''"),
    ],
    "document_records": [
        # 知识库概念层：文档归属到具体知识库；老数据留空 = 默认库
        ("kb_id", "VARCHAR(36)"),
    ],
}

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

    ``create_all`` 只建缺失的**表**，不会给已有表加列。新版本给既有表加了列时，
    老库直接启动会在读写时报 ``no such column``。这里按方言做存在性检查后
    ``ALTER TABLE`` 补列；完整迁移仍走 alembic。
    """
    Base.metadata.create_all(bind=engine)

    for table, columns in _LIGHT_COLUMN_MIGRATIONS.items():
        try:
            with engine.connect() as conn:
                if engine.dialect.name == "sqlite":
                    existing = {
                        row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")
                    }
                else:
                    existing = {
                        row[0]
                        for row in conn.exec_driver_sql(
                            "SELECT column_name FROM information_schema.columns "
                            f"WHERE table_name='{table}'"
                        )
                    }
                for column, ddl in columns:
                    if column not in existing:
                        conn.exec_driver_sql(
                            f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"
                        )
                        conn.commit()
                        _log.info("轻量迁移：%s 新增列 %s", table, column)
        except Exception:  # noqa: BLE001
            _log.warning("%s 列迁移跳过（可能已存在）", table, exc_info=True)


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
