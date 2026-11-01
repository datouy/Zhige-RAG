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
from typing import Generator

from sqlalchemy import create_engine, event, inspect, text, Engine
from sqlalchemy.orm import sessionmaker, Session
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


def _migrate_document_record_key() -> None:
    """把文档台账唯一键从 ``(user_id, source)`` 升级为含知识库的版本。

    ``create_all`` 不会修改既有约束。若不主动迁移，同名文件写入第二个
    知识库时会撞上旧唯一键，登记异常又被调用方吞掉，最终表现为“上传成功但
    台账/统计没更新”。SQLite 通过重建表完成约束替换；PostgreSQL 直接替换。
    """
    insp = inspect(engine)
    if "document_records" not in insp.get_table_names():
        return

    columns = {column["name"] for column in insp.get_columns("document_records")}
    if "kb_id" not in columns:
        return

    constraints = insp.get_unique_constraints("document_records")
    old = next(
        (
            item
            for item in constraints
            if item.get("column_names") == ["user_id", "source"]
        ),
        None,
    )
    new = next(
        (
            item
            for item in constraints
            if item.get("column_names") == ["user_id", "kb_id", "source"]
        ),
        None,
    )
    if new is not None and old is None:
        return

    if engine.dialect.name == "sqlite":
        with engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE document_records RENAME TO document_records_legacy_key")
            conn.exec_driver_sql(
                """
                CREATE TABLE document_records (
                    id VARCHAR(36) NOT NULL PRIMARY KEY,
                    user_id VARCHAR(36) NOT NULL,
                    kb_id VARCHAR(36),
                    source VARCHAR(512) NOT NULL,
                    title VARCHAR(512),
                    doc_status VARCHAR(20) NOT NULL DEFAULT 'active',
                    acl VARCHAR(512) NOT NULL DEFAULT '*',
                    department VARCHAR(100),
                    content_hash VARCHAR(64),
                    chunk_count INTEGER DEFAULT 0,
                    file_ext VARCHAR(16),
                    updated_at_source DATETIME,
                    created_at DATETIME,
                    updated_at DATETIME,
                    CONSTRAINT uq_docrecord_user_kb_source
                        UNIQUE (user_id, kb_id, source)
                )
                """
            )
            conn.exec_driver_sql(
                """
                INSERT INTO document_records (
                    id, user_id, kb_id, source, title, doc_status, acl,
                    department, content_hash, chunk_count, file_ext,
                    updated_at_source, created_at, updated_at
                )
                SELECT
                    id, user_id, kb_id, source, title, doc_status, acl,
                    department, content_hash, chunk_count, file_ext,
                    updated_at_source, created_at, updated_at
                FROM document_records_legacy_key
                """
            )
            conn.exec_driver_sql("DROP TABLE document_records_legacy_key")
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_document_records_user_id "
                "ON document_records (user_id)"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_document_records_kb_id "
                "ON document_records (kb_id)"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_document_records_content_hash "
                "ON document_records (content_hash)"
            )
    else:
        with engine.begin() as conn:
            if old is not None and old.get("name"):
                conn.exec_driver_sql(
                    f'ALTER TABLE document_records DROP CONSTRAINT "{old["name"]}"'
                )
            if new is None:
                conn.exec_driver_sql(
                    "ALTER TABLE document_records ADD CONSTRAINT "
                    "uq_docrecord_user_kb_source "
                    "UNIQUE (user_id, kb_id, source)"
                )
    _log.info("文档台账唯一键已升级为 (user_id, kb_id, source)")

# 登录/注册时代的遗留结构清理：{表名: [列名]} 与 (表名, ...)
#
# 老版本支持注册登录，库里因此有 ``users.password_hash``（NOT NULL）和
# ``refresh_tokens`` 表。服务现在不需要登录：password_hash 留着不仅无用，
# 还会让"自动创建本地用户"撞上 NOT NULL 约束直接报错；refresh_tokens
# 只是一堆死数据。启动时按存在性删除，重复执行安全。
_LEGACY_COLUMN_CLEANUP = {"users": ["password_hash", "subscription_tier", "max_chunks", "max_queries_per_day"]}
_LEGACY_TABLE_CLEANUP = ("refresh_tokens",)

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
    """初始化数据库表，并对老库做轻量列迁移 + 遗留结构清理。

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

    _migrate_document_record_key()
    _cleanup_legacy_auth_schema()


def _cleanup_legacy_auth_schema() -> None:
    """删除登录 / 注册时代遗留的列与表。

    见 ``_LEGACY_COLUMN_CLEANUP`` / ``_LEGACY_TABLE_CLEANUP``。只在结构确实存在时
    动手，失败也不阻塞启动 —— 老库能继续用，只是空留一个无用列/表。
    """
    try:
        existing_tables = set(inspect(engine).get_table_names())
    except Exception:  # noqa: BLE001
        _log.warning("遗留结构清理跳过：无法读取表清单", exc_info=True)
        return

    for table, columns in _LEGACY_COLUMN_CLEANUP.items():
        if table not in existing_tables:
            continue
        try:
            existing_columns = {c["name"] for c in inspect(engine).get_columns(table)}
        except Exception:  # noqa: BLE001
            continue
        for column in columns:
            if column not in existing_columns:
                continue
            try:
                with engine.begin() as conn:
                    # 表名/列名来自本模块的常量，不是外部输入
                    conn.exec_driver_sql(f"ALTER TABLE {table} DROP COLUMN {column}")
                _log.info("遗留结构清理：%s 已删除列 %s", table, column)
            except Exception:  # noqa: BLE001
                _log.warning("%s.%s 删除失败（已跳过，不影响启动）", table, column, exc_info=True)

    for table in _LEGACY_TABLE_CLEANUP:
        if table not in existing_tables:
            continue
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql(f"DROP TABLE {table}")
            _log.info("遗留结构清理：已删除表 %s", table)
        except Exception:  # noqa: BLE001
            _log.warning("删除表 %s 失败（已跳过，不影响启动）", table, exc_info=True)


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
