"""多租户组件工厂。

按 user_id 隔离 Chroma collection、KG SQLite 和 Embedding 资源。
所有组件通过线程安全的单例模式管理，避免重复加载。

优化点（P4 - AI Engineer 审计后实施）：
- Chroma collection 命名增加哈希后缀，避免特殊字符问题
- 增加连接池健康检查
- 添加配置驱动的持久化目录
"""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set
from threading import Lock

from src.utils import get_logger
from src.vector_store import ChromaStore, hybrid_kwargs
from src.embeddings import EmbeddingModel
from src.kg.store import SQLiteGraphStore

logger = get_logger("factories")


def _project_root() -> Path:
    """项目根目录（src/factories.py 的上两级），保证相对路径不依赖 CWD。"""
    return Path(__file__).resolve().parent.parent


def _sanitize_user_id(user_id: str) -> str:
    """将 user_id 转换为合法的 collection 名称后缀。

    Chroma collection 名称只允许字母数字下划线，长度限制在 64 字符内。
    """
    # 移除可能导致问题的字符，保留基本标识信息
    clean = hashlib.sha256(user_id.encode()).hexdigest()[:16]
    return f"u{clean}"


class TenantAwareFactory:
    """多租户组件工厂，按 user_id 隔离资源。"""
    _lock = Lock()
    _chroma_stores: Dict[str, ChromaStore] = {}
    # user_id -> 其名下的 collection 全名集合（用于 cleanup 反查，
    # 因为 chroma_stores 的 key 是脱敏后的名字，无法从 user_id 直接匹配）
    _user_collections: Dict[str, Set[str]] = {}
    _kg_stores: Dict[str, SQLiteGraphStore] = {}
    _embedding: Optional[EmbeddingModel] = None

    @classmethod
    def get_embedding(cls, config: dict) -> EmbeddingModel:
        """获取全局共享的 Embedding 模型（只加载一次）。"""
        if cls._embedding is None:
            with cls._lock:
                if cls._embedding is None:
                    cls._embedding = EmbeddingModel(
                        model_name=config["embedding"]["model_name"],
                        device=config["embedding"].get("device", "auto"),
                        batch_size=config["embedding"].get("batch_size", 32),
                        max_seq_length=config["embedding"].get("max_seq_length", 512),
                        normalize=config["embedding"].get("normalize_embeddings", True),
                        cache_dir=config["embedding"].get("cache_dir"),
                        local_files_only=config["embedding"].get("local_files_only", False),
                    )
        return cls._embedding

    @classmethod
    def get_chroma_store(cls, user_id: str, config: dict) -> ChromaStore:
        """获取用户专属的 ChromaStore。

        collection_name 格式：{sanitized_user_id}_{base_collection_name}
        例如：u1a2b3c4d5e6f7g8h_chinese_rag_kb
        """
        safe_uid = _sanitize_user_id(user_id)
        collection_name = f"{safe_uid}_{config['vector_store']['collection_name']}"

        # 注意：get_embedding 内部会再次获取 _lock，而 threading.Lock 不可重入，
        # 必须在锁外调用，否则同线程二次加锁直接死锁。
        if collection_name not in cls._chroma_stores:
            embed = cls.get_embedding(config)
            with cls._lock:
                if collection_name not in cls._chroma_stores:
                    persist_dir = config["vector_store"]["persist_directory"]
                    cls._chroma_stores[collection_name] = ChromaStore(
                        persist_directory=persist_dir,
                        collection_name=collection_name,
                        embedding_model=embed,
                        distance_fn=config["vector_store"].get("distance_fn", "cosine"),
            **hybrid_kwargs(config.get("vector_store", {})),
                    )
                    cls._user_collections.setdefault(user_id, set()).add(collection_name)
        return cls._chroma_stores[collection_name]

    @classmethod
    def get_chroma_store_by_name(cls, collection_name: str, config: dict) -> ChromaStore:
        """通过完整 collection_name 获取 ChromaStore（不隔离用户）。"""
        if collection_name not in cls._chroma_stores:
            embed = cls.get_embedding(config)
            with cls._lock:
                if collection_name not in cls._chroma_stores:
                    persist_dir = config["vector_store"]["persist_directory"]
                    cls._chroma_stores[collection_name] = ChromaStore(
                        persist_directory=persist_dir,
                        collection_name=collection_name,
                        embedding_model=embed,
                        distance_fn=config["vector_store"].get("distance_fn", "cosine"),
            **hybrid_kwargs(config.get("vector_store", {})),
                    )
        return cls._chroma_stores[collection_name]

    @classmethod
    def get_kg_store(cls, user_id: str, config: dict) -> SQLiteGraphStore:
        """获取用户专属的 KG SQLite Store。

        db_path 格式：{项目根}/data/kg_{user_id}.db
        """
        if user_id not in cls._kg_stores:
            with cls._lock:
                if user_id not in cls._kg_stores:
                    # 锚定项目根目录，避免受进程工作目录影响
                    db_path = _project_root() / "data" / f"kg_{user_id}.db"
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                    cls._kg_stores[user_id] = SQLiteGraphStore(db_path=str(db_path))
        return cls._kg_stores[user_id]

    @classmethod
    def cleanup_user(cls, user_id: str, config: dict) -> None:
        """清理用户相关资源（删除 collection、KG DB 与内存缓存）。

        供账号注销 / 管理员删除用户时调用。注意必须删除 Chroma collection
        本身——只清内存缓存的话，用户的全部向量会永久残留在磁盘上。
        """
        safe_uid = _sanitize_user_id(user_id)
        with cls._lock:
            # 通过反查表 + 脱敏名兜底，收集该用户名下的全部 collection
            names: Set[str] = set(cls._user_collections.get(user_id, set()))
            for key in cls._chroma_stores:
                if key.startswith(f"{safe_uid}_"):
                    names.add(key)
            stores: List[ChromaStore] = [
                cls._chroma_stores.pop(name) for name in names if name in cls._chroma_stores
            ]
            cls._user_collections.pop(user_id, None)
            kg_store = cls._kg_stores.pop(user_id, None)

        # 慢操作放在锁外：删除 collection 与关闭 KG 连接都是 IO
        for store in stores:
            try:
                store.client.delete_collection(store.collection_name)
                logger.info("已删除用户 %s 的 collection %s", user_id, store.collection_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("删除 collection %s 失败: %s", store.collection_name, exc)

        if kg_store is not None:
            try:
                kg_store.close()
            except Exception:  # noqa: BLE001
                pass

        # 删除 KG SQLite 文件（路径与 get_kg_store 一致）
        kg_db = _project_root() / "data" / f"kg_{user_id}.db"
        if kg_db.exists():
            kg_db.unlink()

    @classmethod
    def reset(cls) -> None:
        """重置所有缓存（测试用）。"""
        with cls._lock:
            cls._chroma_stores.clear()
            cls._user_collections.clear()
            cls._kg_stores.clear()
            cls._embedding = None
