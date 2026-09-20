"""为新用户创建默认知识库。

在用户首次注册时调用此脚本，确保每个用户都有一个默认知识库。
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.database import init_db
from src.factories import TenantAwareFactory
from src.utils import load_config, apply_env_overrides

def setup_default_knowledge_base(user_id: str, user_name: str = "默认用户") -> dict:
    """为指定用户创建默认知识库。
    
    Args:
        user_id: 用户 ID
        user_name: 用户名称（用于日志）
    
    Returns:
        创建结果字典，包含 status、collection_name 等信息
    """
    init_db()
    cfg = load_config("config/config.yaml")
    cfg = apply_env_overrides(cfg)
    
    # 获取用户专属的 ChromaStore
    chroma_store = TenantAwareFactory.get_chroma_store(user_id, cfg)
    
    # 获取默认 collection 名称
    default_collection = cfg["vector_store"].get("collection_name", "chinese_rag_kb")
    user_collection = f"{default_collection}_{user_id[:8]}"
    
    # 初始化 collection（如果不存在）
    try:
        count = chroma_store.count()
        status = "ready"
    except Exception:
        # Collection 不存在，创建一个空的
        chroma_store.collection = chroma_store._get_or_create_collection(user_collection)
        count = 0
        status = "created"
    
    result = {
        "status": status,
        "user_id": user_id,
        "user_name": user_name,
        "collection_name": user_collection,
        "initial_chunk_count": count,
        "message": f"用户 {user_name} 的默认知识库已就绪",
    }
    
    print(f"[setup_default_kb] {result['message']}")
    print(f"  - User ID: {user_id}")
    print(f"  - Collection: {user_collection}")
    print(f"  - Initial chunks: {count}")
    
    return result


def create_sample_knowledge_base(user_id: str) -> dict:
    """为用户创建示例知识库（包含示例文档）。
    
    Args:
        user_id: 用户 ID
    
    Returns:
        创建结果字典
    """
    init_db()
    cfg = load_config("config/config.yaml")
    cfg = apply_env_overrides(cfg)
    
    sample_chunks = [
        {
            "text": "欢迎使用中文知识库 RAG 系统。这是一个示例文档，介绍系统的主要功能。",
            "metadata": {"source": "welcome.txt", "title": "欢迎文档", "type": "sample"},
        },
        {
            "text": "RAG（检索增强生成）是一种结合检索系统和语言模型的技术，可以根据知识库中的内容生成准确的回答。",
            "metadata": {"source": "rag_intro.txt", "title": "RAG 介绍", "type": "sample"},
        },
        {
            "text": "支持的功能包括：文档上传、分块、语义检索、重排序、流式回答等。",
            "metadata": {"source": "features.txt", "title": "系统功能", "type": "sample"},
        },
    ]
    
    store = TenantAwareFactory.get_chroma_store(user_id, cfg)
    added = store.add_chunks(sample_chunks)
    
    result = {
        "status": "sample_created",
        "user_id": user_id,
        "chunks_added": added,
        "message": f"已为用户 {user_id[:8]} 创建示例知识库",
    }
    
    print(f"[setup_default_kb] {result['message']}")
    print(f"  - Chunks added: {added}")
    
    return result


def ensure_user_kb_exists(user_id: str, user_name: str = "默认用户", create_sample: bool = True) -> dict:
    """确保用户知识库存在，如不存在则创建。
    
    Args:
        user_id: 用户 ID
        user_name: 用户名称
        create_sample: 是否创建示例知识库
    
    Returns:
        创建结果字典
    """
    result = setup_default_knowledge_base(user_id, user_name)
    
    if create_sample and result["initial_chunk_count"] == 0:
        sample_result = create_sample_knowledge_base(user_id)
        result.update(sample_result)
    
    return result


if __name__ == "__main__":
    import uuid
    
    # 生成一个测试用户 ID
    test_user_id = str(uuid.uuid4())
    print(f"Testing with user_id: {test_user_id}")
    
    # 运行设置
    result = ensure_user_kb_exists(test_user_id, "测试用户", create_sample=True)
    print("\nResult:")
    print(result)
