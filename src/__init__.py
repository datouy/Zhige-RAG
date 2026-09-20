"""中文个人知识库 RAG 系统核心模块。

该包提供以下子模块：
- document_loader: 多格式文档解析（PDF、Word、Markdown、TXT）
- text_splitter:   中文智能分块
- embeddings:      Embedding 模型封装
- vector_store:    Chroma 向量库封装
- llm:             本地 LLM 加载与推理（支持 4bit 量化）
- reranker:        重排序模型（可选）
- rag_pipeline:    RAG 主流程：检索 → 重排 → 生成
- prompt_template: 中文 Prompt 模板
- utils:           通用工具（日志、计时、文件处理）
"""

# 统一加载项目根目录的 .env。
# 所有入口（api/main.py、scripts/*.py、ui/app.py）都会导入 src 包，
# 在此处 load_dotenv 可保证 .env 中的 JWT_SECRET / DB_PATH / HF_ENDPOINT 等配置生效。
# override=False：真实环境变量（Docker / K8s 注入）优先级高于 .env 文件。
try:  # pragma: no cover - 环境相关的兜底，dotenv 缺失或 .env 不存在时静默跳过
    from pathlib import Path as _Path

    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(_Path(__file__).resolve().parent.parent / ".env", override=False)
except Exception:
    pass

__version__ = "0.1.0"
__all__ = [
    "document_loader",
    "text_splitter",
    "embeddings",
    "vector_store",
    "llm",
    "reranker",
    "rag_pipeline",
    "prompt_template",
    "utils",
]