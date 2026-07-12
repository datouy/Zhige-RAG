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