"""配置验证器，使用 Pydantic 验证配置文件的结构和类型。

确保配置文件符合预期的 schema，避免运行时错误。
支持从 config.yaml 和环境变量加载配置。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ======================== 数据库配置 ========================

class DatabaseConfig(BaseModel):
    """数据库配置模型。"""
    url: Optional[str] = Field(None, description="数据库连接 URL")
    pool_size: int = Field(20, ge=1, le=100, description="连接池大小")
    max_overflow: int = Field(40, ge=0, le=100, description="最大溢出连接数")
    pool_timeout: int = Field(30, ge=1, description="连接获取超时（秒）")
    pool_recycle: int = Field(1800, ge=300, description="连接回收时间（秒）")
    pool_pre_ping: bool = Field(True, description="使用前验证连接")


# ======================== JWT 配置 ========================

class JWTConfig(BaseModel):
    """JWT 配置模型。"""
    secret: str = Field(..., min_length=32, description="JWT 密钥（至少 32 字符）")
    algorithm: str = Field("HS256", description="JWT 算法")
    access_token_expire_minutes: int = Field(30, ge=1, le=1440, description="访问令牌过期时间（分钟）")
    refresh_token_expire_days: int = Field(7, ge=1, le=30, description="刷新令牌过期时间（天）")


# ======================== LLM 配置 ========================

class LLMQuantizationConfig(BaseModel):
    """LLM 量化配置。"""
    enabled: bool = Field(False, description="是否启用量化")
    quant_type: str = Field("nf4", description="量化类型")
    compute_dtype: str = Field("float16", description="计算数据类型")
    double_quant: bool = Field(True, description="是否双重量化")


class LLMGenerationConfig(BaseModel):
    """LLM 生成参数配置。"""
    max_new_tokens: int = Field(512, ge=1, le=4096, description="最大生成 token 数")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="采样温度")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p 采样")
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0, description="重复惩罚")
    do_sample: bool = Field(True, description="是否采样")


class LLMConfig(BaseModel):
    """LLM 配置模型。"""
    model_name: str = Field(..., description="模型名称或路径")
    model_path: Optional[str] = Field("", description="本地模型路径")
    device: str = Field("auto", description="设备类型")
    device_map: str = Field("auto", description="设备映射")
    torch_dtype: str = Field("auto", description="张量数据类型")
    cache_dir: Optional[str] = Field(None, description="模型缓存目录")
    trust_remote_code: bool = Field(False, description="是否信任远程代码")
    local_files_only: bool = Field(False, description="是否仅使用本地文件")
    use_chat_template: bool = Field(True, description="是否使用聊天模板")
    chat_template: str = Field("auto", description="聊天模板类型")
    quantization: LLMQuantizationConfig = Field(default_factory=LLMQuantizationConfig)
    generation: LLMGenerationConfig = Field(default_factory=LLMGenerationConfig)


# ======================== Embedding 配置 ========================

class EmbeddingConfig(BaseModel):
    """Embedding 配置模型。"""
    model_name: str = Field(..., description="Embedding 模型名称")
    device: str = Field("auto", description="设备类型")
    batch_size: int = Field(32, ge=1, le=256, description="批处理大小")
    normalize_embeddings: bool = Field(True, description="是否归一化向量")
    max_seq_length: int = Field(512, ge=1, le=2048, description="最大序列长度")
    query_instruction: Optional[str] = Field(None, description="查询指令")
    use_doc_instruction: bool = Field(False, description="是否使用文档指令")
    cache_dir: Optional[str] = Field(None, description="缓存目录")
    local_files_only: bool = Field(False, description="是否仅使用本地文件")


# ======================== 向量存储配置 ========================

class VectorStoreConfig(BaseModel):
    """向量存储配置模型。"""
    backend: str = Field("chroma", description="向量存储后端")
    collection_name: str = Field("chinese_rag_kb", description="集合名称")
    persist_directory: str = Field("data/chroma_db", description="持久化目录")
    distance_fn: str = Field("cosine", description="距离函数")
    top_k: int = Field(4, ge=1, le=100, description="默认 top-k")
    fetch_k: int = Field(20, ge=1, le=500, description="获取数量")
    
    @field_validator("distance_fn")
    @classmethod
    def validate_distance_fn(cls, v: str) -> str:
        allowed = {"cosine", "l2", "ip"}
        if v not in allowed:
            raise ValueError(f"distance_fn must be one of {allowed}")
        return v


# ======================== RAG 配置 ========================

class RAGConfig(BaseModel):
    """RAG 配置模型。"""
    top_k: int = Field(4, ge=1, le=50, description="检索 top-k")
    use_reranker: bool = Field(False, description="是否使用重排序")
    max_context_tokens: int = Field(2000, ge=100, le=16000, description="最大上下文 token 数")
    max_history_turns: int = Field(0, ge=0, le=10, description="对话历史轮数")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="相似度阈值")
    answer_citations: bool = Field(True, description="是否包含引用")
    return_sources: bool = Field(True, description="是否返回来源")
    fallback_answer: str = Field("抱歉，知识库中未找到相关信息，无法回答该问题。", description="默认回复")


# ======================== CORS 配置 ========================

class CORSConfig(BaseModel):
    """CORS 配置模型。"""
    allowed_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        description="允许的源列表",
    )
    allow_credentials: bool = Field(True, description="是否允许凭证")
    allow_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="允许的 HTTP 方法",
    )
    allow_headers: List[str] = Field(
        default_factory=lambda: ["Authorization", "Content-Type", "X-Request-ID"],
        description="允许的 HTTP 头",
    )


# ======================== 知识图谱配置 ========================

class KnowledgeGraphConfig(BaseModel):
    """知识图谱配置模型。"""
    enabled: bool = Field(True, description="是否启用知识图谱")
    backend: str = Field("sqlite", description="存储后端")
    sqlite_path: str = Field("data/kg.db", description="SQLite 数据库路径")
    max_entities_per_chunk: int = Field(20, ge=1, le=100, description="每块最大实体数")
    max_relations_per_chunk: int = Field(30, ge=1, le=100, description="每块最大关系数")


# ======================== 主配置模型 ========================

class ProjectConfig(BaseModel):
    """项目配置。"""
    name: str = Field("ChineseRAGKB", description="项目名称")
    version: str = Field("0.1.0", description="版本号")
    description: Optional[str] = Field(None, description="项目描述")


class PathsConfig(BaseModel):
    """路径配置。"""
    raw_docs: str = Field("data/raw", description="原始文档目录")
    chroma_db: str = Field("data/chroma_db", description="向量库目录")
    eval: str = Field("data/eval", description="评估目录")
    logs: str = Field("logs", description="日志目录")


class DocumentLoaderConfig(BaseModel):
    """文档加载器配置。"""
    supported_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".md", ".markdown", ".txt"],
        description="支持的文档扩展名",
    )
    pdf_engine: str = Field("pdfplumber", description="PDF 引擎")
    encoding: str = Field("utf-8", description="文本编码")
    clean_text: bool = Field(True, description="是否清洗文本")
    max_pdf_pages: int = Field(1000, ge=1, description="最大 PDF 页数")


class TextSplitterConfig(BaseModel):
    """文本分块配置。"""
    strategy: str = Field("chinese", description="分块策略")
    chunk_size: int = Field(300, ge=50, le=2000, description="块大小")
    chunk_overlap: int = Field(50, ge=0, le=500, description="块重叠")
    min_chunk_size: int = Field(50, ge=1, description="最小块大小")
    
    @model_validator(mode="after")
    def validate_overlap(self) -> "TextSplitterConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class RetrievalConfig(BaseModel):
    """检索配置。"""
    top_k: int = Field(4, ge=1, le=50, description="检索 top-k")
    rerank_candidates: int = Field(12, ge=1, le=100, description="重排候选数")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="相似度阈值")


class RerankerConfig(BaseModel):
    """重排序配置。"""
    enabled: bool = Field(False, description="是否启用")
    model_name: str = Field("BAAI/bge-reranker-base", description="模型名称")
    device: str = Field("cuda", description="设备类型")
    top_n: int = Field(3, ge=1, le=20, description="重排返回数")
    max_length: int = Field(512, ge=64, le=2048, description="最大长度")
    cache_dir: Optional[str] = Field(None, description="缓存目录")


class UIConfig(BaseModel):
    """UI 配置。"""
    title: str = Field("中文个人知识库 RAG", description="标题")
    page_icon: str = Field("📚", description="页面图标")
    theme: str = Field("light", description="主题")
    default_top_k: int = Field(4, ge=1, description="默认 top-k")
    default_temperature: float = Field(0.2, ge=0.0, le=2.0, description="默认温度")
    show_thinking: bool = Field(False, description="显示思考过程")
    max_upload_mb: int = Field(50, ge=1, description="最大上传大小（MB）")
    max_chat_history: int = Field(20, ge=0, description="最大对话历史")


class EvaluationConfig(BaseModel):
    """评估配置。"""
    retrieval_top_k: int = Field(5, ge=1, description="检索 top-k")
    judge_model: str = Field("", description="评判模型")
    dataset_path: str = Field("data/eval/eval_set.jsonl", description="数据集路径")
    output_path: str = Field("data/eval/results.json", description="输出路径")
    report_path: str = Field("data/eval/report.json", description="报告路径")
    metrics: List[str] = Field(
        default_factory=lambda: [
            "retrieval_hit_rate",
            "answer_keyword_containment",
            "avg_latency_sec",
        ],
        description="评估指标",
    )


class ConfigSchema(BaseModel):
    """主配置 Schema。"""
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    app: Dict[str, str] = Field(default_factory=dict)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    document_loader: DocumentLoaderConfig = Field(default_factory=DocumentLoaderConfig)
    text_splitter: TextSplitterConfig = Field(default_factory=TextSplitterConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    
    class Config:
        extra = "allow"  # 允许额外字段（向后兼容）
        validate_assignment = True
        str_strip_whitespace = True


def validate_config(config_dict: dict) -> ConfigSchema:
    """验证配置字典。
    
    Args:
        config_dict: 配置字典
        
    Returns:
        验证后的配置对象
        
    Raises:
        ValidationError: 配置验证失败时抛出
    """
    return ConfigSchema(**config_dict)


def load_and_validate_config(config_path: str = "config/config.yaml") -> ConfigSchema:
    """加载并验证配置文件。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        验证后的配置对象
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValidationError: 配置验证失败
    """
    from src.utils import load_config, apply_env_overrides, resolve_path
    
    path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件未找到: {path}")
    
    cfg = load_config(config_path)
    cfg = apply_env_overrides(cfg)
    
    return validate_config(cfg)


def validate_env_vars() -> List[str]:
    """验证必需的环境变量。
    
    Returns:
        缺失或无效的环境变量列表
    """
    errors = []
    
    # JWT 密钥检查
    jwt_secret = os.getenv("JWT_SECRET", "")
    if jwt_secret and len(jwt_secret) < 32:
        errors.append("JWT_SECRET should be at least 32 characters for security")
    
    # API Key 检查（如果有外部 LLM）
    if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
        errors.append("OPENAI_API_KEY should start with 'sk-'")
    
    return errors
