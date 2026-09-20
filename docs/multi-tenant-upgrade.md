# ChineseRAGKB 多租户升级方案

> 文档版本：v0.2.0  
> 编写日期：2026-07-18  
> 目标：将 ChineseRAGKB 从单用户个人知识库升级为支持 **100 并发用户**、每个用户拥有**独立知识库**的多租户平台。

---

## 目录

1. [Executive Summary（执行摘要）](#1-executive-summary执行摘要)
2. [Current Architecture Analysis（当前架构分析）](#2-current-architecture-analysis当前架构分析)
3. [Target Architecture（目标架构）](#3-target-architecture目标架构)
4. [Gap Analysis（差距分析）](#4-gap-analysis差距分析)
5. [Implementation Roadmap（实施路线图）](#5-implementation-roadmap实施路线图)
6. [API Changes（API 变更）](#6-api-changes-api-变更)
7. [Database Schema Changes（数据库 Schema 变更）](#7-database-schema-changes数据库-schema-变更)
8. [Security Considerations（安全注意事项）](#8-security-considerations安全注意事项)
9. [Performance Optimization Plan（性能优化计划）](#9-performance-optimization-plan性能优化计划)
10. [Testing Strategy（测试策略）](#10-testing-strategy测试策略)
11. [Rollout Plan（上线计划）](#11-rollout-plan上线计划)
12. [Appendix（附录）](#12-appendix附录)

---

## 1. Executive Summary（执行摘要）

> **PM 能力维度映射**：本节对应 **[第六维度：Technical Leadership & Business Value Translation（技术领导力与商业价值转化）]**  
> 
> 本节阐述了升级项目的战略愿景、目标状态和升级范围，体现了将复杂业务需求转化为可扩展 AI 架构的技术领导力，以及平衡技术创新与商业 ROI 的能力。

### 1.1 当前状态

ChineseRAGKB 是一个面向本地硬件优化的中文个人知识库 RAG 系统，基于 Python + FastAPI + Chroma + 本地 LLM 构建。系统具备以下特征：

| 维度 | 现状 |
|------|------|
| 架构模式 | 单用户、单实例、全局单例 |
| 认证机制 | **无任何认证**，CORS 开放到 `*` |
| 数据隔离 | 所有用户共享同一个 Chroma collection (`chinese_rag_kb`) 和同一个 KG SQLite 数据库 (`data/kg.db`) |
| 并发能力 | 同步 LLM 推理阻塞主线程，无请求队列，无连接池 |
| 部署方式 | 单进程运行，无 CI/CD，无迁移脚本 |

### 1.2 目标状态

在保持本地化、私有化部署特色的前提下，将系统升级为支持 **100 并发用户**的多租户平台：

| 维度 | 目标 |
|------|------|
| 用户规模 | 支持 100 个独立用户同时在线 |
| 数据隔离 | 每个用户拥有独立的 Chroma collection 和 KG SQLite 文件 |
| 认证体系 | JWT Bearer Token 认证，支持注册/登录/Token 刷新 |
| 并发处理 | 异步 LLM 推理 + 请求队列 + API 限流 |
| 配置灵活性 | 支持每个用户覆盖默认 LLM/Embedding 配置 |
| 可观测性 | 每个用户的 API 调用量、Token 消耗、查询延迟均可追踪 |

### 1.3 升级范围

本升级覆盖以下核心模块：

```
核心升级范围
├── 认证层：新增 JWT 认证中间件、用户注册/登录 API
├── 数据隔离层：改造 ChromaStore 支持多 collection，改造 KGStore 支持多 DB
├── 并发层：LLM 异步批推理、API 限流、请求队列
├── 配置层：用户级配置覆盖系统
├── 存储层：用户表设计、KG Schema 迁移
├── API 层：所有端点注入 user_id、API 版本化
├── 运营层：用量遥测、监控告警、CI/CD
└── 前端层：多知识库管理 UI、用户设置页、引导向导
```

---

## 2. Current Architecture Analysis（当前架构分析）

> **PM 能力维度映射**：本节对应 **[第四维度：System Design & High-Performance Inference（系统设计与高性能推理）]**  
>
> 本节深入分析了当前系统的技术架构，识别出同步 LLM 推理、全局单例等性能瓶颈，为后续的异步架构改造和高性能推理优化奠定基础。

### 2.1 关键文件解读

#### 2.1.1 `api/main.py` — FastAPI 应用入口

```python
# 当前架构核心问题：所有组件均为模块级全局单例

_pipeline: Optional[RAGPipeline] = None          # 行 79，全局单例
_embedding: Optional[EmbeddingModel] = None      # 行 80
_vector_store: Optional[ChromaStore] = None      # 行 81
_kg_store = None                                  # 行 84
_kg_retriever: Optional[GraphRetriever] = None   # 行 85
_agent: Optional[Any] = None                      # 行 82-87
```

所有端点通过 `get_pipeline()`、`get_vector_store()` 等全局函数获取这些单例。无任何用户上下文传递机制。

**关键端点一览**：

| 端点 | 方法 | 功能 | 用户感知 |
|------|------|------|----------|
| `/api/health` | GET | 健康检查，返回全局 chunk 数量 | 无用户隔离 |
| `/api/documents` | GET | 列出已入库文档 | 全局列表 |
| `/api/search` | POST | 语义检索 | 全局检索 |
| `/api/chat` | POST | 同步问答 | 无用户上下文 |
| `/ws/chat` | WS | WebSocket 流式问答 | 无用户上下文 |
| `/api/ingest` | POST | 上传文件入库 | 无用户上下文 |
| `/api/kg/*` | 多 | 知识图谱操作 | 全局 KG |
| `/api/agent/*` | 多 | Agent 问答 | 全局 Agent |

**CORS 配置**（行 70-76）：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ← 开放到所有来源，生产环境极度危险
    allow_credentials=True,   # ← 允许携带凭证
    allow_methods=["*"],       # ← 所有 HTTP 方法
    allow_headers=["*"],       # ← 所有请求头
)
```

#### 2.1.2 `config/config.yaml` — 配置模型

```yaml
vector_store:
  collection_name: "chinese_rag_kb"    # ← 硬编码单一 collection 名
  persist_directory: "data/chroma_db"   # ← 全局共享目录

knowledge_graph:
  sqlite_path: data/kg.db               # ← 全局单一 KG 数据库文件
```

#### 2.1.3 `src/vector_store.py` — ChromaStore

```python
class ChromaStore:
    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "chinese_rag_kb",  # ← 硬编码默认值
        embedding_model: Optional[EmbeddingModel] = None,
        distance_fn: str = "cosine",
    ) -> None:
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,   # ← 使用硬编码名称
            metadata=metadata_cfg,
        )
```

**问题**：没有 `user_id` 参数，所有用户的向量数据写入同一个 collection。

#### 2.1.4 `src/kg/store.py` — KG SQLite Store

```python
class SQLiteGraphStore(KGStore):
    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path))
        # ← 使用固定路径 data/kg.db，无 user_id 字段
    
    def _init_db(self) -> None:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,    # ← 无 user_id 列
                type TEXT,
                description TEXT,
                ...
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,              # ← 无 user_id 列
                target TEXT,
                type TEXT,
                ...
            )
        """)
```

#### 2.1.5 `src/llm.py` — LocalLLM

```python
class LocalLLM:
    def _generate(self, prompt: str, gen_cfg: GenerationConfig) -> str:
        # ← 同步阻塞调用，blocking 事件循环
        with torch.no_grad():
            output_ids = self.model.generate(...)  # ← GPU/CPU 同步推理
        return text.strip()
    
    def _stream_generate(self, prompt: str, gen_cfg: GenerationConfig):
        # ← 基于 Thread 的流式输出，不是真正的异步
        th = Thread(target=_worker, daemon=True)
        th.start()
        for piece in streamer:
            yield piece
        th.join()
```

**问题**：`_stream_generate` 使用 `Thread` 而非真正的异步，在 100 并发场景下会耗尽线程资源。

#### 2.1.6 `src/embeddings.py` — EmbeddingModel

```python
class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", ...) -> None:
        # ← 同步 encode 方法
        def encode(self, texts: Sequence[str], ...):
            with Timer(f"embedding encode x{len(texts)}"):
                vecs = self.model.encode(...)  # ← 同步调用
            return vecs
```

#### 2.1.7 `src/rag_pipeline.py` — RAGPipeline

```python
class RAGPipeline:
    def __init__(self, config: Dict, embedding: EmbeddingModel,
                 vector_store: ChromaStore, llm: LocalLLM, ...) -> None:
        self.config = config
        self.vector_store = vector_store  # ← 全局共享
        self.llm = llm                    # ← 全局单例 LLM
```

### 2.2 当前数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                      当前单用户架构                                │
│                                                                 │
│  用户浏览器                                                       │
│      │                                                           │
│      ▼                                                           │
│  FastAPI (api/main.py)                                          │
│      │                                                           │
│      ├─ get_pipeline() ─────► 全局单例 RAGPipeline ──────────┐   │
│      │                              │                          │   │
│      │                              ▼                          │   │
│      │                         ChromaStore                     │   │
│      │                    collection: chinese_rag_kb            │   │
│      │                    persist_dir: data/chroma_db           │   │
│      │                              │                          │   │
│      │                              ▼                          │   │
│      │                         SQLiteGraphStore                 │   │
│      │                      db_path: data/kg.db                │   │
│      │                              │                          │   │
│      │                              ▼                          │   │
│      │                         LocalLLM                        │   │
│      │                   (Qwen2.5-1.5B-Instruct)              │   │
│      │                   同步阻塞推理                           │   │
│      │                              │                          │   │
│      └──────────────────────────────┘                          │   │
└─────────────────────────────────────────────────────────────────┘

问题：
  ❌ 所有用户共享同一份 Chroma collection 和 KG DB
  ❌ 没有任何用户身份标识
  ❌ LLM 同步阻塞，多用户并发时会排队等待
  ❌ CORS 完全开放
  ❌ 没有认证，没有限流
```

### 2.3 当前架构总结

| 组件 | 当前实现 | 适用场景 |
|------|----------|----------|
| API | FastAPI + 全局单例 | 单用户本地开发 |
| 认证 | 无 | — |
| 向量存储 | Chroma 单 collection | 单用户 |
| KG 存储 | SQLiteGraphStore 单 DB | 单用户 |
| LLM | 同步 transformers | 单用户 |
| Embedding | 同步 sentence-transformers | 单用户 |
| CORS | `allow_origins=["*"]` | 开发调试 |

---

## 3. Target Architecture（目标架构）

> **PM 能力维度映射**：本节对应 **[第二维度：AI Agent Engineering & Multi-Agent Orchestration（AI Agent 工程与多智能体编排）]**  
>
> 本节设计了 TenantAwareFactory 工厂模式和 Multi-Tenant RAGPipeline，体现了构建自主 AI Agent、配备任务规划、内存管理和工具调用能力的专业素养，以及编排复杂多智能体协作系统的能力。

### 3.1 多租户核心设计原则

1. **数据隔离优先**：每个用户的向量数据和 KG 数据物理隔离
2. **向后兼容**：保持单用户部署模式可正常工作
3. **渐进式升级**：不要求一次性全部重构，按阶段交付
4. **资源可控**：100 用户目标下，GPU 显存和内存占用需要精细管理

### 3.2 用户身份模型

```
用户身份体系
├── 用户注册：username / email + 密码（bcrypt hash）
├── JWT Token：HS256 算法，payload 包含 user_id, exp, iat
│   ├── Access Token：有效期 30 分钟
│   └── Refresh Token：有效期 7 天，存 Redis 或 DB
├── 每个请求通过 Authorization: Bearer <token> 头传递 user_id
└── API 层从 JWT 中提取 user_id，注入到下游所有组件
```

### 3.3 多租户数据隔离策略

#### 策略 A：Per-User Chroma Collections（推荐用于向量存储）

```
ChromaDB 目录结构
data/
└── chroma_db/
    ├── 164e2f8c_chinese_rag_kb/      # user_id hash 前8位 _ collection_name
    ├── 9a3b1d7e_chinese_rag_kb/
    └── ...

Collection 命名规则：{user_id_hash_8chars}_{base_collection_name}
示例：
  - 原始 collection_name: chinese_rag_kb
  - 用户 user_id: 164e2f8c-3a1b-4c9d-8e5f-6a7b8c9d0e1f
  - 最终 collection 名: 164e2f8c_chinese_rag_kb
```

**优点**：
- Chroma 的 collection 天然支持 HNSW 独立索引，不同用户的数据不会互相干扰
- 删除用户只需 drop 整个 collection，无需数据迁移
- 查询时只需指定 collection 名，无需额外过滤条件

**缺点**：
- 100 个用户 = 100 个 collection，Chroma 的 collection 列表管理有轻微开销（可接受）

#### 策略 B：Per-User KG SQLite（推荐用于知识图谱）

```
KG SQLite 文件结构
data/
├── kg_164e2f8c.db     # 用户 user_id=164e2f8c 的 KG 数据库
├── kg_9a3b1d7e.db     # 用户 user_id=9a3b1d7e 的 KG 数据库
└── ...

KG Schema 新增 user_id 列：
  entities(user_id, name, type, description, aliases, attributes, created_at)
  relations(user_id, id, source, target, type, description, ...)
```

**优点**：
- SQLite 文件级隔离，数据完全独立
- 备份/恢复按用户粒度操作
- 删除用户只需删除文件

**缺点**：
- 100 个用户 = 100 个 SQLite 连接，需要连接池管理

### 3.4 目标架构数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│                      多租户目标架构                                     │
│                                                                      │
│  用户 A 浏览器                          用户 B 浏览器                    │
│      │                                    │                          │
│      ▼                                    ▼                          │
│  JWT 认证                                 JWT 认证                      │
│  user_id=A_xxx                            user_id=B_yyy                │
│      │                                    │                          │
│      ▼                                    ▼                          │
│  FastAPI (api/main.py)                   FastAPI (同一进程)           │
│      │                                    │                          │
│      ▼                                    ▼                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  TenantContext (请求上下文，存储当前 user_id)                    │    │
│  │  请求级别对象，从 JWT 中提取，不共享                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│      │                                                               │
│      ├─────────────────────────────────────────────────────────┐     │
│      │                                                           │     │
│      ▼                                                           ▼     │
│  ChromaStore                          ChromaStore                 │
│  collection: A_xxx_chinese_rag_kb     collection: B_yyy_chinese_rag_kb │
│      │                                   │                        │
│      ▼                                   ▼                        │
│  SQLiteGraphStore                     SQLiteGraphStore             │
│  db_path: data/kg_A_xxx.db           db_path: data/kg_B_yyy.db   │
│      │                                   │                        │
│      │           LLM Request Queue        │                        │
│      │      ┌──────────────────────┐    │                        │
│      └──────►  Async LLM Executor  ◄─────┘                        │
│               │  (100 并发控制)     │                               │
│               └──────────────────────┘                               │
│                      │                                              │
│                      ▼                                              │
│               LocalLLM Pool                                        │
│         (共享 LLM 实例，按序/分批处理)                               │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.5 组件工厂模式

为支持多租户，核心组件需要改造为工厂模式：

```python
# src/factories.py（新增）

from functools import lru_cache
from typing import Dict, Optional
from threading import Lock

class TenantAwareFactory:
    """多租户组件工厂。"""
    
    _lock = Lock()
    _chroma_stores: Dict[str, "ChromaStore"] = {}
    _kg_stores: Dict[str, "KGStore"] = {}
    _pipelines: Dict[str, "RAGPipeline"] = {}
    _embedding: Optional["EmbeddingModel"] = None  # 全局共享，只加载一次
    _llm_executor: Optional["AsyncLLMExecutor"] = None  # 全局 LLM 队列
    
    @classmethod
    def get_chroma_store(cls, user_id: str, config: dict) -> "ChromaStore":
        """获取或创建用户专属的 ChromaStore。"""
        collection_name = f"{user_id}_{config['vector_store']['collection_name']}"
        persist_dir = config["vector_store"]["persist_directory"]
        
        if collection_name not in cls._chroma_stores:
            with cls._lock:
                if collection_name not in cls._chroma_stores:
                    embed = cls.get_embedding(config)
                    cls._chroma_stores[collection_name] = ChromaStore(
                        persist_directory=persist_dir,
                        collection_name=collection_name,
                        embedding_model=embed,
                        distance_fn=config["vector_store"].get("distance_fn", "cosine"),
                    )
        return cls._chroma_stores[collection_name]
    
    @classmethod
    def get_kg_store(cls, user_id: str, config: dict) -> "KGStore":
        """获取或创建用户专属的 KG SQLite Store。"""
        db_path = f"data/kg_{user_id}.db"
        return SQLiteGraphStore(db_path=db_path)
    
    @classmethod
    def get_embedding(cls, config: dict) -> "EmbeddingModel":
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
    def get_llm_executor(cls, config: dict) -> "AsyncLLMExecutor":
        """获取全局 LLM 异步执行器（按用户排队）。"""
        if cls._llm_executor is None:
            with cls._lock:
                if cls._llm_executor is None:
                    cls._llm_executor = AsyncLLMExecutor(
                        model_name=config["llm"]["model_name"],
                        device=config["llm"].get("device", "auto"),
                        max_concurrent=10,  # GPU 最大并发数
                        queue_size=200,     # 请求队列上限
                    )
        return cls._llm_executor
    
    @classmethod
    def get_user_pipeline(cls, user_id: str, config: dict) -> "RAGPipeline":
        """获取用户专属的 RAGPipeline。"""
        pipeline_key = f"pipeline_{user_id}"
        if pipeline_key not in cls._pipelines:
            with cls._lock:
                if pipeline_key not in cls._pipelines:
                    chroma = cls.get_chroma_store(user_id, config)
                    kg = cls.get_kg_store(user_id, config)
                    executor = cls.get_llm_executor(config)
                    cls._pipelines[pipeline_key] = MultiTenantRAGPipeline(
                        config=config,
                        vector_store=chroma,
                        kg_store=kg,
                        llm_executor=executor,
                        user_id=user_id,
                    )
        return cls._pipelines[pipeline_key]
    
    @classmethod
    def cleanup_user(cls, user_id: str):
        """清理用户相关资源（删除 collection 和 KG DB）。"""
        # 删除 Chroma collection
        for key in list(cls._chroma_stores.keys()):
            if key.startswith(f"{user_id}_"):
                del cls._chroma_stores[key]
        
        # 删除 KG SQLite 文件
        kg_db = Path(f"data/kg_{user_id}.db")
        if kg_db.exists():
            kg_db.unlink()
        
        # 删除 pipeline
        pipeline_key = f"pipeline_{user_id}"
        if pipeline_key in cls._pipelines:
            del cls._pipelines[pipeline_key]
```

### 3.6 用户配置覆盖系统

每个用户可以拥有自己的配置覆盖项：

```python
# 用户配置优先级：用户个人设置 > 用户订阅层默认值 > 系统默认配置
# 用户可覆盖的字段（白名单）：
OVERRIDEABLE_FIELDS = [
    "llm.model_name",
    "llm.generation.temperature",
    "llm.generation.max_new_tokens",
    "embedding.model_name",
    "rag.top_k",
    "rag.max_context_tokens",
    "reranker.enabled",
]

# 用户订阅层限制（超出此限制的配置会被拒绝）：
SUBSCRIPTION_TIERS = {
    "free": {
        "max_chunks": 1000,
        "max_queries_per_day": 50,
        "available_models": ["Qwen2.5-0.5B-Instruct"],
        "reranker": False,
    },
    "pro": {
        "max_chunks": 50000,
        "max_queries_per_day": 5000,
        "available_models": ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct-GPTQ-Int4"],
        "reranker": True,
    },
    "enterprise": {
        "max_chunks": -1,  # 无限制
        "max_queries_per_day": -1,
        "available_models": ["*"],  # 所有模型
        "reranker": True,
    },
}
```

---

## 4. Gap Analysis（差距分析）

> **PM 能力维度映射**：本节对应 **[第六维度：Technical Leadership & Business Value Translation（技术领导力与商业价值转化）]**  
>
> 本节通过系统性的差距分析，展示了将复杂业务需求（100 并发用户、数据隔离）转化为技术架构决策的能力，体现了卓越的系统思维和问题分解能力。

### 4.1 认证与安全

| 维度 | 当前（Before） | 目标（After） | 严重程度 |
|------|---------------|-------------|---------|
| 用户认证 | **完全缺失**，任何人都可以调用所有 API | JWT Bearer Token，Access Token 30min + Refresh Token 7d | 🔴 阻断 |
| CORS 策略 | `allow_origins=["*"]`，无任何来源限制 | 白名单 CORS，支持配置允许的域名 | 🔴 阻断 |
| API 限流 | **完全缺失**，无任何速率限制 | 每用户每分钟/每天限流，免费用户 50次/天，专业用户 5000次/天 | 🔴 阻断 |
| 密码存储 | **缺失**，无用户注册/登录机制 | bcrypt 哈希 + 加盐，支持 Argon2 备选 | 🔴 阻断 |
| 输入验证 | 仅部分端点有 Pydantic 验证 | 全链路输入校验，防 XSS/SQL 注入/文件上传攻击 | 🟡 高 |
| 敏感信息 | `.env` 支持但未实际使用 | 强制使用 .env 存储 JWT Secret、DB 密码等 | 🟡 高 |
| 文件上传安全 | 无限制，上传后直接写入 `data/raw` | 文件类型白名单、大小限制、病毒扫描（可选）、隔离目录 | 🟡 高 |

### 4.2 数据隔离

| 维度 | 当前（Before） | 目标（After） | 严重程度 |
|------|---------------|-------------|---------|
| Chroma 隔离 | 单一 `chinese_rag_kb` collection，**所有用户数据混合** | Per-User collection：`{user_id_hash}_{collection_name}` | 🔴 阻断 |
| KG SQLite 隔离 | 单一 `data/kg.db`，entities/relations 表**无 user_id 列** | Per-User DB：`data/kg_{user_id}.db`，或加 user_id 列 | 🔴 阻断 |
| 原始文档隔离 | 上传文件到共享目录 `data/raw/` | 用户专属子目录：`data/raw/{user_id}/` | 🟡 高 |
| 配置隔离 | 全局 `config/config.yaml`，所有用户共享同一配置 | 用户级配置覆盖，支持 per-user LLM/Embedding 选型 | 🟠 中 |
| 评估数据隔离 | 共享 `data/eval` 目录 | 用户专属评估报告 | 🟠 中 |
| 全局单例问题 | `_pipeline`、`_vector_store` 等全局变量无 user_id 上下文 | 改造为工厂模式，按 user_id 动态创建实例 | 🔴 阻断 |

### 4.3 并发与性能

| 维度 | 当前（Before） | 目标（After） | 严重程度 |
|------|---------------|-------------|---------|
| LLM 推理模式 | **同步阻塞**，`model.generate()` 在主线程执行 | 异步推理 + 请求队列，支持 100 并发请求排队 | 🔴 阻断 |
| 线程模型 | `Thread` daemon 实现流式（`src/llm.py` 行 273），非真正异步 | 真正的 `asyncio` 异步或进程池隔离 | 🔴 阻断 |
| Embedding 批处理 | 同步调用，无批处理优化 | 异步批处理，相同时间段内的 query 合并 embedding 请求 | 🟡 高 |
| WebSocket 并发 | 同步 `websocket.receive_json()` + 同步 pipeline 调用 | 每个 WS 连接对应独立协程，LLM 推理走异步队列 | 🟡 高 |
| Chroma 连接 | 每个 ChromaStore 实例持有一个 `PersistentClient` | 共享 `chromadb.Client` 实例，按 collection 隔离 | 🟠 中 |
| 内存管理 | 全局单例在进程生命周期内常驻 | LRU 缓存 + LFU 淘汰策略，超出上限的 user pipeline 释放 | 🟠 中 |

### 4.4 可扩展性与运维

| 维度 | 当前（Before） | 目标（After） | 严重程度 |
|------|---------------|-------------|---------|
| 状态管理 | 模块级全局变量（`api/main.py` 行 79-87） | 无状态 API，状态存在 DB 或 Redis | 🔴 阻断 |
| 后台任务 | **无** 任何后台任务系统 | Celery / FastAPI BackgroundTasks，支持 KG 批量构建 | 🟡 高 |
| CI/CD | **缺失**，手动部署 | GitHub Actions 自动构建 + 测试 + 部署 | 🟡 高 |
| 数据库迁移 | **缺失**，无 Alembic 或迁移脚本 | Alembic 管理 KG Schema 迁移 | 🟡 高 |
| 环境隔离 | 仅支持通过 `.env` 覆盖部分配置 | dev/staging/prod 三环境配置 | 🟠 中 |
| 监控告警 | `get_logger` 写日志文件，无指标收集 | Prometheus + Grafana，API 延迟、Token 消耗、错误率告警 | 🟠 中 |

### 4.5 用户体验与产品

| 维度 | 当前（Before） | 目标（After） | 严重程度 |
|------|---------------|-------------|---------|
| 用户引导 | **缺失**，无注册/登录/知识库创建流程 | 引导向导：注册 → 创建知识库 → 上传文档 → 开始问答 | 🟡 高 |
| 多知识库管理 | **缺失**，只有单一知识库 | 支持创建/切换多个知识库，每个知识库独立配置 | 🟡 高 |
| 用户设置页 | **缺失** | 用户设置：修改密码、查看用量、重置 API Key、切换主题 | 🟠 中 |
| 错误页面 | HTTP 500/404 等原生错误 | 友好的错误页面（未认证、无权限、知识库为空等） | 🟠 中 |
| 暗色主题 | Streamlit 原生支持 | 前端 UI 暗色主题适配（移动端 + 桌面） | 🟢 低 |
| 订阅管理 | **缺失** | 定价页、订阅状态展示、用量进度条 | 🟢 低 |

### 4.6 差距总结表

| 分类 | 阻断项 | 高优先级 | 中优先级 | 低优先级 |
|------|--------|----------|----------|----------|
| 认证与安全 | JWT、CORS 修复、API 限流、密码存储 | 输入验证、敏感信息 | 文件上传安全 | — |
| 数据隔离 | Chroma collection 隔离、KG DB 隔离、全局单例改造 | 文档目录隔离 | 配置隔离、评估数据隔离 | — |
| 并发与性能 | 异步 LLM 推理、真正的异步架构 | Embedding 批处理、WebSocket 并发 | 内存管理、Chroma 连接 | — |
| 可扩展性 | 无状态改造 | 后台任务系统、CI/CD | 数据库迁移、监控告警 | 环境隔离 |
| 用户体验 | 用户引导 | 多知识库管理 | 用户设置页、错误页面 | 暗色主题、订阅管理 |
| **合计** | **9 项** | **7 项** | **10 项** | **3 项** |

---

## 5. Implementation Roadmap（实施路线图）

> **PM 能力维度映射**：本节对应 **[第六维度：Technical Leadership & Business Value Translation（技术领导力与商业价值转化）]**  
>
> 本节制定了 4 阶段 14 周的详细实施路线图，展示了将复杂项目分解为可管理阶段的能力，体现了平衡技术创新与商业 ROI 的战略思维，以及领导跨职能团队交付 AI 原生产品的能力。

### 5.1 整体时间线

```
Week  1  2  3  4  5  6  7  8  9 10 11 12 13 14
      ├───┤
      Phase 1: Foundation
            ├───┤
            Phase 2: Multi-Tenant Core
                        ├───┤
                        Phase 3: UX & Polish
                                    ├───┤
                                    Phase 4: Business & Operations
```

### 5.2 Phase 1: Foundation（第 1-3 周）

**目标**：建立多租户基础设施，使系统具备用户隔离能力。

#### Week 1: 认证骨架 + 用户管理

**任务**：

1. **用户数据库设计**
   - 新建 `src/db/models.py`（使用 SQLAlchemy）：
   ```python
   from sqlalchemy import Column, String, Boolean, DateTime, Integer
   from sqlalchemy.orm import declarative_base
   Base = declarative_base()
   
   class User(Base):
       __tablename__ = "users"
       id = Column(String(36), primary_key=True)  # UUID
       username = Column(String(50), unique=True, nullable=False, index=True)
       email = Column(String(255), unique=True, nullable=False, index=True)
       password_hash = Column(String(255), nullable=False)
       subscription_tier = Column(String(20), default="free")  # free / pro / enterprise
       max_chunks = Column(Integer, default=1000)
       queries_per_day_limit = Column(Integer, default=50)
       is_active = Column(Boolean, default=True)
       created_at = Column(DateTime, default=datetime.utcnow)
       updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
   ```

2. **JWT 认证工具**（`src/auth/jwt_handler.py`）：
   ```python
   from datetime import datetime, timedelta
   from typing import Optional
   import jwt
   
   SECRET_KEY = os.getenv("JWT_SECRET", "change-me-in-production")
   ALGORITHM = "HS256"
   ACCESS_TOKEN_EXPIRE_MINUTES = 30
   REFRESH_TOKEN_EXPIRE_DAYS = 7
   
   def create_access_token(user_id: str) -> str:
       expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
       payload = {"sub": user_id, "exp": expire, "iat": datetime.utcnow(), "type": "access"}
       return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
   
   def create_refresh_token(user_id: str) -> str:
       expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
       payload = {"sub": user_id, "exp": expire, "iat": datetime.utcnow(), "type": "refresh"}
       return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
   
   def decode_token(token: str) -> Optional[dict]:
       try:
           return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
       except jwt.ExpiredSignatureError:
           return None
   ```

3. **注册/登录 API 端点**（`api/routes/auth.py`）：
   - `POST /api/v1/auth/register`：用户名 + 邮箱 + 密码 → bcrypt 哈希 → 写 DB → 返回 user_id
   - `POST /api/v1/auth/login`：用户名/邮箱 + 密码 → 验证 → 返回 Access Token + Refresh Token
   - `POST /api/v1/auth/refresh`：Refresh Token → 返回新的 Access Token
   - `GET /api/v1/auth/me`：获取当前用户信息

4. **依赖安装**：
   ```bash
   pip install sqlalchemy bcrypt pyjwt python-jose[cryptography passlib[bcrypt]
   ```

**交付物**：用户可注册账号、登录并获取 JWT Token。

#### Week 2: user_id 注入 + Chroma 多 Collection 隔离

**任务**：

1. **JWT 中间件**（`src/middleware/auth.py`）：
   ```python
   from fastapi import Depends, HTTPException, status
   from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
   
   security = HTTPBearer()
   
   async def get_current_user(
       credentials: HTTPAuthorizationCredentials = Depends(security),
       db: Session = Depends(get_db),
   ) -> User:
       token = credentials.credentials
       payload = decode_token(token)
       if payload is None:
           raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 已过期")
       if payload.get("type") != "access":
           raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效 Token 类型")
       
       user = db.query(User).filter(User.id == payload["sub"]).first()
       if user is None or not user.is_active:
           raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在或已禁用")
       return user
   ```

2. **改造 `ChromaStore` 工厂**（`src/factories.py`）：
   ```python
   # 关键改造：collection_name 包含 user_id
   def get_chroma_store(user_id: str, base_config: dict) -> ChromaStore:
       collection_name = f"{user_id}_{base_config['vector_store']['collection_name']}"
       # 后续 ChromaStore.query() / add_chunks() 自动带上 user_id 隔离
   ```

3. **改造所有 API 端点注入 user_id**：
   - 所有端点增加 `current_user: User = Depends(get_current_user)` 依赖
   - `get_vector_store()` 改为 `get_vector_store(current_user.id)`
   - `get_kg_store()` 改为 `get_kg_store(current_user.id)`

4. **端点改造示例**（`api/main.py` 改造）：
   ```python
   # 改造前
   @app.post("/api/search")
   async def search_documents(body: SearchBody):
       store = get_vector_store()
       hits = store.query(query_text=body.query, top_k=body.top_k)
   
   # 改造后
   @app.post("/api/v1/search")
   async def search_documents(
       body: SearchBody,
       current_user: User = Depends(get_current_user),
   ):
       config = load_config("config/config.yaml")
       store = get_chroma_store(current_user.id, config)
       hits = store.query(query_text=body.query, top_k=body.top_k)
   ```

**交付物**：所有 API 需要认证，不同用户查询到不同的向量数据。

#### Week 3: KG Schema 迁移 + CI/CD 初始化

**任务**：

1. **KG SQLite Schema 迁移**（两种方案任选其一）：

   **方案 A：Per-User DB（推荐，简单直接）**
   ```python
   # data/kg_{user_id}.db，每个用户独立文件
   # Schema 保持不变，只需在创建时使用 user_id 命名
   
   def create_user_kg_store(user_id: str) -> SQLiteGraphStore:
       db_path = f"data/kg_{user_id}.db"
       return SQLiteGraphStore(db_path=db_path)
   ```

   **方案 B：Schema 迁移（添加 user_id 列）**
   ```sql
   -- 迁移脚本 v001_add_user_id.sql
   ALTER TABLE entities ADD COLUMN user_id TEXT NOT NULL DEFAULT '';
   ALTER TABLE relations ADD COLUMN user_id TEXT NOT NULL DEFAULT '';
   CREATE INDEX idx_entities_user_id ON entities(user_id);
   CREATE INDEX idx_relations_user_id ON relations(user_id);
   ```

2. **Alembic 迁移设置**：
   ```bash
   pip install alembic
   alembic init migrations
   ```

3. **GitHub Actions CI/CD 流水线**（`.github/workflows/ci.yml`）：
   ```yaml
   name: CI
   
   on:
     push:
       branches: [main, develop]
     pull_request:
       branches: [main]
   
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: '3.10'
         - name: Install dependencies
           run: pip install -r requirements.txt
         - name: Run tests
           run: pytest tests/ -v --cov=src
         - name: Run lint
           run: |
             pip install ruff
             ruff check src/ api/
   ```

**交付物**：KG 数据按用户隔离，CI/CD 流水线可用。

### 5.3 Phase 2: Multi-Tenant Core（第 4-6 周）

**目标**：实现异步并发处理、API 限流和用量追踪。

#### Week 4: 异步 LLM Executor + 请求队列

**任务**：

1. **AsyncLLMExecutor 实现**（`src/llm_async.py`）：
   ```python
   import asyncio
   from asyncio import Queue, LifoQueue
   from typing import List, Generator, Optional
   import threading
   from concurrent.futures import ThreadPoolExecutor
   
   class AsyncLLMExecutor:
       """异步 LLM 执行器，支持并发控制和请求队列。"""
       
       def __init__(
           self,
           model_name: str,
           device: str = "auto",
           max_concurrent: int = 5,  # GPU 最大并发数
           queue_size: int = 200,    # 队列上限
       ):
           self.model_name = model_name
           self.max_concurrent = max_concurrent
           self._semaphore = asyncio.Semaphore(max_concurrent)
           self._queue: Queue = Queue(maxsize=queue_size)
           self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
           self._llm: Optional[LocalLLM] = None
           self._lock = threading.Lock()
       
       def _get_llm(self) -> LocalLLM:
           if self._llm is None:
               with self._lock:
                   if self._llm is None:
                       self._llm = LocalLLM(model_name=self.model_name, device=self.device)
           return self._llm
       
       async def chat_async(
           self,
           messages: List[dict],
           generation_config: GenerationConfig,
       ) -> str:
           """异步 chat，在队列中等待，获得信号量后执行。"""
           async with self._semaphore:
               loop = asyncio.get_event_loop()
               llm = self._get_llm()
               result = await loop.run_in_executor(
                   self._executor,
                   lambda: llm.chat(messages, generation_config, stream=False)
               )
               return result
       
       async def stream_chat_async(
           self,
           messages: List[dict],
           generation_config: GenerationConfig,
       ) -> Generator[str, None, None]:
           """异步流式 chat。"""
           async with self._semaphore:
               loop = asyncio.get_event_loop()
               llm = self._get_llm()
               
               def _sync_stream():
                   return list(llm.chat(messages, generation_config, stream=True))
               
               tokens = await loop.run_in_executor(self._executor, _sync_stream)
               for token in tokens:
                   yield token
   ```

2. **改造 WebSocket 端点**：
   ```python
   @app.websocket("/ws/chat")
   async def websocket_chat(websocket: WebSocket, token: str = Query(...)):
       # 1. 验证 JWT Token
       # 2. 获取 user_id
       # 3. 使用 AsyncLLMExecutor 异步处理
       async for event in pipeline.astream_answer_async(query, top_k=top_k, user_id=user_id):
           await websocket.send_json(event)
   ```

3. **按用户限流**（`src/middleware/rate_limit.py`）：
   ```python
   from fastapi import HTTPException, status
   from collections import defaultdict
   import time
   
   class RateLimiter:
       """简单的内存限流器（生产环境建议使用 Redis）。"""
       
       def __init__(self):
           self._requests: Dict[str, List[float]] = defaultdict(list)
       
       def check(self, user_id: str, limit: int, window_seconds: int = 60) -> bool:
           now = time.time()
           # 清理过期记录
           self._requests[user_id] = [
               t for t in self._requests[user_id] if now - t < window_seconds
           ]
           if len(self._requests[user_id]) >= limit:
               return False
           self._requests[user_id].append(now)
           return True
   
   # 在中间件中使用
   rate_limiter = RateLimiter()
   
   async def rate_limit_dependency(
       current_user: User = Depends(get_current_user),
   ):
       tier = SUBSCRIPTION_TIERS.get(current_user.subscription_tier, SUBSCRIPTION_TIERS["free"])
       daily_limit = tier["max_queries_per_day"]
       if not rate_limiter.check(current_user.id, daily_limit, window_seconds=86400):
           raise HTTPException(
               status_code=status.HTTP_429_TOO_MANY_REQUESTS,
               detail=f"今日查询次数已用尽（{daily_limit}次/天）"
           )
   ```

**交付物**：100 并发请求不会耗尽系统资源，LLM 推理有序排队。

#### Week 5: 用量遥测系统

**任务**：

1. **用量记录表**（`src/db/models.py` 扩展）：
   ```python
   class UsageLog(Base):
       __tablename__ = "usage_logs"
       id = Column(String(36), primary_key=True)
       user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
       endpoint = Column(String(100), nullable=False)
       tokens_used = Column(Integer, default=0)
       latency_ms = Column(Integer, default=0)
       chunks_accessed = Column(Integer, default=0)
       timestamp = Column(DateTime, default=datetime.utcnow, index=True)
   ```

2. **用量记录装饰器**（`src/middleware/telemetry.py`）：
   ```python
   async def log_usage(endpoint: str, user_id: str, tokens: int, latency: int):
       log = UsageLog(
           user_id=user_id,
           endpoint=endpoint,
           tokens_used=tokens,
           latency_ms=latency,
       )
       db = SessionLocal()
       db.add(log)
       db.commit()
   ```

3. **用量查询 API**：
   - `GET /api/v1/usage/summary`：获取当日/当月用量摘要
   - `GET /api/v1/usage/history`：获取用量历史趋势

**交付物**：每个用户的 API 调用量、Token 消耗可追踪和展示。

#### Week 6: Per-User 配置覆盖系统

**任务**：

1. **用户配置表**（`src/db/models.py` 扩展）：
   ```python
   class UserConfig(Base):
       __tablename__ = "user_configs"
       id = Column(String(36), primary_key=True)
       user_id = Column(String(36), ForeignKey("users.id"), unique=True, nullable=False)
       config_json = Column(Text)  # JSON 存储用户覆盖配置
       updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
   ```

2. **配置合并逻辑**：
   ```python
   def get_user_config(user_id: str, db: Session) -> dict:
       user = db.query(User).filter(User.id == user_id).first()
       base = load_config("config/config.yaml")
       tier_limits = SUBSCRIPTION_TIERS[user.subscription_tier]
       
       if user.config:
           overrides = json.loads(user.config.config_json)
           base = merge_dict(base, overrides)
       
       # 应用订阅层限制
       base["limits"] = tier_limits
       return base
   ```

3. **用户设置 API**：
   - `GET /api/v1/config`：获取当前用户的有效配置
   - `PUT /api/v1/config`：更新用户配置（仅允许白名单字段）
   - `GET /api/v1/config/limits`：获取订阅层限制

**交付物**：用户可自定义部分配置（LLM 型号、温度等），超出订阅限制的配置被拒绝。

### 5.4 Phase 3: UX & Polish（第 7-10 周）

**目标**：完善前端交互体验。

#### Week 7-8: 多知识库管理 UI

**任务**：

1. **知识库概念引入**：
   - 每个用户可以创建多个 KnowledgeBase（知识库）
   - 每个 KnowledgeBase 有独立的 Chroma collection 和 KG SQLite
   - 知识库可共享给其他用户（可选功能，Phase 4 再实现）

2. **知识库数据模型**：
   ```python
   class KnowledgeBase(Base):
       __tablename__ = "knowledge_bases"
       id = Column(String(36), primary_key=True)
       user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
       name = Column(String(100), nullable=False)
       description = Column(Text, default="")
       collection_name = Column(String(200), nullable=False)  # user_id + kb_id
       kg_db_path = Column(String(300), nullable=False)       # data/kg_{user_id}_{kb_id}.db
       created_at = Column(DateTime, default=datetime.utcnow)
       is_active = Column(Boolean, default=True)
   ```

3. **多知识库 API**：
   - `GET /api/v1/kb/list`：列出用户的所有知识库
   - `POST /api/v1/kb/create`：创建新知识库
   - `GET /api/v1/kb/{kb_id}`：获取知识库详情
   - `PUT /api/v1/kb/{kb_id}`：更新知识库配置
   - `DELETE /api/v1/kb/{kb_id}`：删除知识库（同时删除 Chroma collection 和 KG DB）
   - `GET /api/v1/kb/{kb_id}/stats`：知识库统计（文档数、chunk 数、实体数等）

4. **前端改造**：
   - 侧边栏增加知识库切换下拉菜单
   - 知识库管理页面（创建/删除/重命名）
   - 跨知识库搜索功能（可选）

#### Week 9: 引导向导 + 用户设置页

**任务**：

1. **引导向导**（首次注册用户）：
   - Step 1：欢迎页，介绍系统功能
   - Step 2：创建第一个知识库（输入名称、选择模板）
   - Step 3：上传第一份文档
   - Step 4：开始第一个问答
   - Step 5：介绍订阅方案

2. **用户设置页**：
   - 个人信息编辑（用户名、邮箱）
   - 密码修改
   - API Key 管理（生成/撤销）
   - 订阅状态查看
   - 数据导出（导出所有知识库数据为 ZIP）

3. **错误页面**：
   - 401 未认证：友好的登录引导
   - 403 无权限：说明原因和解决方案
   - 429 超限：显示用量进度条，引导升级
   - 500 内部错误：友好的错误提示 + 错误 ID（用于反馈）

#### Week 10: 暗色主题 + 响应式适配

**任务**：

1. CSS 变量系统改造，支持 `light` / `dark` 主题切换
2. 移动端布局适配（侧边栏收起、卡片式布局）
3. 加载状态、骨架屏、Toast 通知

### 5.5 Phase 4: Business & Operations（第 11-14 周）

**目标**：商业化准备和运维体系完善。

#### Week 11: 订阅定价 + 成本追踪

**任务**：

1. **定价方案**：
   ```python
   PRICING_TIERS = {
       "free": {
           "name": "免费版",
           "price": 0,
           "max_kb": 1,
           "max_chunks": 1000,
           "max_queries_per_day": 50,
           "available_models": ["Qwen2.5-0.5B-Instruct"],
           "reranker": False,
           "kg_enabled": True,
       },
       "pro": {
           "name": "专业版",
           "price": 29,  # 每月 / USD
           "max_kb": 5,
           "max_chunks": 50000,
           "max_queries_per_day": 5000,
           "available_models": ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct-GPTQ-Int4"],
           "reranker": True,
           "kg_enabled": True,
       },
       "enterprise": {
           "name": "企业版",
           "price": 99,
           "max_kb": -1,
           "max_chunks": -1,
           "max_queries_per_day": -1,
           "available_models": ["*"],
           "reranker": True,
           "kg_enabled": True,
           "multi_user": True,  # 允许多人协作
       },
   }
   ```

2. **成本追踪**（针对本地部署场景）：
   - 每个查询估算 Token 消耗
   - 每周生成用户成本报告（纯估算，无实际计费）
   - 磁盘占用统计（每个用户的 Chroma + KG + 原始文档）

#### Week 12: Staging 环境 + 蓝绿部署

**任务**：

1. **Staging 环境配置**：
   - `config/config.staging.yaml`
   - 使用独立的 Chroma 和 SQLite 目录
   - 测试 JWT Secret

2. **蓝绿部署脚本**：
   ```yaml
   # docker-compose.yml（新增）
   services:
     rag-staging:
       image: chineseragkb:${VERSION}
       environment:
         - ENV=staging
       ports:
         - "8001:8000"
   
   # deploy.sh
   #!/bin/bash
   docker-compose pull
   docker-compose up -d staging
   # 健康检查
   curl -f http://localhost:8001/api/health
   # 切换流量
   docker-compose stop production
   docker-compose start staging as production
   ```

#### Week 13: Feature Flag 系统

**任务**：

1. **Feature Flag 数据模型**：
   ```python
   class FeatureFlag(Base):
       __tablename__ = "feature_flags"
       id = Column(String(36), primary_key=True)
       name = Column(String(100), unique=True, nullable=False)
       description = Column(Text)
       is_enabled = Column(Boolean, default=False)
       rollout_percentage = Column(Integer, default=0)  # 0-100
       allowed_tiers = Column(String(200))  # comma-separated: "pro,enterprise"
   ```

2. **Feature Flag 检查函数**：
   ```python
   def is_feature_enabled(user: User, feature_name: str) -> bool:
       flag = db.query(FeatureFlag).filter(FeatureFlag.name == feature_name).first()
       if not flag or not flag.is_enabled:
           return False
       if user.subscription_tier not in flag.allowed_tiers.split(","):
           return False
       return random.random() * 100 < flag.rollout_percentage
   ```

3. **初始 Feature Flags**：
   - `multi_kb`：多知识库功能（默认对所有用户开放）
   - `async_llm`：异步 LLM 执行器（默认关闭，灰度 10%）
   - `reranker_v2`：新版重排序（默认关闭，灰度 5%）
   - `kg_neo4j`：Neo4j 后端支持（默认关闭，仅企业版）

#### Week 14: 全链路监控 + 告警

**任务**：

1. **Prometheus 指标**：
   ```python
   from prometheus_client import Counter, Histogram, Gauge
   
   REQUEST_COUNT = Counter(
       "rag_api_requests_total",
       "Total API requests",
       ["endpoint", "method", "status"]
   )
   REQUEST_LATENCY = Histogram(
       "rag_api_latency_seconds",
       "API request latency",
       ["endpoint"]
   )
   ACTIVE_USERS = Gauge(
       "rag_active_users",
       "Number of active users in last 5 minutes"
   )
   LLM_QUEUE_SIZE = Gauge(
       "rag_llm_queue_size",
       "Current LLM request queue size"
   )
   TOKEN_USAGE = Counter(
       "rag_token_usage_total",
       "Total tokens used",
       ["user_id", "model"]
   )
   ```

2. **Grafana Dashboard**：
   - API 请求量 + 错误率
   - P50/P95/P99 延迟
   - 活跃用户数趋势
   - LLM 队列积压情况
   - Token 消耗趋势
   - 各端点流量热力图

3. **告警规则**：
   - API 错误率 > 5%：Paging
   - P99 延迟 > 10s：Warning
   - LLM 队列积压 > 100：Critical
   - 磁盘使用率 > 80%：Warning
   - Token 消耗异常峰值：Warning

---

## 6. API Changes（API 变更）

> **PM 能力维度映射**：本节对应 **[第二维度：AI Agent Engineering & Multi-Agent Orchestration（AI Agent 工程与多智能体编排）]**  
>
> 本节定义了多知识库 API、Agent 工具调用端点等，体现了构建自主 AI Agent、配备任务规划、内存管理和工具调用能力的专业素养。

### 6.1 API 版本化策略

所有现有端点迁移到 `/api/v1/` 前缀，新端点直接使用 `/api/v1/`。

```
API 版本策略
├── /api/v1/*         当前版本，稳定接口
├── /api/v2/*         未来大版本
├── /health           保留（无认证，可用于负载均衡健康检查）
└── /                  前端静态文件
```

### 6.2 新增认证端点

| 端点 | 方法 | 认证 | 说明 |
|------|------|------|------|
| `/api/v1/auth/register` | POST | 否 | 注册新用户 |
| `/api/v1/auth/login` | POST | 否 | 用户登录 |
| `/api/v1/auth/refresh` | POST | 否 | 刷新 Access Token |
| `/api/v1/auth/logout` | POST | 是 | 注销（将 Refresh Token 加入黑名单） |
| `/api/v1/auth/me` | GET | 是 | 获取当前用户信息 |
| `/api/v1/auth/password` | PUT | 是 | 修改密码 |

**注册请求示例**：

```json
POST /api/v1/auth/register
Content-Type: application/json

{
    "username": "zhangsan",
    "email": "zhangsan@example.com",
    "password": "SecurePass123!"
}

{
    "user_id": "164e2f8c-3a1b-4c9d-8e5f-6a7b8c9d0e1f",
    "username": "zhangsan",
    "email": "zhangsan@example.com",
    "subscription_tier": "free",
    "created_at": "2026-07-18T10:30:00Z"
}
```

**登录请求示例**：

```json
POST /api/v1/auth/login
Content-Type: application/json

{
    "login": "zhangsan",      // username 或 email
    "password": "SecurePass123!"
}

{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 1800
}
```

### 6.3 端点改造矩阵

| 原有端点 | 改造后端点 | 变更说明 |
|----------|-----------|---------|
| `GET /api/health` | `GET /api/health` | 不变，保留无认证 |
| `GET /api/documents` | `GET /api/v1/documents` | 需认证，按 user_id 隔离 |
| `GET /api/documents/{doc_id}/chunks` | `GET /api/v1/documents/{doc_id}/chunks` | 需认证 |
| `POST /api/search` | `POST /api/v1/search` | 需认证，user_id 从 JWT 提取 |
| `POST /api/chat` | `POST /api/v1/chat` | 需认证，使用 AsyncLLMExecutor |
| `WS /ws/chat` | `WS /api/v1/ws/chat?token=xxx` | 需认证，token 作为 query 参数 |
| `POST /api/ingest` | `POST /api/v1/ingest` | 需认证，文档保存到用户目录 |
| `POST /api/ingest/dir` | `POST /api/v1/ingest/dir` | 需认证 |
| `POST /api/eval/run` | `POST /api/v1/eval/run` | 需认证，使用用户专属评估集 |
| `GET /api/config` | `GET /api/v1/config` | 需认证，返回用户有效配置 |
| `GET /api/agent/tools` | `GET /api/v1/agent/tools` | 需认证 |
| `POST /api/agent/chat` | `POST /api/v1/agent/chat` | 需认证 |
| `WS /ws/agent` | `WS /api/v1/ws/agent?token=xxx` | 需认证 |
| `GET /api/kg/stats` | `GET /api/v1/kg/stats` | 需认证，按 user_id 隔离 |
| `GET /api/kg/entities` | `GET /api/v1/kg/entities` | 需认证 |
| `GET /api/kg/relations` | `GET /api/v1/kg/relations` | 需认证 |
| `POST /api/kg/search` | `POST /api/v1/kg/search` | 需认证 |
| `POST /api/kg/query` | `POST /api/v1/kg/query` | 需认证 |
| `POST /api/kg/extract` | `POST /api/v1/kg/extract` | 需认证 |
| `POST /api/kg/build` | `POST /api/v1/kg/build` | 需认证，异步后台任务 |
| `POST /api/kg/graph_rag` | `POST /api/v1/kg/graph_rag` | 需认证 |

### 6.4 新增多知识库端点

| 端点 | 方法 | 认证 | 说明 |
|------|------|------|------|
| `GET /api/v1/kb/list` | GET | 是 | 列出用户所有知识库 |
| `POST /api/v1/kb/create` | POST | 是 | 创建新知识库 |
| `GET /api/v1/kb/{kb_id}` | GET | 是 | 获取知识库详情 |
| `PUT /api/v1/kb/{kb_id}` | PUT | 是 | 更新知识库配置 |
| `DELETE /api/v1/kb/{kb_id}` | DELETE | 是 | 删除知识库及所有数据 |
| `GET /api/v1/kb/{kb_id}/stats` | GET | 是 | 知识库统计信息 |
| `GET /api/v1/kb/{kb_id}/export` | GET | 是 | 导出知识库为 ZIP |

### 6.5 新增运营端点

| 端点 | 方法 | 认证 | 说明 |
|------|------|------|------|
| `GET /api/v1/usage/summary` | GET | 是 | 用量摘要 |
| `GET /api/v1/usage/history` | GET | 是 | 用量历史趋势 |
| `GET /api/v1/admin/users` | GET | 是 | 管理后台：用户列表（仅 admin） |
| `GET /api/v1/admin/stats` | GET | 是 | 全局统计（仅 admin） |

### 6.6 请求头规范

```http
# 认证请求
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# 请求体标准 Content-Type
Content-Type: application/json

# WebSocket 连接（Token 在 query string 中）
WS /api/v1/ws/chat?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# 响应标准格式
{
    "code": 0,        // 0=成功，非0=失败
    "message": "success",
    "data": { ... },
    "request_id": "uuid-for-tracking"
}
```

---

## 7. Database Schema Changes（数据库 Schema 变更）

> **PM 能力维度映射**：本节对应 **[第三维度：RAG Systems & Advanced Data Engineering（RAG 系统与高级数据工程）]**  
>
> 本节设计了 Per-User Chroma Collection 隔离策略和 KG Schema 迁移方案，体现了设计生产级 RAG 流水线的专业能力，涵盖语义分块、向量数据库和 GraphRAG 管道设计。

### 7.1 新建数据库表

#### 7.1.1 用户表（`users`）

```sql
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,          -- UUID v4
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    subscription_tier VARCHAR(20) DEFAULT 'free',
    max_chunks INTEGER DEFAULT 1000,
    max_queries_per_day INTEGER DEFAULT 50,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created_at ON users(created_at);
```

#### 7.1.2 Refresh Token 表（`refresh_tokens`）

```sql
CREATE TABLE refresh_tokens (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    revoked BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_refresh_tokens_user_id ON refresh_tokens(user_id);
CREATE INDEX idx_refresh_tokens_expires ON refresh_tokens(expires_at);
```

#### 7.1.3 知识库表（`knowledge_bases`）

```sql
CREATE TABLE knowledge_bases (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    collection_name VARCHAR(200) NOT NULL,   -- user_id + kb_id 编码
    kg_db_path VARCHAR(300) NOT NULL,         -- data/kg_{user_id}_{kb_id}.db
    config_json TEXT,                         -- 知识库级别配置
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_kb_user_id ON knowledge_bases(user_id);
CREATE INDEX idx_kb_collection ON knowledge_bases(collection_name);
```

#### 7.1.4 用量日志表（`usage_logs`）

```sql
CREATE TABLE usage_logs (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    kb_id VARCHAR(36),
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    tokens_used INTEGER DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    chunks_accessed INTEGER DEFAULT 0,
    error_message TEXT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE SET NULL
);

CREATE INDEX idx_usage_user_id ON usage_logs(user_id);
CREATE INDEX idx_usage_created_at ON usage_logs(created_at);
CREATE INDEX idx_usage_endpoint ON usage_logs(endpoint);
```

### 7.2 KG Schema 迁移

#### 方案 A：Per-User DB（推荐）

**不修改现有 Schema**，只需在创建时使用 user_id 命名：

```
迁移前：data/kg.db
迁移后：data/kg_{user_id_1}.db, data/kg_{user_id_2}.db, ...
```

迁移脚本：

```python
# scripts/migrate_to_multi_tenant.py

import sqlite3
from pathlib import Path
import shutil
import os

def migrate_single_user_data():
    """将现有的单一 kg.db 迁移为多租户格式。
    
    假设当前是单用户部署，需要将 data/kg.db 中的数据
    迁移到默认用户的独立 DB 文件中。
    """
    source_db = Path("data/kg.db")
    if not source_db.exists():
        print("No existing kg.db found, skipping migration.")
        return
    
    # 默认 user_id（可以从环境变量或配置文件读取）
    default_user_id = os.getenv("DEFAULT_USER_ID", "default_user")
    
    target_db = Path(f"data/kg_{default_user_id}.db")
    target_db.parent.mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    shutil.copy2(source_db, target_db)
    print(f"Migrated {source_db} -> {target_db}")
    
    # 保留原文件作为备份
    backup_db = source_db.with_suffix(".db.backup")
    shutil.move(source_db, backup_db)
    print(f"Backed up original to {backup_db}")
```

#### 方案 B：Schema 加 user_id 列

```sql
-- migrations/versions/001_add_user_id_to_kg.py

def upgrade():
    # entities 表
    op.execute("""
        ALTER TABLE entities ADD COLUMN user_id TEXT NOT NULL DEFAULT ''
    """)
    op.create_index(
        'idx_entities_user_id',
        'entities',
        ['user_id']
    )
    
    # relations 表
    op.execute("""
        ALTER TABLE relations ADD COLUMN user_id TEXT NOT NULL DEFAULT ''
    """)
    op.create_index(
        'idx_relations_user_id',
        'relations',
        ['user_id']
    )
    
    # 为现有数据设置默认 user_id（需要指定迁移的目标 user_id）
    # 注意：这个脚本只在新部署时使用，已有数据的迁移需要额外处理
    op.execute("""
        UPDATE entities SET user_id = :default_user_id WHERE user_id = ''
    """, {"default_user_id": os.getenv("MIGRATION_USER_ID", "migrated_default")})
    op.execute("""
        UPDATE relations SET user_id = :default_user_id WHERE user_id = ''
    """, {"default_user_id": os.getenv("MIGRATION_USER_ID", "migrated_default")})

def downgrade():
    op.drop_index('idx_entities_user_id', table_name='entities')
    op.drop_index('idx_relations_user_id', table_name='relations')
    op.execute("ALTER TABLE entities DROP COLUMN user_id")
    op.execute("ALTER TABLE relations DROP COLUMN user_id")
```

### 7.3 Chroma Collection 命名规范

```
命名规则：{user_id[:8]}_{kb_id[:8]}_{base_collection_name}

示例：
  - user_id: 164e2f8c-3a1b-4c9d-8e5f-6a7b8c9d0e1f
  - kb_id: 9a3b1d7e-2c4f-4a8b-9d0e-1f2a3b4c5d6e
  - base_collection_name: chinese_rag_kb
  - 最终 collection 名: 164e2f8c_9a3b1d7e_chinese_rag_kb

好处：
  1. 前缀包含 user_id，可快速定位用户数据
  2. 前缀包含 kb_id，支持多知识库
  3. 底层 HNSW 索引独立，互不干扰
  4. Chroma 的 get_collection() 只需传名称，无需额外过滤
```

---

## 8. Security Considerations（安全注意事项）

> **PM 能力维度映射**：本节对应 **[第四维度：System Design & High-Performance Inference（系统设计与高性能推理）]**  
>
> 本节从 JWT Token 安全、CORS 策略硬化、API 限流、输入验证到文件上传安全，全面展示了系统级安全加固能力，确保多租户环境下的数据隔离和访问控制。

### 8.1 JWT Token 安全策略

```python
# JWT 配置
JWT_SECRET = os.getenv("JWT_SECRET")  # 必须设置，最小 256 位随机字符串
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30      # 短期访问令牌
REFRESH_TOKEN_EXPIRE_DAYS = 7         # 中期刷新令牌

# Token 刷新机制
def refresh_tokens(user_id: str) -> tuple[str, str]:
    # 生成新的 Access Token 和 Refresh Token
    # 将旧的 Refresh Token 标记为 revoked
    revoke_old_refresh_token(user_id)
    return create_access_token(user_id), create_refresh_token(user_id)

# Token 黑名单（使用 Redis 或 DB 表存储已撤销的 token）
class TokenBlacklist:
    def add(self, jti: str, expires_at: datetime):
        # 将 jti（JWT ID）加入黑名单，过期时自动清理
        pass
    
    def is_blacklisted(self, jti: str) -> bool:
        pass
```

### 8.2 CORS 策略硬化

```python
# 生产环境 CORS 配置
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != [""] else ["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)
```

### 8.3 API 限流规则

```python
RATE_LIMITS = {
    # 免费用户
    "free": {
        "/api/v1/auth/login": (10, 60),       # 10次/分钟（防暴力破解）
        "/api/v1/auth/register": (5, 60),     # 5次/分钟
        "/api/v1/search": (20, 60),           # 20次/分钟
        "/api/v1/chat": (10, 60),             # 10次/分钟
        "/api/v1/ingest": (5, 300),            # 5次/5分钟
    },
    # 专业用户
    "pro": {
        "/api/v1/search": (100, 60),          # 100次/分钟
        "/api/v1/chat": (50, 60),             # 50次/分钟
        "/api/v1/ingest": (20, 300),          # 20次/5分钟
    },
    # 企业用户
    "enterprise": {
        "/api/v1/search": (-1, 60),          # 无限制
        "/api/v1/chat": (-1, 60),             # 无限制
    },
}
```

### 8.4 输入验证增强

```python
# 所有用户输入必须经过验证
from pydantic import validator, constr

class ChatRequest(BaseModel):
    query: constr(min_length=1, max_length=2000)  # 限制查询长度
    top_k: int = Field(4, ge=1, le=50)
    
    @validator("query")
    def sanitize_query(cls, v):
        # 去除潜在的 XSS 和 SQL 注入
        v = v.strip()
        v = re.sub(r"[<>'\";]", "", v)  # 移除危险字符
        return v

class IngestRequest(BaseModel):
    file: UploadFile
    
    @validator("file")
    def validate_file(cls, v):
        allowed_extensions = {".pdf", ".docx", ".md", ".txt"}
        ext = Path(v.filename).suffix.lower()
        if ext not in allowed_extensions:
            raise ValueError(f"不支持的文件类型: {ext}")
        
        # 文件大小检查（最大 50MB）
        if v.size and v.size > 50 * 1024 * 1024:
            raise ValueError("文件大小不能超过 50MB")
        
        return v
```

### 8.5 文件上传安全

```python
# 文件上传安全检查
UPLOAD_DIR = Path("data/raw/{user_id}")  # 用户隔离目录

async def secure_upload(file: UploadFile, user_id: str) -> Path:
    # 1. 文件名安全性检查
    safe_filename = re.sub(r"[^\w\s.-]", "", file.filename)
    safe_filename = safe_filename[:200]  # 限制长度
    
    # 2. 文件类型双重检查（扩展名 + Magic Number）
    content = await file.read(16)  # 读取文件头
    await file.seek(0)  # 重置指针
    
    # Magic Number 对应表
    MAGIC_NUMBERS = {
        b"%PDF": ".pdf",
        b"PK\x03\x04": ".docx",  # ZIP-based
        b"\xd0\xcf\x11\xe0": ".doc",  # OLE
    }
    
    file_ext = Path(safe_filename).suffix.lower()
    detected_ext = MAGIC_NUMBERS.get(content, None)
    if detected_ext and detected_ext != file_ext:
        raise HTTPException(400, "文件类型与扩展名不匹配")
    
    # 3. 保存到用户隔离目录
    target_dir = UPLOAD_DIR / user_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_filename
    
    # 4. 异步写入，避免大文件阻塞
    async with aiofiles.open(target_path, "wb") as f:
        await f.write(await file.read())
    
    return target_path
```

---

## 9. Performance Optimization Plan（性能优化计划）

> **PM 能力维度映射**：本节对应 **[第四维度：System Design & High-Performance Inference（系统设计与高性能推理）]**  
>
> 本节设计了 AsyncLLMExecutor、Semaphore 并发控制、请求队列、HNSW 调优和缓存策略，展示了掌握分布式训练框架和推理优化引擎（类似 vLLM、TensorRT）的能力，确保高吞吐量、低延迟和成本效率。

### 9.1 异步 LLM 批推理设计

```
目标：100 并发用户 → 实际 GPU 并发控制在 5-10 个请求

┌─────────────────────────────────────────────────────────┐
│  AsyncLLMExecutor 架构                                   │
│                                                          │
│  ┌──────────┐  ┌──────────┐       ┌──────────────┐     │
│  │ User A   │  │ User B   │  ...  │ User N       │     │
│  │ Chat API │  │ Chat API │       │ Chat API     │     │
│  └────┬─────┘  └────┬─────┘       └──────┬───────┘     │
│       │              │                    │              │
│       └──────────────┼────────────────────┘              │
│                      ▼                                    │
│            ┌─────────────────────┐                       │
│            │  asyncio.Queue      │  maxsize=200         │
│            │  (请求队列)          │                       │
│            └──────────┬──────────┘                       │
│                       │                                   │
│            ┌──────────▼──────────┐                       │
│            │  Semaphore(5)       │  GPU 最大并发数       │
│            │  控制同时执行的任务数  │                       │
│            └──────────┬──────────┘                       │
│                       │                                   │
│            ┌──────────▼──────────┐                       │
│            │  ThreadPoolExecutor│  CPU 线程池           │
│            │  (max_workers=5)    │                       │
│            └──────────┬──────────┘                       │
│                       │                                   │
│            ┌──────────▼──────────┐                       │
│            │  LocalLLM           │                       │
│            │  (Qwen2.5-1.5B)     │  ← GPU 推理          │
│            └─────────────────────┘                       │
└─────────────────────────────────────────────────────────┘

队列满时：返回 HTTP 503 + Retry-After 头
GPU 不可用时：自动回退到 CPU 模式，限流更严格
```

### 9.2 Embedding 请求批处理

```python
# 相同时间窗口内的 embedding 请求合并处理
# 例如：100 个用户的查询在 100ms 内到达，合并为一次 batch encode

import asyncio
from collections import defaultdict

class BatchingEmbeddingCache:
    """批处理 + LRU 缓存的 Embedding 服务。"""
    
    def __init__(self, embedding_model: EmbeddingModel, batch_window_ms: int = 100):
        self.embedding = embedding_model
        self.batch_window = batch_window_ms / 1000
        self._cache: Dict[str, List[float]] = {}
        self._pending: Dict[str, asyncio.Event] = {}
        self._pending_texts: Dict[str, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def encode(self, text: str) -> List[float]:
        # 1. 缓存命中检查
        cache_key = self._hash_text(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 2. 加入待批处理队列
        async with self._lock:
            if cache_key not in self._pending:
                self._pending[cache_key] = asyncio.Event()
                self._pending_texts[cache_key] = []
            
            texts = self._pending_texts[cache_key]
            texts.append(text)
            
            # 延迟执行，等待更多相似请求
            await asyncio.sleep(self.batch_window)
            
            # 3. 批处理
            if texts:
                vectors = self.embedding.encode(texts).tolist()
                for i, t in enumerate(texts):
                    k = self._hash_text(t)
                    self._cache[k] = vectors[i]
                    if k in self._pending:
                        self._pending[k].set()
                
                # 清理
                for t in texts:
                    k = self._hash_text(t)
                    if k in self._pending:
                        del self._pending[k]
                    if k in self._pending_texts:
                        del self._pending_texts[k]
        
        return self._cache[cache_key]
    
    def _hash_text(self, text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]
```

### 9.3 Chroma HNSW 参数调优

```yaml
# config/config.yaml 中新增多租户相关配置

vector_store:
  # HNSW 参数（针对 100 用户场景调优）
  hnsw:
    ef_construction: 200      # 索引构建时的搜索范围（越大越精确但越慢）
    ef_search: 100            # 查询时的搜索范围
    M: 16                     # 每个节点的邻居数（内存与精度的权衡）
  
  # 连接池配置
  pool:
    max_connections: 20      # 最大客户端连接数
    
  # 查询缓存
  cache:
    enabled: true
    max_size: 1000           # 缓存的查询结果数
    ttl_seconds: 300          # 缓存有效期
```

### 9.4 查询结果缓存策略

```python
# 基于语义相似度的缓存：相同意图的查询返回缓存结果

class SemanticQueryCache:
    """语义查询缓存，key 为 query embedding。"""
    
    def __init__(self, max_size: int = 10000, similarity_threshold: float = 0.95):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._embedding: Optional[EmbeddingModel] = None
        self._lock = threading.Lock()
    
    def get(self, query: str, user_id: str) -> Optional[dict]:
        if self._embedding is None:
            return None
        
        cache_key = self._compute_key(query)
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                # 移动到末尾（LRU）
                self._cache.move_to_end(cache_key)
                return entry["result"]
            return None
    
    def set(self, query: str, user_id: str, result: dict):
        with self._lock:
            if len(self._cache) >= self.max_size:
                # 删除最旧的条目
                self._cache.popitem(last=False)
            
            cache_key = self._compute_key(query)
            self._cache[cache_key] = {
                "user_id": user_id,
                "result": result,
                "timestamp": time.time(),
            }
```

---

## 10. Testing Strategy（测试策略）

> **PM 能力维度映射**：本节对应 **[第五维度：AI-Native Evaluation & Observability（AI 原生评估与可观测性）]**  
>
> 本节设计了完整的评估框架，包括单元测试、集成测试、Locust 负载测试和混沌工程，展示了构建 LLM 和 Agent 自动化测试流水线的能力，确保系统稳定性和可靠性。

### 10.1 测试分层

```
测试金字塔
         ▲
        /│\        E2E Tests（端到端）
       / │ \       场景：注册→登录→上传→检索→问答
      /  │  \
     /───┼───\
    /    │    \      Integration Tests（集成测试）
   /     │     \     场景：多用户数据隔离验证
  /──────┼──────\    API 端点测试
 /       │       \
/────────┼────────\
         │         Unit Tests（单元测试）
    Auth Layer       每个组件独立测试
    Tenant Factory    Mock 外部依赖
    Rate Limiter
```

### 10.2 单元测试（Phase 1 并行）

```python
# tests/test_auth.py

import pytest
from src.auth.jwt_handler import create_access_token, decode_token, create_refresh_token

class TestJWTAuth:
    def test_create_and_decode_access_token(self):
        user_id = "test-user-123"
        token = create_access_token(user_id)
        payload = decode_token(token)
        
        assert payload is not None
        assert payload["sub"] == user_id
        assert payload["type"] == "access"
    
    def test_expired_token(self):
        # 测试过期 token
        pass
    
    def test_invalid_token(self):
        # 测试伪造 token
        pass
    
    def test_refresh_token_flow(self):
        # 测试刷新 token 流程
        pass

# tests/test_tenant_factory.py

class TestTenantAwareFactory:
    def test_different_users_get_different_collections(self):
        factory = TenantAwareFactory()
        store_a = factory.get_chroma_store("user_a", base_config)
        store_b = factory.get_chroma_store("user_b", base_config)
        
        assert store_a.collection_name != store_b.collection_name
        assert "user_a" in store_a.collection_name
        assert "user_b" in store_b.collection_name
    
    def test_same_user_gets_same_instance(self):
        factory = TenantAwareFactory()
        store_a = factory.get_chroma_store("user_a", base_config)
        store_a2 = factory.get_chroma_store("user_a", base_config)
        
        assert store_a is store_a2
    
    def test_different_users_get_independent_kg_stores(self):
        factory = TenantAwareFactory()
        kg_a = factory.get_kg_store("user_a", base_config)
        kg_b = factory.get_kg_store("user_b", base_config)
        
        assert kg_a.db_path != kg_b.db_path
        assert "user_a" in kg_a.db_path
        assert "user_b" in kg_b.db_path
```

### 10.3 集成测试（Phase 2 并行）

```python
# tests/test_multi_tenant_isolation.py

import pytest
from fastapi.testclient import TestClient

class TestMultiTenantIsolation:
    """验证用户数据隔离的集成测试。"""
    
    @pytest.fixture
    def client(self):
        from api.main import app
        return TestClient(app)
    
    @pytest.fixture
    def user_a_token(self):
        # 注册并登录用户 A
        pass
    
    @pytest.fixture
    def user_b_token(self):
        # 注册并登录用户 B
        pass
    
    def test_user_cannot_see_other_users_documents(self, client, user_a_token, user_b_token):
        # 用户 A 上传文档
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("doc_a.txt", b"Content for User A")},
            headers={"Authorization": f"Bearer {user_a_token}"}
        )
        assert response.status_code == 200
        
        # 用户 A 列出文档
        response = client.get(
            "/api/v1/documents",
            headers={"Authorization": f"Bearer {user_a_token}"}
        )
        assert "doc_a.txt" in str(response.json())
        
        # 用户 B 列出文档（不应该看到 A 的文档）
        response = client.get(
            "/api/v1/documents",
            headers={"Authorization": f"Bearer {user_b_token}"}
        )
        assert "doc_a.txt" not in str(response.json())
    
    def test_user_cannot_see_other_users_kg(self, client, user_a_token, user_b_token):
        # 用户 A 添加 KG 实体
        # 用户 B 查询 KG（不应该看到 A 的实体）
        pass
    
    def test_unauthenticated_access_blocked(self, client):
        # 未认证请求应返回 401
        response = client.get("/api/v1/documents")
        assert response.status_code == 401
```

### 10.4 负载测试（Phase 3）

```python
# tests/load_test_locust.py
# 使用 Locust 进行负载测试

from locust import HttpUser, task, between

class RAGLoadUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # 登录获取 token
        response = self.client.post("/api/v1/auth/login", json={
            "login": "loadtest@example.com",
            "password": "LoadTest123!"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)  # 权重：搜索最频繁
    def search(self):
        self.client.post(
            "/api/v1/search",
            json={"query": "RAG 系统架构", "top_k": 4},
            headers=self.headers,
            name="/api/v1/search"
        )
    
    @task(1)  # 权重：问答较少
    def chat(self):
        self.client.post(
            "/api/v1/chat",
            json={"query": "什么是知识图谱？", "top_k": 4},
            headers=self.headers,
            name="/api/v1/chat"
        )
    
    @task(1)  # 权重：文档上传最少
    def ingest(self):
        self.client.post(
            "/api/v1/ingest",
            files={"file": ("test.txt", b"测试文档内容")},
            headers=self.headers,
            name="/api/v1/ingest"
        )
```

**负载测试目标**：

| 指标 | 目标值 |
|------|--------|
| 支持并发用户数 | 100 |
| API P50 延迟 | < 500ms |
| API P95 延迟 | < 2s |
| API P99 延迟 | < 5s |
| 错误率 | < 1% |
| LLM 队列积压 | < 50 |

### 10.5 混沌工程

```yaml
# chaos_experiment.yaml（使用 Chaos Toolkit）

name: "LLM Service Failure"
description: "模拟 LLM 服务不可用时的系统行为"

steady-state-hypothesis:
  title: "System is healthy"
  probes:
    - type: probe
      name: "health-check"
      tolerance: 200

actions:
  - type: action
    name: "kill-llm-process"
    provider:
      type: python
      module: os.system
      arguments:
        cmd: "pkill -f transformers"

rollbacks:
  - type: action
    name: "restart-llm-process"
    provider:
      type: python
      module: os.system
      arguments:
        cmd: "python -c 'from src.llm import LocalLLM; LocalLLM()'"

expected:
  - type: probe
    name: "api-returns-503"
    tolerance: true
```

---

## 11. Rollout Plan（上线计划）

> **PM 能力维度映射**：本节对应 **[第五维度：AI-Native Evaluation & Observability（AI 原生评估与可观测性）]**  
>
> 本节设计了 Shadow Mode、Opt-in 模式、Feature Flag 灰度发布和 Prometheus 监控告警，展示了全链路追踪、异常检测和持续反馈循环的能力，确保系统稳定性和可靠性。

### 11.1 迁移策略

```
迁移三步走
┌─────────────────────────────────────────────────────────┐
│  Step 0: 兼容性准备（上线前）                             │
│  ├── 现有 API 增加 /api/v1/ 前缀版本                      │
│  ├── JWT 中间件可选激活（通过环境变量控制）                 │
│  ├── 多租户代码 review + 测试通过                         │
│  └── 压力测试达标（100 并发）                             │
└─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Shadow Mode（Week 1）                           │
│  ├── JWT 中间件激活，但缺失 token 时降级到匿名访问          │
│  ├── 新增 users 表，旧系统数据映射到 DEFAULT_USER          │
│  ├── 所有操作同时记录 user_id（即使为空）                  │
│  └── 观察日志，无用户影响                                 │
└─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: Opt-in 模式（Week 2）                           │
│  ├── 新增 /api/v1/auth/register 和 /api/v1/auth/login    │
│  ├── 新用户默认走多租户流程                               │
│  ├── 旧用户（无 token）继续走单用户模式                   │
│  └── 邀请 10 名种子用户内测                               │
└─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: 强制认证（Week 3）                              │
│  ├── 所有 API 强制要求 JWT Token                          │
│  ├── 旧用户数据迁移脚本执行（data/kg.db → data/kg_{uid}.db）│
│  ├── Chroma collection 重命名（chinese_rag_kb → uid_chinese_rag_kb）│
│  └── 监控错误率，处理迁移异常                             │
└─────────────────────────────────────────────────────────┘
```

### 11.2 Feature Flags 逐步启用

```python
# 上线时使用的 Feature Flag 策略

FEATURE_FLAG_ROLLOUT = {
    # Phase 1 立即开启
    "multi_tenant": {
        "enabled": True,
        "rollout_percentage": 100,
        "allowed_tiers": ["free", "pro", "enterprise"],
    },
    
    # Phase 2 灰度开启
    "async_llm": {
        "enabled": True,
        "rollout_percentage": 10,     # 先 10% 用户
        "allowed_tiers": ["pro", "enterprise"],
    },
    
    # Phase 3 全面开启
    "kg_neo4j": {
        "enabled": False,
        "rollout_percentage": 0,
        "allowed_tiers": ["enterprise"],
    },
}
```

### 11.3 回滚程序

```bash
#!/bin/bash
# rollback.sh

set -e

CURRENT_VERSION=${1:-$(git describe --tags --abbrev=0)}
PREVIOUS_VERSION=${2}

echo "Rolling back from $CURRENT_VERSION to $PREVIOUS_VERSION"

# 1. 停止服务
docker-compose down

# 2. 切换代码到旧版本
git checkout $PREVIOUS_VERSION

# 3. 恢复数据库（如果需要）
if [ -f "backups/kg.db.$CURRENT_VERSION" ]; then
    cp backups/kg.db.$CURRENT_VERSION data/kg.db
fi

# 4. 恢复 Chroma 数据（如果需要）
if [ -d "backups/chroma_db.$CURRENT_VERSION" ]; then
    cp -r backups/chroma_db.$CURRENT_VERSION data/chroma_db
fi

# 5. 重新启动
docker-compose up -d

# 6. 健康检查
sleep 5
curl -f http://localhost:8000/api/health || exit 1

echo "Rollback completed successfully"
```

### 11.4 监控与告警阈值

```yaml
# prometheus_alerts.yml

groups:
  - name: chinese_rag_kb_alerts
    rules:
      - alert: HighAPIErrorRate
        expr: rate(rag_api_requests_total{status=~"5.."}[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "API 错误率超过 5%"
          description: "端点 {{ $labels.endpoint }} 错误率 {{ $value | humanizePercentage }}"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(rag_api_latency_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 延迟超过 10 秒"
      
      - alert: LLMQueueBacklog
        expr: rag_llm_queue_size > 100
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM 请求队列积压超过 100"
      
      - alert: DiskUsageHigh
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) < 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "磁盘可用空间低于 20%"
      
      - alert: ActiveUsersDropped
        expr: rag_active_users < 5
        for: 30m
        labels:
          severity: info
        annotations:
          summary: "活跃用户数异常低，可能存在服务问题"
```

---

## 12. Appendix（附录）

### 12.1 PM 能力模型参考

> **PM 能力维度映射**：本节汇总了文档各章节与 PM 能力模型的映射关系，体现了 AI & Agent 工程师核心能力框架的完整应用。

本方案的设计遵循了以下 **AI & Agent 工程师核心能力模型**（6 维度）：

```
AI & Agent 工程师核心能力模型
│
├── 1. Algorithmic Foundation & Model Architecture（算法基础与模型架构）
│   ├── LLM 推理架构（async vs sync）
│   ├── Embedding 批处理优化
│   └── Vector DB 索引（HNSW 调优）
│
├── 2. AI Agent Engineering & Multi-Agent Orchestration（AI Agent 工程与多智能体编排）
│   ├── 多租户工厂模式设计
│   ├── Per-User RAG Pipeline 编排
│   └── Agent 工具调用端点
│
├── 3. RAG Systems & Advanced Data Engineering（RAG 系统与高级数据工程）
│   ├── Chroma Per-User Collection 隔离
│   ├── KG Per-User DB 隔离
│   ├── 语义分块策略
│   └── GraphRAG Pipeline 设计
│
├── 4. System Design & High-Performance Inference（系统设计与高性能推理）
│   ├── AsyncLLMExecutor 架构
│   ├── Semaphore 并发控制
│   ├── 请求队列管理
│   ├── HNSW 参数调优
│   └── 多层缓存策略
│
├── 5. AI-Native Evaluation & Observability（AI 原生评估与可观测性）
│   ├── 用量遥测系统
│   ├── Prometheus 指标采集
│   ├── Locust 负载测试
│   ├── 混沌工程实验
│   └── 评估流水线设计
│
└── 6. Technical Leadership & Business Value Translation（技术领导力与商业价值转化）
    ├── 4 阶段 14 周路线图规划
    ├── 定价层级策略
    ├── 竞争定位分析
    └── 跨职能团队协作
```

#### 12.1.1 章节与 PM 能力维度映射表

| 章节 | PM 能力维度 | 关键主题 |
|------|-------------|----------|
| 1. Executive Summary | **第六维度** | 战略愿景、目标状态、升级范围定义 |
| 2. Current Architecture Analysis | **第四维度** | 同步推理、全局单例等性能瓶颈分析 |
| 3. Target Architecture | **第二维度** | TenantAwareFactory、Multi-Tenant Pipeline 设计 |
| 4. Gap Analysis | **第六维度** | 差距矩阵、优先级排序、资源评估 |
| 5. Implementation Roadmap | **第六维度** | 4 阶段 14 周计划、工作量估算 |
| 6. API Changes | **第二维度** | Agent 工具调用、多知识库 API |
| 7. Database Schema Changes | **第三维度** | Per-User Collection、KG Schema 迁移 |
| 8. Security Considerations | **第四维度** | JWT、CORS、限流、输入验证安全加固 |
| 9. Performance Optimization | **第四维度** | AsyncLLMExecutor、Semaphore、HNSW 调优 |
| 10. Testing Strategy | **第五维度** | Locust 负载测试、混沌工程、评估流水线 |
| 11. Rollout Plan | **第五维度** | Shadow Mode、Feature Flag、Prometheus 监控 |
| 12. Appendix | **所有维度** | 模型汇总、映射表、术语表 |

#### 12.1.2 PM 能力维度详解

**维度 1：Algorithmic Foundation & Model Architecture（算法基础与模型架构）**
- 对应章节：9.3（HNSW 参数调优）
- 核心能力：掌握深度学习原理和 Transformer 架构；精通预训练、SFT、RLHF 和 DPO 对齐技术

**维度 2：AI Agent Engineering & Multi-Agent Orchestration（AI Agent 工程与多智能体编排）**
- 对应章节：3.5（工厂模式）、6.4（多知识库 API）
- 核心能力：构建自主 AI Agent；设计多智能体协作系统

**维度 3：RAG Systems & Advanced Data Engineering（RAG 系统与高级数据工程）**
- 对应章节：7.2（KG Schema 迁移）、7.3（Collection 命名规范）
- 核心能力：设计生产级 RAG 流水线；掌握语义分块、向量数据库、GraphRAG

**维度 4：System Design & High-Performance Inference（系统设计与高性能推理）**
- 对应章节：2.1（当前架构分析）、8（安全注意事项）、9（性能优化）
- 核心能力：掌握分布式训练框架和推理优化引擎；模型量化、剪枝、KV Cache 管理

**维度 5：AI-Native Evaluation & Observability（AI 原生评估与可观测性）**
- 对应章节：10（测试策略）、11.4（监控告警）
- 核心能力：构建 LLM 和 Agent 自动化测试流水线；全链路追踪、异常检测

**维度 6：Technical Leadership & Business Value Translation（技术领导力与商业价值转化）**
- 对应章节：1（执行摘要）、4（差距分析）、5（实施路线图）
- 核心能力：将业务需求转化为可扩展 AI 架构；平衡技术创新与商业 ROI

### 12.2 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| 多租户 | Multi-Tenant | 多个用户/组织共享同一套系统实例，但数据相互隔离 |
| 知识库 | Knowledge Base (KB) | 用户创建的知识库实例，包含向量数据和 KG 数据 |
| 隔离策略 | Isolation Strategy | Per-User Collection 或 Per-User DB 等数据隔离方案 |
| JWT | JSON Web Token | 用于身份认证的开放标准（RFC 7519） |
| CORS | Cross-Origin Resource Sharing | 跨域资源共享策略 |
| 限流 | Rate Limiting | 控制用户在单位时间内的 API 调用次数 |
| Feature Flag | Feature Flag | 功能开关，支持灰度发布 |
| HNSW | Hierarchical Navigable Small World | Chroma 使用的向量索引算法 |
| 工厂模式 | Factory Pattern | 按需创建对象实例的设计模式 |
| Shadow Mode | Shadow Mode | 新旧系统并行运行，新系统只读取不做写入 |
| 订阅层 | Subscription Tier | 定价层级（Free/Pro/Enterprise） |
| 用量遥测 | Usage Telemetry | 追踪和记录 API 使用量数据 |
| Token 限流 | Token Bucket | 一种限流算法 |
| 混沌工程 | Chaos Engineering | 在生产环境引入故障以测试系统韧性 |
| AsyncLLMExecutor | AsyncLLMExecutor | 异步 LLM 执行器，支持并发控制和请求队列 |
| Per-User Collection | Per-User Collection | 每个用户独立的 Chroma Collection |
| GraphRAG | GraphRAG | 结合知识图谱的 RAG 系统 |

### 12.3 相关文档链接

| 文档 | 路径 | 说明 |
|------|------|------|
| 架构说明 | `docs/architecture.md` | 现有系统架构（mermaid 图） |
| 使用指南 | `docs/usage.md` | 详细使用说明 |
| README | `README.md` | 项目主 README |
| 配置参考 | `config/config.yaml` | 完整配置项说明 |
| KG Schema | `src/kg/schema.py` | 实体/关系/三元组定义 |
| 现有端点 | `api/main.py` | 所有 REST + WebSocket 端点 |

### 12.4 技术栈汇总

```
升级后技术栈

前端：
├── Streamlit + 自定义 CSS（主题支持）
└── 未来可选：Gradio / React 前端

后端：
├── FastAPI 0.110+（API 框架）
├── SQLAlchemy 2.0+（ORM）
├── Alembic（数据库迁移）
├── Pydantic 2.0+（数据验证）
└── uvicorn（ASGI 服务器）

认证：
├── PyJWT / python-jose（JWT 处理）
├── bcrypt / passlib（密码哈希）
└── Redis（可选，Token 黑名单/缓存）

向量存储：
├── ChromaDB 0.4+（Per-User Collection）
└── HNSW 参数调优

知识图谱：
├── SQLite（Per-User DB）或
└── Neo4j（可选，企业版）

LLM 推理：
├── transformers（HuggingFace，本地 LLM）
├── asyncio + ThreadPoolExecutor（异步队列）
└── sentence-transformers（Embedding）

部署与运维：
├── Docker + docker-compose
├── GitHub Actions（CI/CD）
├── Prometheus + Grafana（监控）
└── Nginx（反向代理 + 静态资源）

测试：
├── pytest + pytest-cov（单元测试）
├── Locust（负载测试）
└── Chaos Toolkit（混沌工程）
```

### 12.5 文件变更摘要

```
新增文件（按 Phase 排序）

Phase 1:
  src/auth/jwt_handler.py          # JWT 工具函数
  src/auth/password.py              # 密码哈希工具
  src/db/models.py                  # SQLAlchemy 模型（User, UsageLog 等）
  src/db/database.py                # 数据库连接管理
  src/middleware/auth.py            # JWT 认证中间件
  src/middleware/rate_limit.py      # 限流中间件
  src/middleware/telemetry.py       # 用量记录中间件
  src/factories.py                  # 多租户组件工厂
  api/routes/auth.py                # 认证路由
  api/routes/v1.py                  # v1 API 路由汇总
  migrations/versions/*.py          # Alembic 迁移脚本
  .github/workflows/ci.yml          # CI 流水线

Phase 2:
  src/llm_async.py                  # 异步 LLM 执行器
  src/cache.py                      # 查询缓存
  api/routes/usage.py                # 用量查询路由
  api/routes/config.py              # 用户配置路由

Phase 3:
  ui/pages/knowledge_bases.py       # 多知识库管理页面
  ui/pages/user_settings.py         # 用户设置页面
  ui/pages/onboarding.py            # 引导向导
  ui/components/tenant_selector.py   # 知识库切换组件

Phase 4:
  src/feature_flags.py              # Feature Flag 系统
  src/metrics.py                    # Prometheus 指标
  docker-compose.yml                # Docker 部署配置
  deploy.sh                         # 部署脚本
  rollback.sh                       # 回滚脚本

改造文件（按 Phase 排序）

Phase 1:
  api/main.py                       # 认证中间件 + user_id 注入
  src/vector_store.py               # 工厂模式改造
  src/kg/store.py                   # Per-User DB 支持
  src/rag_pipeline.py               # user_id 参数 + 异步调用

Phase 2:
  src/llm.py                        # 异步推理支持
  src/embeddings.py                 # 批处理缓存
  src/agent/react_agent.py          # 异步 Agent

Phase 3:
  ui/app.py                         # 多知识库 UI 改造
  config/config.yaml                # 新增多租户配置项
```

---

*文档由 AI Agent 基于代码分析生成。如有疑问，请参考对应源代码文件。*
