# 📚 中文个人知识库 RAG 系统（ChineseRAGKB）

> 一个面向 **GTX 1060 4GB / 32GB** 低端硬件、本地化、零外部依赖的中文 RAG 知识库系统。
> 支持 PDF / Word / Markdown / TXT 多格式入库、轻量中文 Embedding、4bit 量化本地 LLM、可选重排序、Streamlit Web UI 与命令行工具。

![arch](docs/architecture.md)

---

## ✨ 项目特色

- 🇨🇳 **中文优化**：默认 Embedding 采用 `BAAI/bge-small-zh-v1.5`，中文语义召回更强
- 💰 **轻量友好**：4bit 量化 + 1.5B 参数 LLM，仅需 ~1.2GB 显存即可运行
- 🧱 **零外部依赖**：完全本地运行，不上传任何数据
- 📄 **多格式文档**：PDF（pdfplumber）/ DOCX / Markdown / TXT
- 🪜 **完整 RAG 链路**：智能分块 → 向量检索 → 可选重排 → 引用生成 → 流式回答
- 🖥 **Streamlit UI**：聊天、上传、管理、设置、评估一体化界面
- 📊 **评估闭环**：检索命中率、关键词覆盖、响应时间一站式统计
- 📊 **指标可视化与历史对比**：Plotly 多维图表 + NDCG/MRR/TTFT/tokens/s + 自动历史存档
- 📊 **指标可视化面板** — Streamlit 中一键查看 NDCG@5、MRR、TTFT、tokens/s，附带历史趋势对比（详见 docs/usage.md 第 11 节）
- 🧪 **工程化**：模块化、配置驱动、可替换组件（Embedding / LLM / Reranker）
- 🧬 **数据合成 Pipeline** — 从 markdown 自动抽取实体 → 填空 10 类问题模板，批量产出 Easy/Medium 评估样本（详见 docs/usage.md 第 12 节）

---

## 🧭 目录

- [硬件要求](#-硬件要求)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [配置说明](#-配置说明)
- [性能优化建议](#-性能优化建议)
- [参考项目](#-参考项目)
- [路线图](#-路线图)

---

## 🔧 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | GTX 1050 4GB | **GTX 1060 6GB / RTX 3050 8GB** |
| 内存 | 16 GB | 32 GB |
| 硬盘 | 10 GB | SSD 20 GB+ |
| Python | 3.10 | 3.10 / 3.11 |

**默认模型组合（GTX 1060 4GB 可用）**：
- Embedding: `BAAI/bge-small-zh-v1.5`（约 100MB）
- LLM: `Qwen/Qwen2.5-1.5B-Instruct` + 4bit 量化（约 1.2GB 显存）
- Reranker（可选）: `BAAI/bge-reranker-base`（约 700MB）

若需要更强能力，可切换至 `Qwen2.5-3B-Instruct-GPTQ-Int4` 或 `gemma-2-2b-it`。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
git clone <your-repo> ChineseRAGKB
cd ChineseRAGKB

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Linux/macOS

pip install -r requirements.txt
```

> ⚠️ Pascal 架构（GTX 10 系）建议安装 CUDA 11.8 版 PyTorch：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. 下载模型（首次使用）

设置国内镜像后下载默认模型：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-small-zh-v1.5
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
```

或运行后让代码自动下载（首次运行稍慢）。

### 3. 启动 Web UI

```bash
streamlit run ui/app.py --server.port 8501
```

浏览器打开 `http://localhost:8501`，进入"文档上传"页面上传资料，然后切换到"智能问答"开始提问。

### 4. 命令行快速体验

```bash
# 入库示例文档
python scripts/ingest.py data/raw

# 提问
python scripts/query.py "什么是 RAG？"
```

---

## 📁 项目结构

```
ChineseRAGKB/
├── README.md                  # 本文档
├── requirements.txt           # 依赖清单（带版本号）
├── .gitignore                 # Git 忽略规则
├── .env.example               # 环境变量示例
│
├── config/
│   ├── config.yaml            # 主配置（模型路径、chunk、Top-K、设备、量化…）
│   └── prompts.yaml           # 中文 RAG Prompt 模板
│
├── src/
│   ├── document_loader.py     # 多格式文档解析
│   ├── text_splitter.py       # 中文智能分块
│   ├── embeddings.py          # Embedding 模型封装
│   ├── vector_store.py        # Chroma 向量库封装
│   ├── llm.py                 # 本地 LLM 加载与推理
│   ├── reranker.py            # 可选重排序模型
│   ├── rag_pipeline.py        # RAG 主流程
│   ├── prompt_template.py     # Prompt 模板与引用格式化
│   └── utils.py               # 通用工具（日志/计时/路径/配置）
│
├── ui/
│   └── app.py                 # Streamlit Web 界面
│
├── scripts/
│   ├── ingest.py              # 命令行入库
│   ├── query.py               # 命令行查询
│   ├── evaluate.py            # 评估脚本
│   └── synthesize.py          # 评估集数据合成 pipeline（从 md 自动出题）
│
├── data/
│   ├── raw/                   # 原始文档（.gitkeep 占位）
│   ├── chroma_db/             # Chroma 持久化目录（.gitkeep 占位）
│   └── eval/                  # 评估数据集（.gitkeep 占位）
│
├── tests/
│   ├── test_document_loader.py
│   ├── test_text_splitter.py
│   ├── test_rag_pipeline.py
│   ├── test_evaluate_metrics.py
│   └── test_synthesize.py     # 数据合成 pipeline 单元测试
│
└── docs/
    ├── architecture.md        # 架构说明（含 mermaid 图）
    └── usage.md               # 详细使用指南
```

---

## ⚙️ 配置说明

主配置位于 `config/config.yaml`，关键项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `embedding.model_name` | `BAAI/bge-small-zh-v1.5` | 中文 Embedding 模型 |
| `embedding.batch_size` | `32` | Embedding 批大小 |
| `text_splitter.chunk_size` | `256` | 单块最大字符数 |
| `text_splitter.chunk_overlap` | `32` | 块间重叠字符数 |
| `vector_store.persist_directory` | `data/chroma_db` | Chroma 持久化目录 |
| `vector_store.distance_fn` | `cosine` | 距离函数 |
| `retrieval.top_k` | `5` | 召回条数 |
| `reranker.enabled` | `false` | 是否启用重排序 |
| `llm.model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | LLM 模型 |
| `llm.quantization.enabled` | `true` | 是否启用 4bit 量化 |
| `llm.quantization.quant_type` | `nf4` | 量化类型 |
| `llm.generation.temperature` | `0.7` | 采样温度 |
| `rag.max_context_tokens` | `2048` | 上下文最大 token 数 |
| `ui.max_chat_history` | `20` | 内存中保留的最大对话轮数 |

部分常用项支持 `.env` 覆盖：

```bash
LLM_MODEL_PATH=Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4
USE_4BIT=true
DEVICE=cuda
```

详见 `config/config.yaml` 注释与 `docs/usage.md`。

---

## ⚡ 性能优化建议

### 显存/内存

1. **量化优先**：4bit + double quant 是低端 GPU 的最佳选择（已默认开启）
2. **小模型优先**：Qwen2.5-1.5B 已是性价比甜点；进一步压缩可用 0.5B
3. **CPU-GPU 混合**：Embedding 放 GPU，LLM 显存吃紧时切 CPU
4. **流式输出**：UI 中开启 `llm.generation.stream=true`，首 token 延迟可大幅降低

### 检索质量

1. **Chunk 参数**：中文通常 `chunk_size=200~300`、`overlap=20~50` 较优
2. **重排序**：启用 `BAAI/bge-reranker-base` 后 Top-5 准确率通常 +10~20%
3. **混合检索**：可启用 BM25 + 向量混合（`vector_store.hybrid.enabled: true`）
4. **metadata 过滤**：长文档按章节入库，通过 metadata 缩小召回范围

### 启动速度

1. **延迟加载 LLM**：UI 启动时不加载 LLM，进入问答页时才加载（已实现）
2. **复用对象**：通过 `@st.cache_resource` 缓存模型（已实现）
3. **小 Embedding**：bge-small 比 bge-base 快约 2 倍

---

## 📚 参考项目

本项目设计参考了以下优秀开源项目：

- [RAGFlow](https://github.com/infiniflow/ragflow) — 工程化深度文档解析与 RAG 引擎
- [QAnything](https://github.com/netease-youdao/QAnything) — 网易有道本地知识库问答
- [MaxKB](https://github.com/1Panel-dev/MaxKB) — 1Panel 出品的开箱即用 RAG 平台
- [LangChain](https://github.com/langchain-ai/langchain) — LLM 应用开发框架
- [Chroma](https://github.com/chroma-core/chroma) — 嵌入式向量数据库
- [BAAI BGE](https://github.com/FlagOpen/FlagEmbedding) — 中文 Embedding / Reranker
- [Qwen](https://github.com/QwenLM/Qwen2.5) — 阿里通义千问开源模型
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) — 句子向量库

---

## 🛣 路线图

- [ ] 支持图片 OCR 与表格识别
- [ ] 多知识库切换（多租户）
- [ ] Agent 工具调用（计算器 / SQL）
- [ ] FastAPI + WebSocket 流式问答服务
- [ ] 基于 LlamaIndex 的多跳推理
- [ ] RAGAs 集成评估
- [ ] Web UI 暗色主题与移动端适配
- [ ] 一键安装脚本（`install.sh` / `install.ps1`）

---

## 📝 License

MIT License — 详见 `LICENSE`。

## 🙏 致谢

感谢开源社区，特别是 HuggingFace、Chroma、Streamlit、BAAI、阿里通义千问团队提供的高质量工具与模型。

---

> 💬 有任何问题或建议，欢迎提 Issue 或 PR。Happy RAG! 🚀


---


> 本项目代码同时托管在 Gitee：<https://gitee.com/tokyi/agent.git>

