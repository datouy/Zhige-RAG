# 📚 中文个人知识库 RAG 系统（ChineseRAGKB）

> 一个面向 **GTX 1060 4GB / 32GB** 低端硬件、本地化、零外部依赖的中文 RAG 知识库系统。
> 支持 PDF / Word / Markdown / TXT 多格式入库、轻量中文 Embedding、4bit 量化本地 LLM、可选重排序、Streamlit Web UI 与命令行工具。

📦 **文档导航**：[安装文档 INSTALL.md](INSTALL.md) · [部署手册 DEPLOYMENT.md](DEPLOYMENT.md) · [架构 docs/architecture.md](docs/architecture.md) · [使用指南 docs/usage.md](docs/usage.md)

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
- [全新电脑从 Git 拉取运行](#-全新电脑从-git-拉取运行完整流程)
- [常见问题 FAQ](#-常见问题faq)
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

## 🗄️ 数据库迁移

项目使用 Alembic 进行数据库版本管理。

### 迁移命令

```bash
# 安装依赖
pip install alembic>=0.9.0

# 初始化迁移环境（首次）
alembic init alembic

# 创建新迁移
alembic revision --autogenerate -m "描述迁移内容"

# 应用所有迁移
alembic upgrade head

# 回滚上一个迁移
alembic downgrade -1

# 回滚到指定版本
alembic downgrade <revision>

# 查看迁移状态
alembic current
alembic history

# 验证迁移（检查是否与模型同步）
alembic check
```

### 配置说明

数据库 URL 通过以下顺序加载：
1. 环境变量 `DATABASE_URL`
2. `config/config.yaml` 中的 `database.url`
3. 默认 SQLite（`data/users.db`）

生产环境推荐使用 PostgreSQL：
```bash
export DATABASE_URL=postgresql://user:password@localhost:5432/chineseragkb
```

### PostgreSQL 连接池配置

```bash
export DB_POOL_SIZE=20          # 基础连接数
export DB_MAX_OVERFLOW=40       # 最大溢出连接数
export DB_POOL_TIMEOUT=30       # 获取连接超时（秒）
export DB_POOL_RECYCLE=1800      # 连接回收时间（秒）
export DB_POOL_PRE_PING=true     # 使用前验证连接
```

---

## 🆕 全新电脑从 Git 拉取运行（完整流程）

> 假设你已经把项目推到了 GitHub / Gitee，另一台电脑是**干净环境**。

### 第 0 步：确认对方电脑已装 Python

```bash
python --version
# 要求 Python 3.10 ~ 3.12，推荐 3.11
```

没有的话：
- **Windows**：去 [python.org](https://www.python.org/downloads/) 下载安装，**勾选 Add to PATH**
- **Linux**：`sudo apt install python3.11 python3.11-venv python3-pip`
- **macOS**：`brew install python@3.11`

### 第 1 步：克隆代码

```bash
# HTTPS（推荐）
git clone https://github.com/你的用户名/ChineseRAGKB.git
# 或者 Gitee（国内快）
# git clone https://gitee.com/tokyi/agent.git

cd ChineseRAGKB
```

### 第 2 步：一键安装（自动完成虚拟环境 + 依赖 + 模型）

```bash
# Windows
python scripts\install.py

# Linux / macOS
python scripts/install.py
```

> ⚠️ 首次运行脚本需要网络下载依赖（约 1-2 GB），请耐心等待。脚本会自动：
> - 创建 `.venv` 虚拟环境
> - 检测是否有 NVIDIA 显卡，自动选择 PyTorch 版本
> - 用清华镜像加速下载
> - 下载 Embedding + LLM 模型

### 第 3 步：一键启动（同时开后端 + 前端 + LLM 预加载）

```bash
# Windows
python scripts\start_all.py

# Linux / macOS
python scripts/start_all.py
```

浏览器会自动打开 `http://localhost:8501`。

---

## 🧯 常见问题（FAQ）

### Q1: `pip install` 报错 "Microsoft Visual C++ 14.0 or greater is required"
**A:** Windows 上 `pdfplumber`、`chroma` 等依赖需要 C++ 编译环境。
- 安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，勾选 **"使用 C++ 的桌面开发"**
- 或者使用 conda：`conda install -c conda-forge pdfplumber chromadb`

### Q2: `bitsandbytes` 安装失败 / 报 `CUDA Setup failed`
**A:** 三种解决方案任选：
1. **CPU-only 模式**：跳过 bitsandbytes，按上面的方法不装它
2. **更新版本**：`pip install -U bitsandbytes`
3. **Windows 用户**：`bitsandbytes 0.43+` 才支持 Windows + CUDA 11.8+

### Q3: 启动报 `RuntimeError: No GPU found` / `CUDA not available`
**A:** 检查 PyTorch 是否识别到 GPU：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
- 返回 `False`：CUDA 没装好，重装 PyTorch：
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```
- 返回 `True` 但启动还报错：把 `config/config.yaml` 里的 `device` 改成 `cpu`

### Q4: 下载模型超时 / 失败
**A:** 必须用国内镜像：
```bash
# 设置环境变量后再装
$env:HF_ENDPOINT = "https://hf-mirror.com"   # PowerShell
# export HF_ENDPOINT=https://hf-mirror.com   # Bash
```

### Q5: ChromaDB 报 `sqlite3` 版本过旧（Linux）
**A:** 部分老 Linux 自带 sqlite < 3.35，安装新版：
```bash
# 在 requirements.txt 顶部添加：
pysqlite3-binary; sys_platform == 'linux'
```
然后在 `api/main.py` 开头加：
```python
import sys, pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

### Q6: 端口 8000 / 8501 被占用
**A:** 换端口：
```bash
python -m uvicorn api.main:app --port 8001
python -m streamlit run ui/app.py --server.port 8502
```

### Q7: Streamlit 一直转圈 / 不显示内容
**A:**
1. 看后端终端有没有报错
2. 浏览器开无痕模式排除缓存
3. 检查防火墙是否拦截了 8501

### Q8: "端口拒绝访问" / "Connection refused"
**A:**
- 后端没启动 / 启动失败 → 看后端终端日志
- 启动时加上 `--host 0.0.0.0` 才能从局域网访问

### Q9: 重新换电脑后数据没了？
**A:** 数据都存在 `data/` 目录下（向量库 + 用户库 + 上传文件）。
把这个目录整体备份/迁移即可恢复。

### Q10: `.env` / 密钥怎么处理？
**A:** `.env` 文件**不要**提交到 git（已在 `.gitignore` 中）。
提供 `.env.example` 作为模板，新电脑复制一份：
```bash
cp .env.example .env
# 编辑填入真实值
```

### Q11: 想换 LLM 模型怎么办？
**A:** 修改 `config/config.yaml`：
```yaml
llm:
  model_name: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"  # 换这里
  quantization:
    enabled: true
    quant_type: gptq  # 同步改这里
```
或者用 `.env` 临时覆盖：
```bash
LLM_MODEL_PATH=Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4
```

### Q12: 如何确认我的环境装好了？
**A:** 运行一键诊断脚本：
```bash
python -c "
import torch, transformers, chromadb, fastapi, streamlit
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
print('Transformers:', transformers.__version__)
print('ChromaDB:', chromadb.__version__)
print('FastAPI:', fastapi.__version__)
print('Streamlit:', streamlit.__version__)
print('All OK!')
"
```
输出 "All OK!" 就说明没问题。

---

## 📝 License

MIT License — 详见 `LICENSE`。

## 🙏 致谢

感谢开源社区，特别是 HuggingFace、Chroma、Streamlit、BAAI、阿里通义千问团队提供的高质量工具与模型。

---

> 💬 有任何问题或建议，欢迎提 Issue 或 PR。Happy RAG! 🚀


---


> 本项目代码同时托管在 Gitee：<https://gitee.com/tokyi/agent.git>
> GitHub: <https://github.com/你的用户名/ChineseRAGKB>

