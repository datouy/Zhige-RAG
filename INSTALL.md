# 安装文档（Installation Guide）

> 适用对象：本地开发 / 本地运行 / 单机部署
> 配套系统：中文知识库 RAG 系统（ChineseRAGKB）
> 关联文档：[README.md](README.md) · [DEPLOYMENT.md](DEPLOYMENT.md)

本系统为纯本地、零外部依赖的中文 RAG 知识库，支持 PDF / Word / Markdown / TXT 入库、本地 Embedding、4bit 量化本地 LLM、可选重排序、Streamlit Web UI 与 FastAPI 接口。

---

## 1. 环境要求

| 项目 | 最低要求 | 推荐 |
|------|----------|------|
| 操作系统 | Windows 10+ / Linux / macOS | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | **3.11** |
| 内存 | 16 GB | 32 GB |
| 硬盘 | 10 GB（不含模型） | SSD 20 GB+ |
| GPU | 可选（无 GPU 走 CPU） | GTX 1060 6GB / RTX 3050 8GB |
| 网络 | 可访问 HuggingFace 镜像（首次下载模型） | 同上 |

> ⚠️ **模型权重不入库**：`models/`（BGE、Qwen 等）体积达数 GB，已被 `.gitignore` 忽略，需本地自行下载或在部署时挂载，请见第 4 节。

---

## 2. 方式一：一键安装（推荐）

脚本会自动完成：创建虚拟环境 → 升级 pip → 检测 CUDA → 安装依赖（含 PyTorch）→ 下载默认模型。

### Windows

```powershell
cd ChineseRAGKB
python scripts\install.py
```

### Linux / macOS

```bash
cd ChineseRAGKB
bash scripts/install.sh
```

脚本特性：
- 自动检测 NVIDIA 显卡并选择 CUDA 11.8 版 PyTorch；无显卡则回退 CPU 版。
- 使用清华镜像（`pypi.tuna.tsinghua.edu.cn`）加速。
- Windows 下若 `chroma-hnswlib` 缺少预编译 wheel，会自动固定到 `0.7.5` 兼容版本重试。
- 通过 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像下载模型。

---

## 3. 方式二：手动安装

### 3.1 创建虚拟环境

```bash
cd ChineseRAGKB
python -m venv .venv
# Windows 激活
.venv\Scripts\activate
# Linux / macOS 激活
source .venv/bin/activate
```

### 3.2 安装依赖

```bash
python -m pip install --upgrade pip

# 有 NVIDIA GPU（Pascal 架构 GTX 10 系建议 cu118）
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 安装其余依赖
pip install -r requirements.txt
```

> 依赖版本冲突已通过 `requirements.txt` 与安装脚本中的固定组合（numpy 1.26.4 / transformers 4.46.1 / chromadb<0.6 等）收敛。如遇 `Microsoft Visual C++ 14.0` 报错，请安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)（勾选“使用 C++ 的桌面开发”）。

### 3.3 下载模型

```bash
export HF_ENDPOINT=https://hf-mirror.com      # Bash
# $env:HF_ENDPOINT="https://hf-mirror.com"   # PowerShell

huggingface-cli download BAAI/bge-small-zh-v1.5
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
# 可选：重排序模型（提升 Top-5 准确率约 10~20%）
huggingface-cli download BAAI/bge-reranker-base
```

模型默认存放到 `models/`（已被 `.gitignore` 忽略）。也可配置 `config/config.yaml` 中的 `embedding.model_name` / `llm.model_name` 指向本地路径或 HuggingFace ID，并设置 `local_files_only: true` 避免运行时联网。

### 3.4 配置环境变量

复制模板并填写真实值：

```bash
cp .env.example .env
```

关键变量见附录。**切勿将 `.env` 提交到 Git**（已在 `.gitignore` 中）。

---

## 4. 初始化数据库（可选）

系统默认使用 SQLite（`data/users.db`）。如需运行数据库迁移（Alembic）：

```bash
# 应用所有迁移
alembic upgrade head
# 查看当前版本
alembic current
```

生产环境推荐 PostgreSQL，通过 `DATABASE_URL` 环境变量覆盖（见 DEPLOYMENT.md）。

---

## 5. 启动

### 方式 A：一键启动（前后端 + 可选 LLM 预加载）

```bash
python scripts/start_all.py      # 或双击 start.bat (Windows)
```

启动后自动打开浏览器：
- 前端 UI：http://localhost:8501
- 后端 API：http://localhost:8000
- API 文档：http://localhost:8000/docs

### 方式 B：分别启动

```bash
# 终端 1：后端（FastAPI，端口 8000）
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 终端 2：前端（Streamlit，端口 8501）
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
```

### 首次使用

1. 进入“文档上传”页面，上传 PDF / DOCX / MD / TXT 资料完成入库。
2. 切换到“智能问答”开始提问。
3. 命令行体验：
   ```bash
   python scripts/ingest.py data/raw      # 入库示例文档
   python scripts/query.py "什么是 RAG？"  # 提问
   ```

---

## 6. 验证安装

运行一键诊断：

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

或直接访问 `http://localhost:8000/api/health` 查看健康检查。

---

## 7. 常见问题（FAQ）

**Q1：`pip install` 报 “Microsoft Visual C++ 14.0 or greater is required”**
安装 Visual Studio Build Tools（C++ 桌面开发），或用 conda：`conda install -c conda-forge pdfplumber chromadb`。

**Q2：`bitsandbytes` 安装失败 / `CUDA Setup failed`**
- CPU-only 模式可跳过该依赖。
- Windows 需 `bitsandbytes>=0.43` 才支持 CUDA 11.8+；执行 `pip install -U bitsandbytes`。

**Q3：启动报 `CUDA not available`**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
返回 `False`：重装 CUDA 版 PyTorch；或把 `config/config.yaml` 的 `device` 改为 `cpu`。

**Q4：模型下载超时**
务必设置 `HF_ENDPOINT=https://hf-mirror.com` 后再下载。

**Q5：端口 8000 / 8501 被占用**
```bash
python -m uvicorn api.main:app --port 8001
streamlit run ui/app.py --server.port 8502
```

**Q6：换 LLM 模型**
修改 `config/config.yaml`：
```yaml
llm:
  model_name: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
  quantization:
    enabled: true
    quant_type: gptq
```

---

## 附录：`.env` 关键变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `DB_PATH` | SQLite 路径 | `data/users.db` |
| `DATABASE_URL` | PostgreSQL 连接串（生产） | 空（用 SQLite） |
| `JWT_SECRET` | JWT 密钥（生产务必 ≥32 字符） | 占位值 |
| `JWT_ALGORITHM` | JWT 算法 | `HS256` |
| `LLM_MODEL_PATH` | 本地 LLM 路径 | `./models/Qwen2.5-1.5B-Instruct` |
| `DEVICE` | `auto` / `cuda` / `cpu` | `auto` |
| `USE_4BIT` | 是否 4bit 量化 | `false` |
| `ALLOWED_ORIGINS` | CORS 允许源（逗号分隔） | 本地 8501/8000 |
| `HF_ENDPOINT` | HuggingFace 镜像 | `https://hf-mirror.com` |
| `AUDIT_LOG_FILE` | 审计日志路径 | `logs/audit.jsonl` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

更多变量与说明见 `.env.example`。
