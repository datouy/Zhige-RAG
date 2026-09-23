<div align="center">

# 知阁 · Zhige

**开箱即用的本地中文知识库 RAG 平台**

上传文档 → 自然语言提问 → 得到**带原文引用**的答案。数据全程留在你自己的机器上。

面向 **个人开发者** 与 **中小企业**。没有 GPU 也能跑。

`Python 3.10+` · `FastAPI` · `ChromaDB` · `BM25 + 向量 + RRF 混合检索` · `多模型后端`

</div>

---

## 为什么做知阁

搭一套能用的本地 RAG，通常卡在这三件事上：

1. **必须先搞定模型** —— 大部分开源方案默认要你预先下载 GB 级权重，没下完就跑不起来；
2. **没有 GPU 就废了** —— 本地小模型效果有限，而接云端 API 又要改一堆代码；
3. **文档进不来** —— 上传了 PDF，结果解析失败，或者检索出来答非所问。

知阁把这三件事作为**默认路径**解决：

- **三种模型来源随时切换**，改一行配置或一个环境变量即可：本地权重 / **Ollama** / **任意 OpenAI 兼容 API**；
- **Embedding 同样可走 Ollama 或 API** —— 这意味着**完全不装 torch（省 2~3 GB）、不需要显卡**也能跑完整链路；
- **检索用混合策略**（BM25 关键词 + 向量语义 + RRF 融合 + 可选重排），比单一向量检索稳得多；
- 上手前先跑 `scripts/doctor.py`，它会直接告诉你缺什么、模型通不通、该怎么修。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **混合检索** | BM25 关键词 + 稠密向量召回，RRF（K=60）倒数排名融合 |
| **重排序** | 可选 bge-reranker 交叉编码重排，并带最低相关度过滤（抑制"带引用的编造"） |
| **扫描件 OCR** | 图片型 PDF（扫描件 / 拍照件 / 传真件）自动补 OCR，**混合型 PDF 逐页补齐** |
| **网页导入** | 粘贴 URL 直接入库，自动提取正文并剥离导航/页脚噪声；内置 SSRF 防护 |
| **多知识库** | 按主题分库（人事制度 / 产品手册 / 客户合同），各自独立向量库，互不串数据 |
| **会话管理** | 会话列表 / 自动标题 / 历史回看 / 重命名 / 删除，问答持久化 |
| **知识图谱** | 可选 GraphRAG，支持 SQLite / Neo4j 双后端，含 Cypher 注入防护 |
| **引文可验证** | 答案标注来源片段，并校验引用覆盖率与接地性 |
| **多轮记忆** | 会话内短期记忆 + 跨会话长期记忆（支持显式"记住…"） |
| **Agent** | 内置 ReAct 多步推理与工具调用 |
| **多租户** | 按用户隔离向量库与图谱库，互不可见 |
| **多模型后端** | `local` / `ollama` / `openai`，LLM 与 Embedding 各自独立配置 |
| **双界面** | Web 前端（`ui/web`）+ Streamlit 控制台（`ui/`），共用同一套 REST API |
| **生产配套** | JWT 认证、限流、审计日志、配额、Prometheus 指标、k8s 清单 |

---

## 架构

```
                    ┌──────────────────────────────────────────┐
                    │  入口：Web 前端 (/)  ·  Streamlit (:8501)  │
                    │        REST API (:8000)  ·  WebSocket     │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │  FastAPI（api/）                          │
                    │  认证 · 限流 · 审计 · 配额 · 多租户隔离    │
                    └───────────────────┬──────────────────────┘
                                        │
        ┌───────────────────────────────▼───────────────────────────────┐
        │  RAG Pipeline（src/rag_pipeline.py）                          │
        │                                                               │
        │   查询 → 检索 ─┬─ 向量召回（ChromaDB）                        │
        │               └─ 关键词召回（BM25）                            │
        │                  ↓ RRF 融合 → 可选 Rerank                     │
        │               → 上下文组装（含图谱增强）→ 生成 → 引文校验      │
        └───────┬─────────────────────┬─────────────────────┬──────────┘
                │                     │                     │
        ┌───────▼────────┐   ┌────────▼────────┐   ┌────────▼─────────┐
        │ LLM 后端        │   │ Embedding 后端   │   │ 存储             │
        │ local/ollama/   │   │ local/ollama/    │   │ Chroma · SQLite  │
        │ openai          │   │ openai           │   │ Neo4j · Redis    │
        └─────────────────┘   └──────────────────┘   └──────────────────┘
                    ↑                     ↑
              src/llm_provider.py   src/embeddings_provider.py
              （零第三方依赖，标准库 HTTP + SSE）
```

---

## 快速开始

### 0. 准备环境

```bash
git clone https://github.com/datouy/rag.git
cd rag

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 只装核心依赖（不含 torch，通常几分钟）
pip install -r requirements-core.txt
pip install -r requirements-optional.txt   # 需要解析 PDF / Word 时
pip install -r requirements-ocr.txt        # 需要处理扫描件（图片型 PDF）时，约 100~150 MB
```

> 三组依赖是**可选叠加**的：只装 `core` 就能跑通全部问答流程；
> 装上 `optional` 才能解析 PDF/Word；再装上 `ocr` 才能读扫描件。
> 不确定缺什么就跑 `python scripts/doctor.py`，它会逐项告诉你。

### 1. 选一条模型路线

模型来源由 `config/config.yaml` 的 `llm.backend` 与 `embedding.backend` 决定。**三条路线任选一条**：

#### 路线 A —— Ollama（推荐：无需 GPU，也无需下载权重）

```bash
# 安装 Ollama: https://ollama.com/download
ollama pull qwen2.5:7b          # 对话模型，约 4.7 GB
ollama pull nomic-embed-text    # 向量模型，约 270 MB
ollama serve
```

```yaml
# config/config.yaml
llm:
  backend: "ollama"
embedding:
  backend: "ollama"
```

#### 路线 B —— 云端 API（最快：连显卡都不需要）

在 `.env` 中填密钥：

```bash
OPENAI_API_KEY=sk-your-key-here
```

```yaml
# config/config.yaml
llm:
  backend: "openai"
  openai:
    model: "deepseek-chat"
    base_url: "https://api.deepseek.com/v1"
embedding:
  backend: "openai"          # 注意：Embedding 与 LLM 可分别设置
  openai:
    model: "text-embedding-3-small"
    base_url: "https://api.openai.com/v1"
```

其他常用的 OpenAI 兼容服务：

| 服务 | base_url | 对话模型示例 | 向量模型示例 |
|------|----------|--------------|--------------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` | — |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` | `text-embedding-v3` |
| 智谱清言 | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` | `embedding-3` |
| 硅基流动 | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-7B-Instruct` | `BAAI/bge-m3` |
| 本地 vLLM | `http://localhost:8000/v1` | 你部署的模型 | 你部署的模型 |

#### 路线 C —— 完全本地推理（数据不出内网）

```bash
pip install -r requirements-local.txt     # 会安装 torch，约 2~3 GB
pip install -r requirements-gpu.txt       # 仅 NVIDIA GPU 需要（4bit 量化省显存）
```

```yaml
# config/config.yaml
llm:
  backend: "local"
  local_files_only: false        # 首次设为 false 让它自动下载权重
embedding:
  backend: "local"
  local_files_only: false
```

> ⚠️ `local_files_only: true` 表示"只用本地已有模型、绝不联网"。
> 权重还没下载时会直接启动失败 —— 这是新手最常踩的坑，先跑 `doctor.py` 可提前发现。

### 2. 环境自检

```bash
python scripts/doctor.py
```

逐项检查 Python 版本、依赖、模型是否真的可用（Ollama 通不通 / API Key 配没配 / 本地权重在不在）、
端口占用、代理影响，并对每个 ❌ 给出**可直接执行**的处理建议。

### 3. 启动

一条命令搞定（检查 → 缺依赖提示 → 起服务）：

```bash
python scripts/quickstart.py
```

或者分步来：

```bash
python scripts/start_all.py      # Windows 也可双击 start.bat
```

启动后：

| 入口 | 地址 |
|------|------|
| Web 前端 | http://localhost:8000 |
| Streamlit 控制台 | http://localhost:8501 |
| API 文档（Swagger） | http://localhost:8000/docs |

日志：`logs/backend.log`、`logs/frontend.log`

---

## 界面

**默认无需登录** —— 拉起来就能用。知阁是本地工具，不该拿注册登录挡人。

- **地址**：http://localhost:8000
- **能力**：多知识库切换 · 会话列表与历史回看 · 文件上传 · 网页导入 ·
  **界面上切换模型** · 流式问答与引用溯源（点引用看原文分块）
- 数据默认归属一个固定的「本地用户」，并给予不限额度 —— 这是你自己的机器。

> 需要"一台服务器多人共用"时，把 `config/config.yaml` 的 `auth.enabled` 设为 `true`
> （或设环境变量 `AUTH_ENABLED=1`），即启用登录注册与租户隔离：
> 各人只能看到自己的知识库。默认是**关闭**的。

### 在界面上换模型

左下角 **模型设置** → 选后端 → 填地址 → **测试连接** → 保存，**立即生效，不用重启**。

对话模型与向量模型可以分别设置。三种来源：

| 后端 | 适用 | 需要什么 |
|------|------|----------|
| `local` | 完全离线、数据不出内网 | 装 `requirements-local.txt`（含 torch）、下载权重 |
| `ollama` | 想省事又要本地（**推荐**） | 装 Ollama + `ollama pull` 模型，**不需要 torch 和显卡** |
| `openai` | 用云端 API（DeepSeek / 通义 / 智谱 / vLLM…） | 一个 API Key |

界面的改动写在 `config/local_overrides.yaml`，**不会动 `config/config.yaml`**，
所以那份带注释的配置始终干净；删掉 overrides 文件即可恢复。

> 优先级：**环境变量 > 界面设置 > config.yaml**。
> 若某项已被环境变量指定，保存时界面会明确提醒你（避免"点了保存却没生效"的困惑）。

## 关键配置

`config/config.yaml` 中改动最频繁的项：

| 配置项 | 作用 | 建议 |
|--------|------|------|
| `llm.backend` | 对话模型来源：`local` / `ollama` / `openai` | 无 GPU 选 `ollama` 或 `openai` |
| `embedding.backend` | 向量模型来源，可独立于 LLM 设置 | 同理；选前两者则**不需要 torch** |
| `llm.local_files_only` | 是否禁止联网下载权重 | 首次部署设为 `false` |
| `retrieval.top_k` | 召回片段数 | 默认 8；资料多可调至 10~15 |
| `reranker.enabled` | 交叉编码重排 | 追求准确率时打开（更慢，需额外模型） |
| `knowledge_graph.enabled` | 知识图谱增强 | 需要实体关系推理时开启 |
| `text_splitter.chunk_size` | 分块大小 | 中文 300~500 较合适 |
| `memory.enabled` | 多轮对话记忆 | 默认开启 |

**用环境变量覆盖（容器化推荐，不必改配置文件）：**

```bash
LLM_BACKEND=ollama                 # 切换对话后端
EMBEDDING_BACKEND=openai           # 切换向量后端
OLLAMA_BASE_URL=http://localhost:11434/v1
OPENAI_BASE_URL=https://api.deepseek.com/v1
```

> 切换 Embedding 模型会改变向量维度。非 `local` 后端会自动为 collection 追加
> "后端+模型"指纹，避免与旧向量混库。

---

## Docker 部署

提供两个镜像：

| 镜像 | 依赖 | 体积 | 适用 |
|------|------|------|------|
| `Dockerfile.lite` | 不含 torch | ~1 GB | **默认推荐**：模型走 Ollama 或云端 API |
| `Dockerfile` | 含 torch | ~10 GB | 需要在容器内做本地推理 |

```bash
cp .env.example .env
# 必填两项：
#   JWT_SECRET=<32 位随机串>
#   POSTGRES_PASSWORD=<强口令>
python -c "import secrets; print(secrets.token_urlsafe(32))"

docker compose up -d
```

compose 默认配置为 `LLM_BACKEND=ollama` + `EMBEDDING_BACKEND=ollama`，
并通过 `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1` 访问**宿主机上的 Ollama**
（容器内的 `localhost` 指向容器自身，这是最常见的踩坑点）。

生产环境请务必设置强 `JWT_SECRET`：容器内已启用 `REQUIRE_STRONG_JWT=1`，
密钥缺失、短于 32 字符、或仍是模板占位符时会**拒绝启动**（有意设计，防止用默认弱密钥上线）。

Kubernetes 清单见 `k8s/`。

---

## 项目结构

```
rag/
├── api/                     FastAPI 服务
│   ├── main.py              应用装配（中间件 / 路由 / 生命周期）
│   ├── deps.py              单例与共享依赖（模型、向量库、Pipeline）
│   └── routes/              各业务路由
├── src/                     领域层
│   ├── rag_pipeline.py      检索 → 重排 → 组装上下文 → 生成 → 校验
│   ├── llm_provider.py      LLM 后端抽象（local / ollama / openai）
│   ├── embeddings_provider.py  Embedding 后端抽象
│   ├── http_client.py       标准库实现的零依赖 HTTP / SSE 客户端
│   ├── vector_store.py      Chroma + BM25 + RRF 混合检索
│   ├── document_loader.py   多格式解析（含扫描件 OCR 兜底）
│   ├── web_loader.py        网页抓取与正文提取（含 SSRF 防护）
│   ├── kb_service.py        知识库与会话的业务逻辑
│   ├── kg/                  知识图谱（抽取 / 存储 / 检索 / GraphRAG）
│   ├── agent/               ReAct Agent 与工具注册表
│   └── memory.py            短期 / 长期记忆
├── scripts/                 离线工具（不被运行时导入）
│   ├── quickstart.py        一键启动
│   ├── doctor.py            环境自检 ← 出问题先跑这个
│   ├── ingest.py            文档入库
│   ├── evaluate.py          效果评估
│   └── preload_llm.py       模型预下载
├── ui/                      界面
│   ├── web/                 Web 前端（由 FastAPI 挂载在 /）
│   └── app.py               Streamlit 控制台
├── config/config.yaml       主配置
├── data/                    知识库、向量库、上传文件
├── requirements-core.txt    核心依赖（必装）
├── requirements-local.txt   本地推理（torch，可选）
├── requirements-optional.txt 文档解析 / UI（可选）
├── requirements-ocr.txt     扫描件 OCR（可选）
├── requirements-gpu.txt     NVIDIA 量化（可选）
└── tests/                   测试
```

---

## 常见问题

**Q：启动报错找不到模型，或卡在加载**
先跑 `python scripts/doctor.py`。绝大多数情况是 `local_files_only: true` 但权重没下载，
或本来就该用 `backend: ollama`。

**Q：Ollama 明明启动了，程序却连不上**
如果本机配置了 `HTTP_PROXY` / `HTTPS_PROXY`，请把本地地址加入 `NO_PROXY`。
知阁对 `localhost` / 内网地址已自动绕过代理，但企业网关仍可能拦截请求。
（这也是本项目的实测发现：错误提示若显示 HTTP 502 而非"连接被拒绝"，基本都是代理导致的。）

**Q：Docker 里连不上宿主机的 Ollama**
容器内的 `localhost` 指向容器自己。用 `http://host.docker.internal:11434/v1`
（compose 已配好 `extra_hosts`；Linux 手动 `docker run` 时需加
`--add-host=host.docker.internal:host-gateway`）。

**Q：没有 GPU，本地模型能跑吗**
能，但 1.5B 级模型效果有限。**推荐用 Ollama 跑 7B 模型** —— 消费级 CPU 即可响应，
效果显著优于小参数量本地推理。

**Q：上传 PDF 没反应**
需要解析依赖：`pip install -r requirements-optional.txt`。
扫描件（图片型 PDF）目前不支持 OCR，见下方「已知限制」。

**Q：切换了 Embedding 模型后检索结果不对**
向量维度变了。非 `local` 后端会自动使用带指纹的 collection 名（如
`chinese_rag_kb__openai_text_embedding_3_small`），旧数据不会混入；
但需要**重新入库**文档才能被检索到。

---

## 已知限制

- **OCR 需额外安装**：扫描件能力依赖 `requirements-ocr.txt`（约 100~150 MB）。
  未安装时图片型 PDF 提不出文本，其它格式不受影响。
- **网页导入只覆盖静态 HTML**：纯前端渲染（SPA）的页面抓不到正文。
- **向量库仅 ChromaDB**：单机 / 中小规模够用，未接入 pgvector、Milvus 等。
- **内部标识仍为 `ChineseRAGKB`**：日志前缀与 Python 包路径沿用原名，
  不影响使用；品牌展示名已统一为「知阁 Zhige」。

## 规划

- 图片理解（VLM 生成图片描述，让图文混排的文档也能被检索到）
- 知识库级别的共享权限（当前隔离粒度到用户）
- 文档在线预览与分块编辑

---

## 开发

```bash
pip install -r requirements-core.txt -r requirements-dev.txt

pytest tests/ -v                                 # 测试
ruff check src/ api/                             # 代码检查
python -m compileall -q src api scripts tests ui # 语法闸门
```

提交前请确保以上三条通过。CI 配置见 `.github/workflows/ci.yml`。

---

## 许可

详见 [LICENSE](LICENSE)。
