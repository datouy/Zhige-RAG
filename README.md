<div align="center">

# 知阁 · Zhige

**开箱即用的本地中文知识库 RAG 平台**

上传文档 → 自然语言提问 → 得到**带原文引用**的答案。数据全程留在你自己的机器上。

面向 **个人开发者** 与 **中小企业**。没有 GPU 也能跑。

`Python 3.10+` · `FastAPI` · `ChromaDB` · `BM25 + 向量 + RRF 混合检索` · `多模型后端`

</div>

> **本机优化预设**：i5-13400 / 32 GB RAM / GTX 1650 4 GB 使用
> `extractive + ONNX`。混合检索和原文溯源留在本机，生成式 LLM 完全不参与；
> Ollama、Qwen 权重、PyTorch/Transformers 与 3.2 GB 重排模型均已移除。

---

## 为什么做知阁

搭一套能用的本地 RAG，通常卡在这三件事上：

1. **必须先搞定模型** —— 大部分开源方案默认要你预先下载 GB 级权重，没下完就跑不起来；
2. **没有 GPU 就废了** —— 本地小模型效果有限，而接云端 API 又要改一堆代码；
3. **文档进不来** —— 上传了 PDF，结果解析失败，或者检索出来答非所问。

知阁把这三件事作为**默认路径**解决：

- **默认无 LLM**：原文摘录 + 引用校验；需要生成式回答时再选本地权重 / Ollama / OpenAI 兼容 API；
- **Embedding 同样可走 Ollama 或 API** —— 这意味着**完全不装 torch（省 2~3 GB）、不需要显卡**也能跑完整链路；
- **检索用混合策略**（BM25 关键词 + 向量语义 + RRF 融合 + 可选重排），比单一向量检索稳得多；
- 上手前先跑 `scripts/doctor.py`，它会直接告诉你缺什么、模型通不通、该怎么修。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **混合检索** | BM25 关键词 + 稠密向量召回，RRF（K=60）倒数排名融合 |
| **重排序** | 可选 bge 交叉编码重排，或 **d1 决策重排**（`score` 原语，毫秒级）；带最低相关度过滤（抑制"带引用的编造"） |
| **d1 决策层** | 可选：用决策原语（`noul`/`choice`/`score`）增强**重排 / 意图路由 / Agent 工具调度 / 答案校验 / 检索充分性 / 差评分流 / 入库质检**七处；支持原生 d1-3B 或用现有本地模型模拟 |
| **扫描件 OCR** | 图片型 PDF（扫描件 / 拍照件 / 传真件）自动补 OCR，**混合型 PDF 逐页补齐** |
| **网页导入** | 粘贴 URL 直接入库，自动提取正文并剥离导航/页脚噪声；内置 SSRF 防护 |
| **多知识库** | 按主题分库（人事制度 / 产品手册 / 客户合同），各自独立向量库，互不串数据 |
| **会话管理** | 会话列表 / 自动标题 / 历史回看 / 重命名 / 删除，问答持久化 |
| **知识图谱** | 可选 GraphRAG，支持 SQLite / Neo4j 双后端，含 Cypher 注入防护 |
| **引文可验证** | 答案标注来源片段，并校验引用覆盖率与接地性 |
| **多轮记忆** | 会话内短期记忆 + 跨会话长期记忆（支持显式"记住…"） |
| **Agent** | 内置 ReAct 多步推理与工具调用 |
| **多模型后端** | `local` / `ollama` / `openai`，LLM 与 Embedding 各自独立配置 |
| **双界面** | Web 前端（`ui/web`）+ Streamlit 控制台（`ui/`），共用同一套 REST API |
| **生产配套** | 限流、审计日志、配额、Prometheus 指标、k8s 清单 |

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
                    │  限流 · 审计 · 身份（本地固定用户）        │
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
git clone https://github.com/datouy/Zhige-RAG.git
cd Zhige-RAG

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 快速模式：服务 + PDF / Word / Markdown 解析，不含 torch
pip install -r requirements-quick.txt
pip install -r requirements-ocr.txt        # 需要处理扫描件（图片型 PDF）时，约 100~150 MB
```

> 三组依赖是**可选叠加**的：只装 `core` 就能跑通全部问答流程；
> 装上 `optional` 才能解析 PDF/Word；再装上 `ocr` 才能读扫描件。
> 不确定缺什么就跑 `python scripts/doctor.py`，它会逐项告诉你。

### 1. 当前默认：无 LLM 本地模式

本项目当前的默认配置为 `llm.backend: extractive` + `embedding.backend: onnx`：

- 文档和向量全部留在本机；
- 中文 BM25 + 本地 BGE 向量 + RRF 混合检索；
- 回答直接从命中段落抽取原文，并给每一句加上来源编号；
- 不安装、不加载 Ollama、Qwen、PyTorch、Transformers 或 CUDA 模型运行时；
- 向量模型约 24 MB，在 CPU 上运行。

```bash
# 首次准备本地向量模型（约 24 MB）
python scripts/setup_onnx_embedding.py

# 验证无 LLM 链路
python scripts/doctor.py
```

> 文章总结、改写和自由生成不属于无 LLM 模式。它返回可核验的知识库原文；
> 对制度查询、产品手册、合同与报告摘录等场景更直接，也不会产生模型编造。

#### 可选路线 A —— 云端 API

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

#### 可选路线 B —— 生成式本地模型

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

> 需要文章级总结或改写时再启用。GTX 1650 4 GB 更适合 0.5B~1.5B 的量化模型，
> 7B 以上模型会明显变慢，且可能溢出到系统内存。

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

**无需登录** —— 拉起来就能用。知阁是本地工具，不该拿注册登录挡人。

- **地址**：http://localhost:8000
- **能力**：多知识库切换 · 会话列表与历史回看 · 文件上传 · 网页导入 ·
  **界面上切换模型** · 流式问答与引用溯源（点引用看原文分块）
- 数据默认归属一个固定的「本地用户」，并给予不限额度 —— 这是你自己的机器。

> **本服务不提供登录与注册**：所有请求自动归属固定的「本地用户」，
> 不需要账号，也不做多用户隔离。要"一台服务器多人共用"得另行接入认证，
> 本项目刻意不带这些，见 `src/middleware/identity.py`。

### 在界面上换模型

左下角 **模型设置** → 选后端 → 填地址 → **测试连接** → 保存，**立即生效，不用重启**。

对话模型与向量模型可以分别设置：

| 后端 | 适用 | 需要什么 |
|------|------|----------|
| `extractive` | **当前默认**，稳定、可核验 | 无（直接返回知识库原文） |
| `onnx` | **当前默认向量后端**，CPU 推理 | 约 24 MB BGE 模型，无 torch |
| `local` | 完全离线、数据不出内网 | 装 `requirements-local.txt`（含 torch）、下载权重 |
| `ollama` | 需要生成式回答且接受 Ollama | 安装 Ollama + 拉取模型（可选） |
| `openai` | 用云端 API（DeepSeek / 通义 / 智谱 / vLLM…） | 一个 API Key |

界面的改动写在 `config/local_overrides.yaml`，**不会动 `config/config.yaml`**，
所以那份带注释的配置始终干净；删掉 overrides 文件即可恢复。

> 优先级：**环境变量 > 界面设置 > config.yaml**。
> 若某项已被环境变量指定，保存时界面会明确提醒你（避免"点了保存却没生效"的困惑）。

---

## d1 决策层（可选加速与增强）

参考 **Liquid AI d1 决策模型**：不做逐 token 生成，而是**单次前向**直接输出决策
（毫秒级、0 输出 token、无格式崩塌）。知阁把它的三种原语接到七个环节：

| 环节 | 原语 | 作用 |
|------|------|------|
| 检索重排 | `score` | 给每个候选分级打分，替代/补充 bge 交叉编码器 |
| 意图路由 | `choice` | 在正则规则之上做语义意图判定 |
| Agent 工具调度 | `choice` | 预选该调哪个工具，替代脆弱的 `Action:` 文本解析 |
| 答案校验 | `noul` + `score` | 语义接地判定 + 质量分级，补齐规则抓不到的"换说法编造" |
| 检索充分性 | `noul` | 片段明显答不了就直接拒答，**省下一次生成**、避免硬编 |
| 差评自动分流 | `choice` | 差评自动归因到处理组（对应"工单分流到业务组"） |
| 入库质检/合规 | `score` + `noul` | 语义质量分级 + 涉黄涉暴/违规判定 |

**默认全关**，开启后逐环节独立生效。两种后端：

- `backend: "llm"`（默认）—— 用你现有的 LLM（本地 Qwen / Ollama / 云端）模拟，
  **不装 torch、不下载新权重**；
- `backend: "d1"` —— 原生加载 `LiquidAI/d1-3B`（需 `requirements-d1.txt`），
  真正的单次前向，最快最稳。

```yaml
decision:
  enabled: true
  backend: "llm"        # llm | d1 | auto | none
  query_route: { enabled: true }
  tool_select: { enabled: true }
  verify:      { enabled: true }
  retrieval_sufficiency: { enabled: true }   # 检索不足 → 跳过生成直接拒答
  feedback_route:        { enabled: true }   # 差评自动归因
reranker:
  enabled: true
  backend: "d1"         # bge | d1 | auto
data_quality:
  ingest_guard:
    enabled: true       # 入库语义质检 + 合规判定
    action: "flag"      # flag=只打标；drop=拦截低质/可疑块
```

> 详细设计、取舍与授权提醒见 `config/config.yaml` 的 `decision` 段与 `requirements-d1.txt`。

> **开启前值得知道的一点**：`backend: llm` 是用生成式模型**模拟**决策，它本身仍是
> 一次自回归调用 —— 加一处决策就多一次生成的等待。所以默认全关是有意为之。
> 想真正拿到 d1 的毫秒级收益，要么用 `backend: d1`（需自备权重），要么只开那些
> **能省掉一次生成**的环节（`retrieval_sufficiency`、`query_route`）—— 这两处是
> 「多花一次短决策、少花一次长生成」，净赚。

## 关键配置

`config/config.yaml` 中改动最频繁的项：

| 配置项 | 作用 | 建议 |
|--------|------|------|
| `llm.backend` | 回答方式：`extractive` / `local` / `ollama` / `openai` | 当前为 `extractive` |
| `embedding.backend` | 向量模型来源：`onnx` / `local` / `ollama` / `openai` | 当前为 `onnx` |
| `retrieval.top_k` | 召回片段数 | 默认 4；资料多可调至 6~10 |
| `reranker.enabled` | 交叉编码重排 | 当前关闭，避免 3.2 GB 模型常驻 |
| `knowledge_graph.enabled` | 知识图谱增强 | 需要实体关系推理时开启 |
| `text_splitter.chunk_size` | 分块大小 | 中文 300~500 较合适 |
| `memory.enabled` | 多轮对话记忆 | 默认开启 |

**用环境变量覆盖（容器化推荐，不必改配置文件）：**

```bash
LLM_BACKEND=extractive             # 无生成式 LLM
EMBEDDING_BACKEND=onnx             # 本地 CPU 向量
# 需要云端后端时再启用：
# OPENAI_BASE_URL=https://api.deepseek.com/v1
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
# 必填一项：
#   POSTGRES_PASSWORD=<强口令>

docker compose up -d
```

compose 默认配置为 `LLM_BACKEND=extractive` + `EMBEDDING_BACKEND=onnx`，
不需要宿主机上的 Ollama，也不需要 GPU。

数据库口令通过 `POSTGRES_PASSWORD` 注入：未设置时 compose 会直接报错退出，
而不是静默用弱口令启动。

Kubernetes 清单见 `k8s/`。

---

## 项目结构

```
Zhige-RAG/
├── api/                     FastAPI 服务
│   ├── main.py              应用装配（中间件 / 路由 / 生命周期）
│   ├── deps.py              单例与共享依赖（模型、向量库、Pipeline）
│   └── routes/              各业务路由
├── src/                     领域层
│   ├── rag_pipeline.py      检索 → 重排 → 组装上下文 → 生成 → 校验
│   ├── decision_provider.py d1 决策原语（noul/choice/score）抽象与后端
│   ├── reranker.py          重排（bge 交叉编码 / d1 决策打分）
│   ├── feedback_router.py   差评自动分流（choice 原语 → 处理环节）
│   ├── ingest_guard.py      入库质检 / 合规门禁（score + noul 原语）
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
├── requirements-quick.txt   快速模式依赖（Ollama / API，开箱即用）
├── requirements-core.txt    核心服务依赖（必装）
├── requirements-local.txt   本地推理（torch，可选）
├── requirements-optional.txt 文档解析 / UI（可选）
├── requirements-ocr.txt     扫描件 OCR（可选）
├── requirements-gpu.txt     NVIDIA 量化（可选）
├── requirements-d1.txt      d1 决策模型（可选，仅 backend=d1 时需要）
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
需要 Streamlit、重排器等可选能力：`pip install -r requirements-optional.txt`。
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
  索引检查、备份、隔离与恢复见 [docs/OPERATIONS.md](docs/OPERATIONS.md)。
- **内部标识仍为 `ChineseRAGKB`**：日志前缀与 Python 包路径沿用原名，
  不影响使用；品牌展示名已统一为「知阁 Zhige」。

## 规划

- 图片理解（VLM 生成图片描述，让图文混排的文档也能被检索到）
- 知识库级别的共享权限（当前隔离粒度到用户）
- 文档在线预览与分块编辑

---

## 最近更新

详细变更记录见 [CHANGELOG.md](./CHANGELOG.md)。

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
