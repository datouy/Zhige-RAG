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

> 详细设计、取舍与授权提醒见 **[DECISION_OPTIMIZATION.md](./DECISION_OPTIMIZATION.md)**。

> **开启前值得知道的一点**：`backend: llm` 是用生成式模型**模拟**决策，它本身仍是
> 一次自回归调用 —— 加一处决策就多一次生成的等待。所以默认全关是有意为之。
> 想真正拿到 d1 的毫秒级收益，要么用 `backend: d1`（需自备权重），要么只开那些
> **能省掉一次生成**的环节（`retrieval_sufficiency`、`query_route`）—— 这两处是
> 「多花一次短决策、少花一次长生成」，净赚。

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
# 必填一项：
#   POSTGRES_PASSWORD=<强口令>

docker compose up -d
```

compose 默认配置为 `LLM_BACKEND=ollama` + `EMBEDDING_BACKEND=ollama`，
并通过 `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1` 访问**宿主机上的 Ollama**
（容器内的 `localhost` 指向容器自身，这是最常见的踩坑点）。

数据库口令通过 `POSTGRES_PASSWORD` 注入：未设置时 compose 会直接报错退出，
而不是静默用弱口令启动。

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
├── requirements-core.txt    核心依赖（必装）
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

## 最近更新

### 2026-10-16：修掉"回答跑偏"的三个真因（检索串味 / 重排失效 / 相似度丢失）

现象：上传论文后问"用的是什么模型 完成的是什么功能？"，回答只复读了标题，
引用里还混进 Spring 面试题和一段通用 RAG 介绍。逐条查证后：

**1. 混合检索里关键词那一侧绕过了过滤（`src/vector_store.py`）**
BM25 只按自己的索引召回，**完全没应用 `where`**，于是被 `doc_status` / `acl`
排除的文档会从这条路溜回结果（越权 + 串味）。实测同样一条限定 `source` 的查询：
关掉混合是 6/6 命中本文档，开启混合后 3/6 越界。现在关键词侧用
`_metadata_matches` 做同条件过滤，认不出的算子一律保守排除。

**2. 关键词分数被当成"相似度"，还把无关文档顶到最前**
中文分词是 unigram+bigram 累加、没有查询长度归一，长文档极易拿高分：
一条 Node.js 面试题对上述提问的关键词分折算后 0.903，**压过真正相关片段的
0.546**。现在 BM25 分数带 `_score_kind=keyword` 标记不再冒充相似度，权重由
`vector_store.hybrid.bm25_weight` 控制（默认 0.5，只做召回补充）。
修复后该提问的 4 条引用**全部来自本文档**。

**3. 重排器一直在"假装工作"（`src/reranker.py`）**
FlagEmbedding 与当前 transformers 不兼容，`compute_score` 抛
`XLMRobertaTokenizer has no attribute prepare_for_model`，而异常被捕获后**静默
降级成初检顺序** —— `enabled: true` 开着却一次都没重排过。现在加载时自检一次，
坏了立刻切到 sentence-transformers 的 CrossEncoder（同一份本地权重）。

**顺带修好的：** 引用里相似度恒为 `—`（只取了 `_rerank_score`，未开重排时是
None）；顶栏"0 文档 / 0 分块"（前端一直拿着旧的知识库对象）；`min_relevance=0.3`
按 FlagEmbedding 的分数量程校准，换到 CrossEncoder 后会误杀候选，改为 0.05
（实测相关 ≥0.087、无关 ≤0.001）。

**效果（本机实测，同一提问）**：

| | 修复前 | 修复后 |
|---|---|---|
| 引用 | 2 条本文档 + Spring 面试题 + 通用 RAG 介绍 | 4 条全部来自本文档 |
| 回答 | 「用的是RAG技术完成的是基于本地化部署的中文RAG知识库系统的研究工作」（复读标题） | 「用的是 Qwen2.5-1.5B-Instruct…1.5B 参数 4bit 量化后约 1.2GB…」 |

新增 `scripts/kb_doctor.py`：体检"界面上的台账"与"检索真正搜到的 collection"
是否一致（默认只读，`--apply` 补登记、`--delete` 清理某个来源）。本机默认知识库
体检结果就是台账 1 文档/156 块、实际 9 文档/717 块 —— 那 8 份看不见的历史文档
正是串味的来源。

### 2026-10-15：本地冷启动与重复加载优化（不动模型，只改项目）

按「能复用就不重算」的思路把项目侧的固定开销清掉。实测数据来自本机
（Qwen2.5-1.5B + bge-small-zh，单卡 4bit）：

**1. 模型不再每问重载**（`src/rag_pipeline.py`）
多租户 / 多知识库路径用 `copy.copy` 复制全局 pipeline，而懒加载让副本的 `llm` 也是
`None`，于是**每个请求各自加载一遍权重**。实测连续 3 个请求加载 3 次模型。现在模型
实例放在所有副本共享的 `_llm_holder` 里，并加锁保证并发首问也只加载一次。
→ 每个问答省 **3.5s**。

**2. 权重就位后不再联网校验**（`config/local_overrides.yaml`）
界面「模型设置」把 `local_files_only` 写成了 `false`，导致**每次加载模型都先联网校验**
（走 `HF_ENDPOINT` 镜像），而权重其实早就在本地。改回 `true` 后：Embedding 首次加载
**26.5s → 0.2s**，租户工厂那次 **14.2s → 0.09s**。

**3. 启动期预热**（`api/main.py::_warm_up_models`）
把 `torch` / `transformers` / `sentence-transformers` 的导入、Embedding 模型、向量库与
pipeline 的初始化放到启动**后台线程**里做完（不阻塞服务就绪）。
→ 第一个检索请求 **7.78s → 0.86s**（约 9 倍）。`RAG_PRELOAD_LLM=1` 可连 LLM 一起预热。

**4. `RAG_LAZY_LLM=0` 真正生效**（`api/deps.py`）
原来写成 `bool(os.getenv(...))`，而字符串 `0` 在 Python 里是真值 —— 开关按了没反应。
已改用统一的 `env_flag()` 解析。

### 2026-10-15：d1 决策层扩展（再接入三处：充分性 / 分流 / 入库质检）

在已有四处（重排 / 路由 / 工具调度 / 校验）之上，继续按 d1 方法论补齐三处：

**新增能力**:
- ✅ **检索充分性决策** `RAGPipeline._assess_retrieval_sufficiency`（`noul` 原语）：
  生成前判定"片段够不够回答"，明显不足则**跳过生成直接拒答** —— 省下一次自回归解码，
  也从源头避免"检索不到却硬编"。`answer` / `stream_answer` / `answer_with_timing` 三入口一致。
- ✅ **差评自动分流** `src/feedback_router.py`（`choice` 原语）：`POST /api/v1/feedback`
  在 `question_type` 为空时自动归因到 `retrieval / answer / grounding / incomplete / format / other`，
  让"差评按类型归因"真正可用。
- ✅ **入库质检 / 合规门禁** `src/ingest_guard.py`（`score` + `noul` 原语）：语义质量分级
  （乱码 / 页眉页脚 / 目录残片 → 低质）+ 合规判定（涉黄涉暴 / 违规 → 可疑）；
  `action=flag` 只打标、`action=drop` 才拦截（默认 flag，非破坏）。

**Bug 修复**:
- ✅ `api/deps.py` 使用了未定义的 `logger`（决策层初始化失败时会抛 `NameError`）—— 补齐导入。

**设计要点**:
- 三处默认全关；开启前行为与改造前完全一致
- 决策失败一律静默降级（分流跳过、拒答跳过、门禁放行）
- 入库门禁两个原语在**同一次** `decide` 调用里问完（对齐 d1 `system_one` 共享一次前向）

**测试**: 新增 21 个用例（`tests/test_decision_extensions.py`），叠加原有 37 个

**文档**: [DECISION_OPTIMIZATION.md](./DECISION_OPTIMIZATION.md)

### 2026-10-15：d1 决策层优化（决策原语接入四处）

参考 Liquid AI d1 决策模型，新增 `src/decision_provider.py`，把
`noul` / `choice` / `score` 三种决策原语接入四个环节：

**新增能力**:
- ✅ **决策重排** `D1Reranker`（`score` 原语），与 bge 重排一键切换（`reranker.backend`）
- ✅ **决策路由**：`classify_query_with_decision` 在正则之上做语义意图判定
- ✅ **决策工具调度**：ReAct Agent 用 `choice` 预选工具并兜底，减少小模型空转
- ✅ **决策校验**：`verify_answer_with_decision` 用 `noul`+`score` 增强可验证性报告
- ✅ 双后端：原生 `LiquidAI/d1-3B` 或用现有本地模型（Qwen / Ollama）模拟

**设计要点**:
- 默认全关（`decision.enabled: false`），开启前行为与改造前完全一致
- 决策层任何失败都静默降级，绝不拖垮主链路
- 保守改判：把真问题判成寒暄需更高置信度

**测试**: 新增 37 个用例（`tests/test_decision_provider.py`），覆盖四种原语归一化与四处集成

**文档**: [DECISION_OPTIMIZATION.md](./DECISION_OPTIMIZATION.md)

### 2026-09-24：商用安全与体验增强（P1 级修复）

**新增功能**:
- ✅ **口令强度校验**：注册时强制 8 位 + 大小写 + 数字（防暴力破解）
- ✅ **流式元数据补全**：SSE 流 `done` 事件现包含 `sources`、`timings`、`metadata`
- ✅ **异常信息脱敏**：生产环境隐藏堆栈跟踪，返回友好错误信息 + 错误 ID

**Bug 修复**:
- ✅ 超时度量不准确（`ThreadPoolExecutor.result(timeout)` 包含排队时间）

**测试增强**:
- 新增 14 个测试（口令强度 9 个 + 异常脱敏 5 个）

**文档**:
- [修复报告](./P1_FIXES_REPORT.md) - 详细技术细节
- [修复摘要](./FIXES_SUMMARY.md) - 快速概览
- [前后对比](./P1_FIXES_COMPARISON.md) - 修复前后变化

**破坏性变更**: 无（新注册用户需使用强口令，现有用户不受影响）

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
