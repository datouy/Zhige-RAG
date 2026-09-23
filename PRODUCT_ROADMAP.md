# 对标 WeKnora 的差距分析与产品化路线图

**日期**：2026-09-23
**目标定位**：面向中小型企业与个人、开箱即用的本地 RAG 平台

---

## 一、先说结论

**1. 检索核心其实不弱，不需要重做。**
本项目已经具备 BM25 + 稠密向量 + RRF 融合 + Rerank + GraphRAG，
`src/vector_store.py` 的混合检索实现（RRF_K=60）与 WeKnora 的检索层是同一思路。
**功能广度不是当前瓶颈。**

**2. 真正的差距是「第一次运行能否成功」。**
WeKnora 拿到 28k star，核心不在于它支持 8 个向量库、10 个 IM 渠道，
而在于 `git clone` → 配 `.env` → `./scripts/start_all.sh` 就能跑起来。
本项目在此之前的状态是：**必须先自己搞定模型权重和环境，否则一行代码都跑不动**。

**3. 明确建议不要抄 WeKnora 的广度。**
8 个向量库、7 个对象存储、10 个 IM 渠道、Helm + E2B 沙箱、Organization 跨租户共享 ——
这些是**千人规模企业的需求**。对中小企业和个人，它们是部署负担和认知负担，不是功能优势。
把子弹打在"零门槛"上，比补齐功能清单更有价值。

---

## 二、能力对比

图例：✅ 已有　🟡 有但需完善　❌ 缺失

| 维度 | 本项目 | WeKnora | 判断 |
|------|--------|---------|------|
| 稠密向量检索 | ✅ Chroma | ✅ pgvector / ES / Milvus 等 8 种 | 单机场景 Chroma 足够，不补 |
| 关键词检索 | ✅ BM25 | ✅ BM25 | 持平 |
| 融合策略 | ✅ RRF (K=60) | ✅ RRF | 持平 |
| 重排序 | 🟡 有，默认关闭 | ✅ | 保持可选 |
| 知识图谱 | 🟡 SQLite / Neo4j，默认关闭 | ✅ | 够用 |
| **LLM 接入** | ✅ **已支持 local / Ollama / OpenAI 兼容**（本轮新增） | ✅ 20+ 提供商 | **本轮补齐，这是原本最致命的短板** |
| Embedding 接入 | ❌ 仅本地 sentence-transformers | ✅ BGE / GTE API + 本地 | **P0 待补** |
| 文档解析 | 🟡 pdf/docx/md/txt | ✅ + 版式分析 | 基础够用 |
| **OCR（扫描件）** | ❌ | ✅ PaddleOCR | **P1 刚需** |
| 网页抓取 | ❌ | ✅ | P1 |
| 图片理解（VLM） | ❌ | ✅ | P2 |
| ReAct Agent | ✅ | ✅ | 持平 |
| MCP 工具生态 | ❌ | ✅ 29 tools | P2 |
| 多租户 | 🟡 数据隔离 + is_admin | ✅ 租户/角色/组织/RBAC | 中小企业够用，不补细粒度 RBAC |
| 会话/对话管理 | 🟡 有记忆，无会话列表 UI | ✅ | P1 |
| **知识库概念层** | ❌ 只有底层 collection | ✅ 知识库→文档→会话 三层 | **P1，影响产品可理解性** |
| 凭据加密落盘 | ❌ 环境变量明文 | ✅ AES-256 | P2 |
| 零门槛启动 | 🟡 **本轮补齐**（doctor + README + 分层依赖 + 多后端） | ✅ 一键 | 继续打磨 |
| 部署形态 | 🟡 Docker / k8s | ✅ + Lite 单二进制 + 桌面 | P2 再看 |

---

## 三、明确「不做」的三件事

避免资源浪费，以下建议**明确排除**：

1. **多向量库适配层**。中小企业单机场景 ChromaDB 完全够用；
   抽象出 8 个后端只会让每次改动都要过一层接口，收益为负。
2. **IM 渠道接入（企业微信/飞书/Slack…）**。
   每个渠道都是独立的鉴权、消息格式、限流体系。中小企业的真实入口是 Web，
   要做也应该是"Web 优先 + 一个渠道验证需求"。
3. **Helm / 单二进制 / 桌面应用三形态并行**。
   当前 Docker Compose 已覆盖绝大多数私有化场景，先做深不做宽。

---

## 四、路线图

### ✅ P0 —— 本轮已落地

| 项 | 交付物 |
|----|--------|
| 多模型后端 | `src/llm_provider.py`（零新增依赖）+ `RAGPipeline._build_llm_from_cfg` 分派，API/UI/脚本全链路生效 |
| 本地服务绕代理 | 修掉"配了 Ollama 却因全局 `HTTP_PROXY` 永远连不上"（表现为 502） |
| 环境自检 | `scripts/doctor.py`：依赖分组、模型可用性、Ollama 连通性、端口、代理提示 |
| 依赖分层 | `requirements-core/local/gpu/optional/dev.txt`；**`bitsandbytes` 移出默认安装**（Windows 装不上） |
| 上手文档 | `README.md`：三条模型路线 + FAQ + 已知限制 |
| CI 提速 | 不再安装 torch（2~3 GB），从"装 10 分钟"到秒级，且测试真正可跑 |
| 配置文件 | `config.yaml` 增加 `llm.backend` 与三后端配置块 |

### ✅ P0 —— 已完成（2026-09-23 第三轮）

> 本节三项已全部实现，并**通过端到端实测**（不装 torch / 无 GPU / 无本地模型文件，
> 跑通 上传→入库→检索→生成 全链路）：
>
> | 原计划项 | 实际交付 |
> |---|---|
> | Embedding 后端抽象 | `src/embeddings_provider.py`（local / ollama / openai）+ `src/http_client.py` 共享 HTTP 层；统一了 6 处直连构造点 |
> | 一键启动脚本 | `scripts/quickstart.py`（检查 → 装依赖 → 验模型 → 起服务） |
> | Docker 默认走 Ollama | `Dockerfile.lite`（~1 GB，不含 torch）+ compose 默认 ollama + `extra_hosts` |
>
> 另交付：命名「知阁 Zhige」、README 重写、`doctor` 支持 embedding 后端分支、
> 依赖分 5 组、环境变量切换后端。

### ✅ P1 —— 已完成（2026-09-23 第四轮）

> 前四项已实现并**逐项端到端实测**：

| 项 | 实际交付 |
|----|----------|
| **扫描件 OCR** | `document_loader._augment_pdf_with_ocr`，RapidOCR + pymupdf，**逐页判断**（混合型 PDF 也能处理），`requirements-ocr.txt`。实测：图片型 PDF 成功提取中文 |
| **网页 URL 导入** | `src/web_loader.py` + `POST /api/v1/ingest/url`，**多层 SSRF 防护**（解析后 IP 校验 / 逐跳重定向 / 大小与类型限制）。实测：7 类危险地址全部拦截 |
| **知识库概念层** | 新表 `knowledge_bases` + `src/kb_service.py` + `api/routes/kb.py`；collection 级隔离，默认库沿用历史命名不破坏老数据。实测：**跨库数据不泄露** |
| **会话管理** | `chat_sessions` / `chat_messages` 持久化，自动标题 + 列表预览 + 重命名 + 删除 + 历史回看。实测通过 |

### ✅ 已收尾（2026-09-23 第五轮）

| 项 | 实际交付 |
|----|----------|
| **前端接入新能力** | `ui/web` 重写（106 行 HTML + 658 行 JS + 278 行 CSS）：登录注册、知识库切换/新建/删除、会话列表与历史回看、文件上传（带 kb_id）、网页导入、流式问答、引用溯源 |
| **前端收敛** | `start_all.py` 默认**只启动 Web（8000）**；Streamlit 降级为调试工具，用 `--with-streamlit` 显式开启 |

> ⚠️ 接手时发现老前端**本来就连不上后端**：WebSocket 不带 token（被 4401 拒）、
> REST 调用不带认证头、没有登录入口、还有多处 DOM id 与 JS 引用不匹配导致功能静默失效。
> 所以这一轮实质是**把它修到能用**，而不是单纯加功能。
>
> 验证：模拟前端完整调用链（含**真实 WebSocket** 流式提问）11 个接口全部打通；
> 静态一致性检查（DOM id / CSS 类 / API 路径）全通过。
> **未做**：浏览器渲染级验证（`agent-browser` 下载 Chromium 在代理环境卡住），
> 页面视觉仍需人眼确认一次。

### 🟡 P2 —— 有明确需求时再做

- API Key 管理与对外 OpenAPI（供第三方系统集成）
- 凭据 AES-256 落盘加密（对齐 WeKnora 的 `SYSTEM_AES_KEY`）
- Wiki 模式（文档自动生成互链 Markdown）
- MCP 工具接入
- 单二进制分发（PyInstaller / Nuitka）

---

## 五、本轮已修复的相关技术债

结合上一轮审计，本轮一并处理的与本主题强相关项：

- `scripts/synthesize.py` 里那套 local/openai/ollama 的 `_LLMClient` 是**半成品**（ollama 分支从未实现，调用即抛"未就绪"），且与主链路重复。本轮把能力提升到 `src/llm_provider.py` 并**真正实现**，主链路与脚本从此共用同一套后端逻辑。**建议后续删除 `synthesize.py` 里的私有实现**，改为导入 `src.llm_provider`。
- `config.yaml` 中 `llm.local_files_only: true` 与"没有模型权重"的组合，
  是最典型的"下载后必然启动失败"配置。已在 `doctor.py` 中显式检测并给出三选一处置建议。

---

## 六、建议的执行顺序

```
本轮已完成
    ↓
1. Embedding 后端抽象        ← 决定"换后端能否彻底不下载模型"
    ↓
2. quickstart + Docker.lite  ← 决定"新人能否 5 分钟跑通"
    ↓
3. OCR + URL 导入            ← 决定"文档能不能真的进来"
    ↓
4. 知识库概念层 + 会话管理   ← 决定"产品能否被理解和日常使用"
```

前三步做完，"开箱即用的本地 RAG 平台"这个定位才算立住；
第 4 步做完，才能谈"面向中小企业"的产品形态。
