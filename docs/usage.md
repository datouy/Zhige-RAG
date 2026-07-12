# 使用指南

本文档给出从安装、启动到日常使用的完整流程。

## 1. 环境准备

### 1.1 硬件建议

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | GTX 1050 4GB | GTX 1060 6GB+ / RTX 3050 8GB |
| 内存 | 16 GB | 32 GB |
| 硬盘 | 10 GB 可用 | SSD，20 GB+ |
| Python | 3.10 | 3.10 / 3.11 |

> 本项目默认开启 4bit 量化，**GTX 1060 4GB** 可运行 Qwen2.5-1.5B-Instruct / Gemma-2-2B-IT。

### 1.2 系统依赖

- Windows / Linux / macOS
- CUDA Toolkit（仅在使用 GPU 时需要；推荐 11.8 或 12.1）
- Git（可选）

### 1.3 Python 环境

推荐使用 `conda` 或 `venv`：

```bash
conda create -n ragkb python=3.10 -y
conda activate ragkb
```

或：

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/macOS
```

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 若 GPU 为 Pascal 架构（GTX 10 系），请安装适配 CUDA 11.8 的 PyTorch：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```
> CPU-only 环境：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

## 3. 模型下载

### 3.1 Embedding 模型（必需，约 100MB）

```bash
# 国内镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir models/bge-small-zh-v1.5
```

或直接使用 `from_pretrained`，首次运行时会自动下载。

### 3.2 LLM 模型（必需，按显存选择）

| 模型 | 量化 | 显存 | 命令 |
|------|------|------|------|
| Qwen2.5-0.5B-Instruct | FP16 | ~1 GB | `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct` |
| Qwen2.5-1.5B-Instruct | 4bit | ~1.2 GB | `huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct` |
| Qwen2.5-3B-Instruct-GPTQ-Int4 | 4bit | ~2 GB | `huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` |
| gemma-2-2b-it | 4bit | ~1.8 GB | `huggingface-cli download google/gemma-2-2b-it` |

### 3.3 Reranker 模型（可选，约 700MB）

```bash
huggingface-cli download BAAI/bge-reranker-base --local-dir models/bge-reranker-base
```

## 4. 配置

复制环境变量模板：

```bash
cp .env.example .env
# Windows: copy .env.example .env
```

按需修改 `config/config.yaml` 中的：
- `embedding.model_name`
- `llm.model_name`
- `llm.quantization.enabled`
- `text_splitter.chunk_size`
- `retrieval.top_k`

## 5. 启动 Web UI

```bash
streamlit run ui/app.py --server.port 8501
```

打开浏览器访问 `http://localhost:8501`。

## 6. 命令行使用

### 6.1 入库

```bash
# 单个文件
python scripts/ingest.py data/raw/manual.pdf

# 整个目录（递归）
python scripts/ingest.py data/raw --recursive
```

### 6.2 查询

```bash
# 流式输出
python scripts/query.py "什么是 RAG？"

# 非流式 + 指定 top_k
python scripts/query.py "什么是 RAG？" --no-stream --top-k 3
```

### 6.3 评估

准备 `data/eval/eval_set.jsonl`（每行一个 JSON 对象）：

```json
{"question": "什么是 RAG？", "expected_sources": ["intro.pdf"], "expected_keywords": ["检索", "生成"]}
{"question": "请简述 RAG 的核心流程", "expected_keywords": ["向量", "提示"]}
```

运行评估：

```bash
python scripts/evaluate.py
```

报告输出至 `data/eval/report.json`。

## 7. 常见问题

### 7.1 bitsandbytes 安装失败（Windows）

Windows 上 `bitsandbytes<0.43` 不支持新 CUDA。可：
1. 升级：`pip install -U bitsandbytes`
2. 或关闭量化：`config.yaml` 中设置 `llm.quantization.enabled: false`，改用 FP16

### 7.2 Chroma 初始化报错（sqlite 版本）

升级 `pysqlite3-binary`：

```bash
pip install pysqlite3-binary
```

并在代码顶部加入：

```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

### 7.3 LLM 生成乱码

检查：
- 模型是否支持中文（Qwen2.5 / Baichuan2 / ChatGLM3 等均支持）
- `config.yaml` 的 `llm.chat_template`：`qwen` 模板对 Qwen 系最佳
- 是否把 tokenizer 的 `pad_token` 设错（默认自动设置）

### 7.4 GPU 显存不够（CUDA OOM）

- 启用 4bit 量化（默认开启）
- 减小 `chunk_size` 与 `top_k`
- 改用更小模型（Qwen2.5-0.5B-Instruct）
- 设置 `LLM_DEVICE=cpu` 强制走 CPU 推理（速度慢但稳定）

### 7.5 模型下载慢

设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或使用 `modelscope`：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen2.5-1.5B-Instruct', cache_dir='./models')"
```

## 8. 进阶定制

### 8.1 切换到 OpenAI 兼容 API

修改 `src/llm.py`，新增 `_call_api` 方法，并在 `RAGPipeline.ensure_llm` 中按配置选择 backend。

### 8.2 加入 BM25 混合检索

启用 `config.yaml` 中的 `vector_store.hybrid.enabled`，并在 `ChromaStore.query` 中增加稀疏召回（建议使用 `rank_bm25`）。

### 8.3 自定义 Prompt

编辑 `config/prompts.yaml`，无需改代码。

## 9. 部署建议

- **本地开发**：直接 `streamlit run`
- **生产部署**：建议改为 FastAPI + 前端，Streamlit 仅适合 Demo / 个人使用
- **离线场景**：提前下载所有模型到 `models/`，关闭外网即可运行

## 10. 路线图

- [ ] 多用户隔离（基于 session 的检索 session）
- [ ] 文档结构化解析（标题/表格/图片 OCR）
- [ ] Agent 工具调用（计算器 / SQL）
- [ ] FastAPI + WebSocket 流式问答服务
- [ ] 基于 LlamaIndex / LangGraph 的多跳推理

## 11. 指标可视化面板

侧栏新增 📊 **指标可视化** 页面，集中展示评估结果与历史趋势。

### 11.1 启动

```bash
streamlit run ui/app.py --server.port 8501
```

在浏览器左侧切换到「📊 指标可视化」即可。

### 11.2 数据来源

- 最新报告：`data/eval/report.json`（每次运行 `scripts/evaluate.py` 覆盖）
- 历史存档：`data/eval/reports/<YYYYMMDD-HHMMSS>.json`
- 索引摘要：`data/eval/reports/index.json`（最近 50 次精简条目）
- 历史记录为空时，页面会给出友好提示而不是报错。

### 11.3 页面内容

| 区块 | 说明 |
|------|------|
| 顶层指标卡 | 样本数、平均命中率、平均关键词覆盖率、平均延迟（ms），下方副卡含 NDCG@5 / MRR / 平均 TTFT / 平均 tokens/s |
| 逐题延迟分布 | 双子图：左为每条问题延迟柱状图，右为整体箱线图 |
| 命中率 vs 关键词覆盖率 | 散点图，颜色区分命中/未命中 |
| 历史报告对比 | 双 y 轴趋势线：左轴命中率+覆盖率（0~1），右轴平均延迟（ms） |
| 详情表 | 显示每条 question 的命中/覆盖率/NDCG/MRR/延迟/TTFT，支持问题关键词搜索 |
| 导出 | 一键下载当前快照为 HTML（含 summary + 历史表） |

### 11.4 含义说明

- **命中率（Retrieval Hit Rate）**：至少有一条命中文档 `source` 命中 `expected_sources` 的问题占比。
- **关键词覆盖率（Keyword Coverage）**：答案中包含 `expected_keywords` 的比例，越接近 1 越好。
- **NDCG@5**：把 Top-5 检索结果按是否命中 `expected_sources` 视作 0/1 相关性，按 NDCG 公式归一化；无期望来源时该指标返回 None（页面显示 —）。
- **MRR**：首个期望命中的倒数排名；同样在无期望来源时返回 None。
- **TTFT（首 token 延迟）**：从发起生成到第一个 token 抵达 UI 的耗时，反映"开卷反应速度"。
- **tokens/s**：去除首 token 等待后的纯生成速率；token 数为精确计数（tokenizer.encode），退化时按 1.5 字/token 估算并标记 `tokens_estimated=true`。

### 11.5 历史趋势如何累积

每次执行 `scripts/evaluate.py` 都会自动：

1. 在 `data/eval/report.json` 覆盖最新一份；
2. 在 `data/eval/reports/<时间戳>.json` 落盘一份精简报告（移除 `details[*].answer` 以减小体积）；
3. 在 `data/eval/reports/index.json` 追加一行，最近 50 条超出部分自动丢弃。

若想关掉存档可加 `--no-archive`。