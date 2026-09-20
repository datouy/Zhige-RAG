# ChineseRAGKB 代码审查报告

审查范围：`src/`（RAG 核心链路）、`api/`（接口层）、`scripts/ingest.py`（入库链路）
审查方式：静态代码走查 + 关键路径实测复现
测试基线：`256 passed, 1 skipped`（全绿，但存在覆盖盲区，见 M-9）

---

## 一、结论摘要

| 严重度 | 数量 | 代表问题 |
|---|---|---|
| 高 | 5 | 标题行入库丢失、count 缓存致检索永久为空、LLM 并发无保护、WS 线程泄漏、分句去重/丢失 |
| 中 | 11 | context 预算低估 2 倍、流式路径绕过重排候选、阈值 0 诱发幻觉、4 处重复代码等 |
| 低 | 6 | 弃用 API、参数失效、锁序隐患等 |

**最需要立即处理的三个问题**（均可被用户直接感知）：

1. **标题行在入库时被静默丢弃** —— 实测「第三章 系统设计与实现」「表1 各模块职责对照」等标题 100% 丢失，它们是检索锚点，直接影响召回。
2. **`count()` 缓存跨进程失效** —— 服务启动后经 CLI 入库，运行中的服务**永远检索不到新文档**（实测 5 条数据命中 0 条）。
3. **共享 LLM 无并发保护** —— 多用户并发请求会同时进入 `model.generate()`。

---

## 二、高严重度问题（High）

### H-1 分句逻辑吞掉标题行 / 无句末标点的行 → 入库内容永久丢失

**位置**：`src/text_splitter.py:55-66`（`_split_into_sentences`）

**根因**：句子正则 `_SENT_END_RE` 的字符类排除了 `\n`，匹配会**跳过**换行符；而函数用
`consumed = sum(len(p) for p in parts)` 反推游标位置，该累加值小于真实消费长度，
导致 `tail = text[consumed:]` 与已匹配区域重叠，同时中间未被匹配的行被整体丢弃。

**实测复现**（`chunk_size=300, chunk_overlap=50`）：

```
原文 197 字符 → 分块后 202 字符（净增 +5，即存在重复）
丢失行数 3：
  - '第三章 系统设计与实现'   出现次数 1 → 0
  - '3.1 模块划分'            出现次数 1 → 0
  - '表1 各模块职责对照'       出现次数 1 → 0
```

对比控制组：若所有行都以「。」结尾，`\s*` 会吸收换行，bug 不触发（1008 → 1008，无损失）。
**因此触发条件是：行尾没有句末标点** —— 恰恰是标题、小节名、表题、列表项、注释行，
即文档中信息密度最高、最需要被检索到的部分。

**影响**：PDF 抽取文本与 Markdown 普遍存在标题行，这类内容在入库阶段即丢失，无法被任何查询召回，
且过程无告警。属于**静默数据损坏**。

**修复建议**：改用 `finditer` 记录真实游标，并保留匹配间隙：

```python
def _split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts: List[str] = []
    pos = 0
    for m in _SENT_END_RE.finditer(text):
        gap = text[pos:m.start()].strip()      # ← 关键：保留被跳过的标题行
        if gap:
            parts.append(gap)
        parts.append(m.group(1))
        pos = m.end()                          # ← 关键：真实游标
    tail = text[pos:].strip()
    if tail:
        parts.append(tail)
    return [p.strip() for p in parts if p and p.strip()]
```

**已实测验证**：修复后上述用例丢失 0 行、净增减 0，且 6 个原有正确用例（纯中文、纯英文、
空串、无标点、双换行分段等）行为完全不变。

---

### H-2 `count()` 缓存跨进程失效 → 检索永久返回空

**位置**：`src/vector_store.py:199`（`query` 提前返回）、`323-329`（`count` 缓存）

**根因**：`count()` 结果缓存在实例上，仅在本实例写入/删除时失效。
`query()` 开头有 `if self.count() == 0: return []`，一旦缓存为 0，
即便其他进程（CLI 入库脚本、其他 uvicorn worker）已写入数据，本实例仍返回空。

**实测复现**：

```
进程A 初始 count(): 0 (已写入缓存)
进程B 写入后 count(): 5
进程A count()（缓存值）: 0
进程A 磁盘真实 count() : 5
进程A query() 命中数  : 0     ← 期望 5
```

**影响**：生产典型场景 —— 先启动 API 服务，再用 `python scripts/ingest.py` 入库，
**运行中的服务将永远检索不到这批文档，必须重启**。这是用户会直接遭遇的“入库成功但搜不到”。
此外 `list_documents` 同样受影响。

**修复建议**（按优先级）：

1. 最简：让 `count()` 缓存带 TTL，或直接去掉缓存（`collection.count()` 本身是轻量 SQLite 查询，
   缓存收益有限却引入一致性风险）。
2. 更彻底：`query()` 移除 `count() == 0` 的提前返回分支 —— Chroma 对空集合查询本就返回空，
   这个分支除了制造失效面，没有实际收益。

```python
def query(self, query_text, top_k=5, where=None, score_threshold=0.0):
    # 删掉 if self.count() == 0: return []
    results = self.query_batch(...)
    return results[0] if results else []
```

---

### H-3 共享 LLM 实例无并发保护 → 多线程同时进入 `generate()`

**位置**：`api/deps.py:199-223`（`get_runtime_pipeline`）、`src/llm.py:306`（`_generate`）

**根因**：`get_runtime_pipeline()` 用 `copy.copy(base)` 为每个请求生成租户 pipeline，
但 `llm` 是**共享的同一个对象**。同步端点 `chat_sync` 由 FastAPI 调度到线程池，
WS 端点受 `AsyncLLMExecutor.semaphore`（默认 3）门控 —— 两条路径都允许多线程
并发调用同一个 `model.generate()`。HF Transformers 的 `generate()` 在共享
`nn.Module` 上并非线程安全，并发会导致生成内容错乱、CUDA 错误或显存 OOM。

**影响**：多用户同时提问时回答内容可能串扰，或整进程崩溃。单用户串行使用时不可见。

**修复建议**：在 `LocalLLM` 内部加一把生成互斥锁（比在 API 层加更可靠，能覆盖所有调用方）：

```python
# src/llm.py __init__
self._gen_lock = threading.Lock()

# _generate / _stream_generate 内包裹 model.generate(...)
with self._gen_lock:
    output_ids = self.model.generate(**inputs, ...)
```

注意：锁会降低吞吐（生成是串行化的），但这是本地单卡部署的正确取舍；
若需并发，应改为批处理（batching）或部署多个 worker 进程各自持有一份模型。

---

### H-4 WebSocket 流式问答无取消机制 → 线程与 GPU 泄漏

**位置**：`src/rag_pipeline.py:522-564`（`astream_answer`）

**根因**：`astream_answer` 为每个请求起一个 daemon 线程生产事件，通过
`asyncio.Queue(maxsize=256)` 转交事件循环。客户端断开后消费端 `await queue.get()` 停止，
队列不再排空；生产者线程的 `asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()`
在队列满后**永久阻塞**。

**影响**：客户端断开不会中断仍在进行的 LLM 推理。泄漏的线程持续占用 GPU 显存，
并与后续请求**并发调用同一个模型**（叠加 H-3 的线程安全问题）。反复断连会累积泄漏，
最终耗尽显存或拖垮服务。

**修复建议**：

1. 给队列 put 加超时，并在 producer 里检查停止信号：

```python
_STOP = object()

def _produce():
    try:
        for event in self.stream_answer(question, top_k=top_k):
            fut = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            try:
                fut.result(timeout=5.0)      # ← 避免永久阻塞
            except Exception:
                return                        # 消费端已消失，放弃生产
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop)
```

2. 更根本的方案：用 `asyncio.to_thread` + `asyncio.CancelledError` 传播，
   或在 `LocalLLM._stream_generate` 中支持 `stopping_criteria`，让生成器关闭时真正中止推理。
3. WS 处理器应捕获 `WebSocketDisconnect` 并显式关闭 async generator
   （`await gen.aclose()`），Python 3.8+ 支持。

---

### H-5 `RecursiveTextSplitter` 与超长句切片的边界缺陷

**位置**：`src/text_splitter.py:156-159`（超长句硬切）、`219-290`（`RecursiveTextSplitter`）

问题点：

- 超长单句按 `range(0, len(sent), chunk_size)` 硬切，**不产生重叠**，
  切断处的语义连续性完全丢失（这是唯一真正需要重叠的场景，反而没有）。
- `RecursiveTextSplitter.__init__` 无任何参数校验：`chunk_size<=0` 时
  `range(0, len(text), 0)` 抛 `ValueError: range() arg 3 must not be zero`，报错信息对用户无意义。
- 该类不实现 `min_chunk_size`，会产生大量零碎块，与 `ChineseTextSplitter` 行为不一致。

**建议**：
- 超长句切片时保留 `chunk_overlap`；
- 两个 splitter 共用一份参数校验（可抽 `@dataclass` + `__post_init__`）；
- 让 `RecursiveTextSplitter` 支持 `min_chunk_size`，或在工厂里明确文档化差异。

---

## 三、中严重度问题（Medium）

### M-1 context 预算按「1 token ≈ 2 字符」估算，实际低估约 2 倍
**位置**：`src/rag_pipeline.py:305-306`
`max_ctx * 2` 把 `max_context_tokens=2000` 换算成 4000 字符。但 Qwen2.5 分词下
中文约 1 字符 ≈ 0.6~1 token，4000 字符实际约 2700~4000 token，超出声明预算约 2 倍。
**影响**：实际 prompt 长度失控，生成变慢、显存压力上升，且配置失去约束意义。
**建议**：用真实 tokenizer 做预算控制，而非字符数估算：

```python
budget_tokens = rag_cfg.get("max_context_tokens", 2048)
# 用 self.llm.tokenizer 逐块累计真实 token 数，超预算即停止拼接
```

---

### M-2 流式路径未使用 `rerank_candidates`，开启重排后召回质量低于非流式
**位置**：`src/rag_pipeline.py:492-495`
`answer()` 用 `initial_k = max(k, rerank_candidates)`（配置 12）召回再重排到 k；
而 `stream_answer()` / `astream_answer()` 只召回 `k` 条再重排到 `k`，**重排等于空转**。
当前 `reranker.enabled: false`，问题尚未暴露；一旦开启，流式（WS、前端主路径）质量会明显低于同步接口。
**建议**：抽取统一的 `_prepare_context(question, top_k)`，三条路径复用，`stream_answer` 一并支持 `initial_k`。

---

### M-3 `score_threshold: 0.0` 导致无关查询也强制返回 Top-K
**位置**：`config/config.yaml` → `retrieval.score_threshold`
bge 归一化向量上，语义无关的中文文本对余弦相似度通常仍在 0.4~0.7。阈值为 0 意味着
**任何提问都会塞满 4 个不相干片段**，模型倾向基于噪声编造答案，而非按 system prompt 要求回答「无法回答」。
**建议**：设为 0.35~0.45（需用现有评测集 `scripts/evaluate.py` 标定），
或改用相对阈值（如只保留 score ≥ top1_score × 0.8 的片段）。

---

### M-4 分块器构造逻辑重复 4 份
**位置**：`api/deps.py:236`、`scripts/ingest.py:42`、`scripts/build_kg.py:35`、`ui/app.py:142`
四处几乎逐行相同的 `ChineseTextSplitter(...)` / `RecursiveTextSplitter(...)` 工厂代码。
**影响**：改分块参数要同步改 4 个地方，极易漂移（例如 `build_kg.py` 已与其他三处不同步）。
**建议**：统一到 `src/text_splitter.py` 暴露 `build_splitter(cfg)`，四处改为调用。

---

### M-5 知识图谱未接入主检索流程
`src/rag_pipeline.py` 全文无任何 KG 引用，而 `config.yaml` 中 `knowledge_graph.graph_rag.enabled: true`。
GraphRAG 只在 `api/routes/kg.py` 与 `ui/page_modules/kg.py` 作为独立入口存在。
**影响**：配置宣称的能力未生效；主问答链路无法利用图谱的多跳关联。
**建议**：要么在 `RAGPipeline._retrieve` 中做向量 + 图谱的混合召回，要么把配置项改为 `false` 并补充说明，避免误导。

---

### M-6 `AsyncLLMExecutor`（277 行）仅被当作信号量使用
**位置**：`src/executor/llm_executor.py`，使用点 `api/routes/chat.py:493`
`start()` 从未被调用，令牌桶限流、优先级队列、队列容量控制全部空转；
代码里只取了 `executor.semaphore`。此外该模块调用的 `llm.agenerate` / `llm.generate` /
`llm.stream_generate` 在 `LocalLLM` 上**均不存在**（实际接口是 `chat()`），即便启用也会 AttributeError。
**建议**：二选一 —— 删除该模块，改为在 `api/deps.py` 直接暴露一个 `asyncio.Semaphore`；
或补全 `LocalLLM` 的 `agenerate` 接口并真正启动执行器。当前状态属于“看起来有限流、实际没有”。

---

### M-7 流式生成无超时保护；`_generate` 无截断
**位置**：`src/llm.py:309`（`_generate`）、`325-368`（`_stream_generate`）
- 非流式走 `_generate_with_timeout`，流式**完全没有超时**，且 `th.join()` 无 timeout，
  异常情况下会永久挂起（配合 H-4 会放大泄漏）。
- `self.tokenizer(prompt, return_tensors="pt")` 未设 `truncation=True`，
  prompt 超过模型位置上限时直接抛错，而非优雅截断。
**建议**：`truncation=True, max_length=<model_max>`；流式路径加 watchdog 或 `streamer` 超时。

---

### M-8 块长可超出 `chunk_size`；重叠按字符硬切且逐块累积
**位置**：`src/text_splitter.py:175`（`max_len = chunk_size + chunk_overlap`）、`190-208`
- 合并短块与加重叠的上限都是 `chunk_size + chunk_overlap`（默认 350），实测块长 336 > 300。
- 重叠取的是**上一块（已含重叠）的尾部字符**，导致重叠沿链条逐块累积，第 N 块可能残留第 1 块内容。
- 按字符切重叠会切断词语/句子，语义碎片化。
**建议**：重叠改为按句子回退（从上一块末尾取完整句子直到满足 overlap 长度），
并让最终块长严格 ≤ `chunk_size`。

---

### M-9 测试全绿但存在覆盖盲区
`256 passed`，而 H-1 的标题丢失场景在 `tests/test_text_splitter.py`（8 个用例）中**无覆盖** ——
现有用例的文本每行都以「。」结尾，恰好绕过了 bug（`\s*` 吸收换行）。
**建议**：补充回归用例：

```python
def test_heading_lines_are_preserved():
    text = "第三章 系统设计与实现\n检索增强生成提升了准确率。\n3.1 模块划分\n其余内容。"
    out = "".join(c.text for c in ChineseTextSplitter().split_text(text))
    assert "第三章 系统设计与实现" in out
    assert "3.1 模块划分" in out

def test_no_text_loss():
    import re
    norm = lambda s: re.sub(r"\s+", "", s)
    assert norm("".join(c.text for c in sp.split_text(doc, chunk_overlap=0))) == norm(doc)
```

---

### M-10 入库「先删后插」非原子；配额校验顺序错误
**位置**：`api/routes/chat.py:352-360`、`scripts/ingest.py:146-154`
`_delete_stale_chunks` 与 `add_chunks` 之间无事务保护：并发入库或期间查询，
会让知识库**短暂为空**，检索到一半数据。
另：配额校验 `check_chunk_quota(user, existing, len(all_chunks))` 在删除旧块**之前**执行，
重新上传一份大文档时，会把即将被删除的旧块也算进占用，**误判为超配额**。
**建议**：先删再校验再写入；或改为「先写临时 collection，再原子切换别名」，
Chroma 无事务能力，最低成本是先加一把 per-source 的入库锁。

---

### M-11 TXT 硬编码 UTF-8；PDF 解析失败静默继续
**位置**：`src/document_loader.py:214`、`96-98`
- `errors="ignore"` 会把 GBK/GB2312 编码的中文 TXT 读成乱码，且**无任何日志**。
  这是一个中文知识库项目，编码问题高发。
- `_load_pdf` 捕获异常后仅 log，返回已解析的部分页面，调用方无法区分「文档为空」与「解析失败」。
**建议**：按 `utf-8 → gbk → gb18030 → big5` 依次尝试并在失败时告警；
`load_document` 返回解析状态，入库时统计并上报失败文件数。

---

## 四、低严重度问题（Low）

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| L-1 | 多处（372 条警告） | `datetime.utcnow()` 已弃用，Python 3.12+ | 改为 `datetime.now(timezone.utc)` |
| L-2 | `api/deps.py:168` | `bool(os.getenv("RAG_LAZY_LLM","1"))` —— `RAG_LAZY_LLM=0` 仍为惰性（非空字符串恒为真） | 改用 `os.getenv(...).lower() in ("1","true","yes")` |
| L-3 | `scripts/ingest.py:82` | `--recursive` 参数解析后未使用，硬编码 `recursive=True` | 传入 `args.recursive` |
| L-4 | `api/deps.py:291-323` vs `464` | 锁序反转隐患：`get_kg_extractor` 持 `_kg_lock` 取 `_singletons_lock`，`reset_all_singletons` 反向 | 统一加锁顺序 |
| L-5 | `src/vector_store.py:95` | `collection_name` 无 3-63 长度校验，非法名直接抛 chromadb 底层 ValueError | 构造时校验并给出可读错误 |
| L-6 | `src/factories.py:44-48` | `_chroma_stores` / `_kg_stores` 无上限，租户增长即内存泄漏 | 引入 LRU 或空闲超时回收 |
| L-7 | `src/utils.py:181-195` | `_ensure_request_id_filter` 在根 logger 无 handler 时也会置标志位，之后新增的 handler 拿不到 filter | 置位前先判断 handler 是否存在 |
| L-8 | `src/vector_store.py:331-337` | `reset()` 硬编码 `{"hnsw:space":"cosine"}`，忽略 `self.distance_fn` | 改用 `self.distance_fn` |
| L-9 | `src/vector_store.py:159-164` | 外部传入 `embeddings` 长度与 `chunks` 不一致时，`aligned[i]` 抛无提示 KeyError | 前置长度校验 |
| L-10 | `config/config.yaml` | `vector_store.fetch_k`、`hybrid`、`rag.max_history_turns` 等配置项代码中无实现 | 删除或标注“未实现” |

---

## 五、性能专项

| 项 | 现状 | 建议 |
|---|---|---|
| `count()` 缓存 | 收益小、一致性风险大（H-2） | 直接移除 |
| `list_sources()` | `collection.get(include=["metadatas"])` 全量拉取元数据，O(N) 内存 | 维护一张 source→chunk_count 的索引表，或在 UI 层分页 |
| 每次查询的 embedding 调用 | 单条 query 也走 `query_batch`，逻辑统一，无浪费 | 保持 |
| Prompt 预算 | 字符估算偏差 2 倍，实际 prompt 偏长（M-1） | 改用 tokenizer 精确计量，可直接降低生成延迟 |
| 流式队列 | 每 token 一次 `run_coroutine_threadsafe().result()`，跨线程往返开销大 | 改为批量搬运或 `asyncio.to_thread` + 队列批处理 |
| 模型并发 | 无锁，并发即风险（H-3） | 加锁后吞吐下降属预期，若需吞吐应改为批处理 |

---

## 六、建议的修复顺序

1. **H-1**（分句丢失标题）+ 补 M-9 回归测试 → **需重新入库**，否则已有索引仍缺标题
2. **H-2**（count 缓存）→ 改动最小、收益最大
3. **H-3**（LLM 并发锁）+ **H-4**（WS 取消）
4. **M-4**（合并 4 份 splitter）→ 为后续分块调优扫清障碍
5. **M-1 / M-2 / M-3** → 检索质量调优，配合 `scripts/evaluate.py` 用数据验证收益
6. 其余中低项按迭代排期

> 注意 H-1 修复后，`_make_id(source, page, chunk_index)` 的分块编号会变化，
> 需全量重建索引（先 `reset()` 再入库），不能只做增量更新。
