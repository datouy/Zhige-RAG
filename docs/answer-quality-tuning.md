# 回答质量调优记录（从"碎片清单"到"分层论证"）

> 现象：问"总结一下这一章"，系统给出「1. 以…为叙事背景 [1]；2. 揭示… [7][8]；3. 点明… [2]」
> 这样的内容目录，末尾再加一句"知识库中未提供其余部分，无法进一步确认"。
> 而同一个模型直连读全文时，给出的是一句总判断 + 三层递进论证 + 收束。
> 两者是同一个 `deepseek-v4.1-flash`，差距全部来自 RAG 链路本身。

---

## 一、根因（按影响从大到小）

| # | 根因 | 位置 | 后果 |
|---|---|---|---|
| 1 | **`_retrieve` 把全库返回给上下文装配** | `rag_pipeline._retrieve` | `top_k = max(top_k, corpus_size)` 让 253 块全部进入候选；`_build_context` 均分 3600 字预算 → **每块 14 个字**，全是残句，引用编号跳号（这就是回答里 `[1]-[5][7][8]` 缺 `[6]` 的来源） |
| 2 | **只有 `answer()` 做全文扩展** | `rag_pipeline.answer` | Web UI 与异步接口走的是 `stream_answer`，那条路径**完全没有**全文扩展 → 界面上问"总结这一章"永远只拿到 Top-K 碎片 |
| 3 | **上下文预算 1800 token** | `config.rag.max_context_tokens` | ≈ 3600 中文字符，一整章都装不满；且旧 `_build_context` 是"先到先得"扣预算，`budget<=0` 就 `break`，长文档永远只剩开头两段 |
| 4 | **Prompt 命令"照抄原文"** | `config/prompts.yaml` | 「优先直接采用上下文中的原文表述」「必须逐条照抄」「宁可拒答」「保持简洁、编号列表」—— 这套约束的唯一安全解就是"摘原句 + 挂编号 + 声明不确定" |
| 5 | **校验器把"提炼"判成"编造"** | `src/verifier.py` | `min_cited_ratio=0.5` 要求过半句子带引用；`min_grounding=0.6` 要求句子词元 60% 出现在所引分块里。**判断句的词元天然不在原文里**，于是任何做了综合的回答都被判 `unverified` |

第 1 条最隐蔽，也最致命：它让"扩了全文"也白扩 —— 内容是进来了，但每块只剩十几个字。

---

## 二、改动清单

### 配置

- `config/config.yaml`
  - `rag.max_context_tokens`: `1800` → `12000`
  - `rag.verification`: `min_cited_ratio 0.5→0.2`、`min_grounding 0.6→0.3`、新增 `synthesis_aware: true`
  - 新增 `rag.fulltext_expand`（`enabled / pattern / max_documents / max_chars`）
  - 新增 `rag.mapreduce`（`soft_limit_chars / group_chars / max_groups / detail_chunks`）
- `config/prompts.yaml`：重写 `system` 与 `user_template`
  - 材料是**依据**不是抄写对象；读懂后重述是本职工作
  - 答案结构固定为「总判断 → 2~4 层递进 → 收束」，禁止「以…为背景」「本文提到」这类目录句式
  - 引用只标关键数据与关键论断，**不必逐句**；概括与因果分析不需要引用
  - 区分三种声音（作者判断 / 文中人物 / 引述第三方），避免把文中人物的话当成作者结论
  - 免责声明只在真无法回答时用，不作为结尾装饰
- `src/config/validator.py`：`max_context_tokens` 上限 `16000→64000`，新增 `fulltext_expand` / `mapreduce` 字段

### 代码

- `src/rag_pipeline.py`
  - **`_retrieve`**：全库打分后按分数**截回调用方的 `top_k`**（候选集 ≠ 上下文）
  - **`_expand_to_fulltext`**：从 `answer()` 里抽出来，`answer / answer_with_timing / stream_answer` 三条路径统一调用
  - **`_build_context`**：
    - 均分份额 + 余量回收（短块用不完的额度补给被截断的长块），保证每块都有代表
    - 单文档多分块时标注「第 i/n 段」，剥掉每块重复的 `【章节路径】` 面包屑与长标题（46 块省下约 3000 字噪音）
    - 有 `chunk_index` 时按原文顺序装配
  - **`_maybe_compress_context` + `_compress_group`**：超长上下文走 map-reduce（分段提要点 → 合成全文摘要块 + 最相关细节），适配"一次吃不下整章"的推理服务
- `src/vector_store.py`：入库时把 `chunk_index` 写进 metadata；`get_by_source` 按它升序返回
- `src/verifier.py`：新增 `synthesis_aware`（默认 `False`，零回归）
  - 提炼/判断句豁免词面接地检查（带硬事实的句子**不豁免**）
  - 引用覆盖率只按"含硬事实的句子"计算；`cited_sentence_ratio` 字段仍是纯统计值

### 脚本

- `scripts/backfill_chunk_index.py`：给已有索引回填 `chunk_index`（只改 metadata，不重算向量）
- `scripts/diagnose_answer_quality.py`：对比"修复前 / 只加扩展 / 修复后"三种配置的上下文装配，并可选跑真实回答

---

## 三、验证

```bash
# 1) 看上下文到底多了多少（不调模型，几秒）
python scripts/diagnose_answer_quality.py "四万亿计划出台这一章讲了什么"

# 2) 连真实模型看回答形态
python scripts/diagnose_answer_quality.py "RAG 的全称是什么？它通过什么方式缓解大模型的幻觉问题？" --ask

# 3) 老索引补原文顺序（只写一次）
python scripts/backfill_chunk_index.py --dry-run
python scripts/backfill_chunk_index.py --write
```

第 18 章（46 块）实测：

| 配置 | 进 prompt |
|---|---|
| 修复前 1800tok / 无扩展 | 8 块 / **792 字** |
| 只加全文扩展（预算不变） | 46 块 / 3438 字，**37 块被截断** |
| 修复后 12000tok + 扩展 | 46 块 / **5551 字，0 块截断** |

`chunk_index` 回填后，46 块按原文顺序进入上下文（标题 → 银杏叶 → 马骏的报告 → 发电量 -8.9% → 商品房销售 -32.5% → 温家宝批注 → 张平"3万亿不够" → …），因果链完整。

回答形态实测（`scripts/diagnose_answer_quality.py ... --ask`，走 Web UI 同一条 `stream_answer` 路径）：

**短文、不压缩**（`rag-intro.md`，2814 字上下文 / 8.5 秒）：

> **总判断**：这批材料是一篇本科毕业论文的绪论与研究方案部分，核心主张是：通用大模型存在"幻觉、知识过时、不懂垂直领域"三类明显缺陷，RAG 是对这三类缺陷的工程化解决方案…
>
> **第一层：RAG 全称与出处**
> RAG 全称 **Retrieval-Augmented Generation**，中文为"检索增强生成"，由 Lewis 等人于 2020 年首次提出 [3]…
>
> **第二层：缓解幻觉的机制在"先检索、后生成"的顺序**
> 材料先明确了幻觉的性质：模型在缺乏相关知识时，会"编造"看似合理但与事实不符的回答 [2]…

`verification: verified, cited_ratio=0.273, unsupported_facts=0, ungrounded=0`；列表行 0；无免责收尾。

**长文、走 map-reduce**（毕业论文，93 块 / 23965 字 → 压缩后约 1500 字）：

> 这篇毕业论文的核心不是提出新的模型结构，而是论证一件事：RAG 不必绑定云端 API，在消费级显卡上也能跑出一套可用的中文知识库系统，并且用实测指标把"可用"量化出来。
>
> **一、选题的动因：云端 RAG 的四个问题倒逼本地化**
> 作者对主流 RAG 的判断是，它已经是缓解 LLM 幻觉、知识时效性差、领域知识缺乏的主流方案，但依赖云端 API 带来隐私风险、服务依赖、长期成本高、中文支持不足四类问题 [1]…
>
> **二、系统机制：用小模型 + 重排 + 混合检索补能力**
> 系统的技术路线可以看成一条"能力补偿链"…

`verification: verified, cited_ratio=0.286`。

---

## 四、环境限制（不是 RAG 逻辑能修的）

当前 `.env` 指向自建中转服务 `http://106.55.225.14/v1`（模型 `deepseek-v4.1-flash`）。
上两节证明它**能吃下 3600 字左右的上下文并正常流式输出**，但有一类内容它不会：

- 同一 prompt、同一长度，换成 `rag-intro.md`、毕业论文 → 正常输出
- 换成第 18 章（涉及国务院决策、"绝密"报告、政治人物）→ **返回空字符串**，
  `stream=False` 也不抛异常；加 `max_new_tokens=200` 限制输出长度 → 仍然空（输入侧判定）
- 表现为：`stream_answer` 只发出 `hits` 和 `done`，`token` 事件数 0，`verification.status = refused`

**处置**：换直连 `https://api.deepseek.com/v1`（改 `OPENAI_BASE_URL` / `OPENAI_MODEL` 两个环境变量），
或改用本地权重（`llm.backend: local`）。这不是 prompt 措辞能解决的 —— 已试过删掉
"分清三种声音"等条款、把 system 压到 74 字，均无改善。

> 调试提示：`scripts/diagnose_answer_quality.py` 的前三节会临时替换 `pipeline.config`
> 去做配置对比。若在同一个 pipeline 实例上接着跑生成，可能读到残留状态、表现为
> "回答为空"。脚本已在 `--ask` 分支重建 pipeline；自己写脚本时也建议照做。

---

## 五、可调参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `rag.max_context_tokens` | `12000` | 送进 prompt 的 token 上限；服务窗口小就调低，`mapreduce` 会接管超长部分 |
| `rag.fulltext_expand.max_documents` | `1` | 总结类问题扩几篇全文。问"对比两章"时调到 2 |
| `rag.fulltext_expand.max_chars` | `40000` | 扩展的字符护栏，防止误触发时把整库倒进 prompt |
| `rag.mapreduce.soft_limit_chars` | `12000` | 超过就分段压缩；服务能吃更长就调大 |
| `rag.mapreduce.detail_chunks` | `0` | 摘要之外附多少块原始细节；`0` = 只给摘要 |
| `rag.verification.synthesis_aware` | `true` | `false` 恢复旧行为（提炼句也会被判编造） |
| `retrieval.top_k` | `8` | 候选分块数；检索不准时调大，但会被 `max_context_tokens` 截 |
