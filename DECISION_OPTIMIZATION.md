# d1 决策层优化说明（Decision Primitives）

> 本文说明「知阁」如何按照 **Liquid AI d1 决策模型** 的方法论做的一轮针对性优化：  
> 把"逐 token 生成"换成"单次前向决策"，在 **重排 / 意图路由 / Agent 工具调度 /  
> 答案校验 / 检索充分性 / 差评分流 / 入库质检** 七个环节引入决策原语。

---

## 1. 方法来源：d1 决策模型

传统生成式 LLM 不管问题多简单，都要走自回归解码，一个 token 一个 token 往外蹦，  
延迟动辄几百毫秒到数秒，还得靠 JSON Schema 严防格式崩塌。

Liquid AI 的 d1 换了个思&#x8DEF;**：不做生成，做决策。一次前向直接输出结论，  
RTX 4090 上单次决策约 8ms，`outpu`**`t_tokens` 恒为 0，从根本上消灭幻觉与格式问题。

它定义了三种基础原语：

| 原语       | 语义             | 输出                |
| -------- | -------------- | ----------------- |
| `noul`   | 是/否判定          | 0~1 的校准概率         |
| `choice` | 多选一分类          | 各选项概率分布 + 最高置信项   |
| `score`  | 有序梯度评分（2~10 级） | 期望分值 + 置信度 + 概率分布 |

文章点名的最佳落地场景里，有两处正是本项目的核心链路：  
**RAG 流程里的精准重排与质量打分**、**Agent 收到指令后判断该调哪个 Tool**。

---

## 2. 映射到知阁的七个环节

| 环节         | 文件                         | 改造前                                 | 改造后              | 原语               |
| ---------- | -------------------------- | ----------------------------------- | ---------------- | ---------------- |
| 检索重排       | `src/reranker.py`          | bge 交叉编码器（CPU 上慢，默认关）               | 可选 `D1Reranker`  | `score`          |
| 意图路由       | `src/query_router.py`      | 仅正则规则                               | 规则打底 + 决策叠加      | `choice`         |
| Agent 工具调度 | `src/agent/react_agent.py` | 靠 LLM 输出 `Action:` 文本再正则解析（小模型极易失败） | 决策预选 + 兜底        | `choice`         |
| 答案校验       | `src/verifier.py`          | 纯词面规则                               | 规则 + 语义接地 + 质量分级 | `noul` + `score` |
| 检索充分性      | `src/rag_pipeline.py`      | 只要检索到片段就一律生成（哪怕片段只是沾边）               | 明显不足则跳过生成、直接拒答    | `noul`           |
| 差评自动分流     | `src/feedback_router.py`   | `question_type` 靠前端手填，多数为空            | 决策自动归因到处理组       | `choice`         |
| 入库质检 / 合规  | `src/ingest_guard.py`      | 规则式准入门，只按 状态 / 权限 / 过期 过滤            | 语义质量分级 + 合规判定     | `score` + `noul` |

**为什么这七处最值得改**：它们都是"高频、需要确定性、不需要长篇输出"的判断点 ——  
恰好是生成式模型的短板、决策模型的长板。

其中后三处直接对应文章点名的落地场景：

- **检索充分性** —— 文章反复强调"毫秒级决策替代逐 token 生成"。生成前先问一句
  "这些片段够不够回答"，不够就跳过生成、直接拒答：**省下一次自回归解码，同时从
  源头掐掉"没检索到却硬编"的幻觉**。
- **差评自动分流** —— 对应"客服工单进来瞬间自动分流到具体业务组"。一条差评进库时
  自动归因到 `retrieval / answer / grounding / incomplete / format / other`，
  让"差评按类型归因"真正可用，bad case 自动流到该管的人手里。
- **入库质检 / 合规** —— 对应"内容合规检测、涉黄涉暴与毒性过滤" + "自动化质检打分"。
  规则式 `DataGate` 只能按状态 / 权限 / 过期过滤，抓不到"排版完好但通篇乱码"或
  "含违规内容"这类语义问题；本处用 `score` 给分块做质量分级、`noul` 做合规判定。

---

## 3. 架构

新增 `src/decision_provider.py`，把"决策"抽象成可替换的 provider，  
与项目既有的 `local / ollama / openai` 三后端哲学保持一致。

```
                 create_decision_provider(cfg)
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
  D1LocalProvider                        LLMDecisionProvider
  （原生 d1-3B）                          （用现有 LLM 模拟）
  transformers>=5.14 + torch             复用 llm_provider.create_llm
  单次前向、0 输出 token                  无需 torch / 无需新权重
  8ms 级（GPU）                          兼容本地 Qwen / Ollama / 云端 API
        │                                       │
        └───────────────────┬───────────────────┘
                            ▼
              decide(text, questions) → {名称: Result}
                            │
        ┌───────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
        ▼           ▼           ▼           ▼           ▼           ▼           ▼
   D1Reranker   意图路由   Agent 工具预选   校验增强    检索充分性    差评分流    入库质检
    (score)     (choice)     (choice)     (noul+score)  (noul)      (choice)  (score+noul)
        │           │           │           │           │           │           │
     reranker.py query_router react_agent  verifier.py rag_pipeline feedback_  ingest_
                                                          .py       router.py   guard.py
```

核心接口对齐 d1 官方的 `model.system_one(text, questions)`：  
一次调用、多个决策问题、返回结构化结果。基类 `BaseDecisionProvider`  
统一封装 `noul()` / `choice()` / `score()` 三个便捷方法，两种后端行为完全一致。

### 后端选择

| backend   | 需要什么                                                        | 适用                |
| --------- | ----------------------------------------------------------- | ----------------- |
| `llm`（默认） | 你现有的 LLM（本地 Qwen / Ollama / 云端）                             | **先跑通、零新增依赖**     |
| `d1`      | `requirements-d1.txt`（torch + transformers≥5.14 + 约 6GB 权重） | 有 GPU，追求极致速度      |
| `auto`    | —                                                           | 先试 d1，不可用自动回落 llm |
| `none`    | —                                                           | 关闭决策层             |

> `llm` 后端不是"单次前向"，而是用严格 JSON 约束把生成式模型"逼"成分类器。  
> 延迟与稳定性不如原生 d1，但**相比让模型自由输出再正则解析，格式可靠性显著更高**，  
> 且天然产出可用的概率 / 置信度。

---

## 4. 快速上手

### 路线 A：用你现有的本地 Qwen（推荐先跑这条）

`.env` 里把 `llm.backend` 保持你现在的配置（本地 Qwen 或 Ollama），然后：

```yaml
# config/config.yaml
decision:
  enabled: true
  backend: "llm"          # 用现有 LLM 模拟三种原语

  rerank:
    enabled: true
  query_route:
    enabled: true
  tool_select:
    enabled: true
  verify:
    enabled: true
  retrieval_sufficiency:
    enabled: true         # 检索明显不足 → 跳过生成直接拒答
  feedback_route:
    enabled: true         # 差评自动归因到处理组

reranker:
  enabled: true
  backend: "d1"           # 重排走决策层（score 原语）

data_quality:
  ingest_guard:
    enabled: true         # 入库语义质检 + 合规判定
    action: "flag"        # flag=只打标；drop=拦低质/可疑块
```

### 路线 B：原生 d1-3B（有 GPU 时）

```bash
pip install -r requirements-core.txt
pip install -r requirements-d1.txt
```

```yaml
decision:
  enabled: true
  backend: "d1"           # 原生单次前向
  model_name: "LiquidAI/d1-3B"
  device: "auto"
  local_files_only: false # 首次设 false 自动下载权重
```

### 单独启用某几个环节

四个开关彼此独立。例如只想改善 Agent 的工具调度：

```yaml
decision:
  enabled: true
  backend: "llm"
  tool_select:
    enabled: true        # 其余环节保持 false
```

---

## 5. 降级与安全设计

决策层是"锦上添花"，绝不能拖垮主链路。所有集成点都遵循同一套约束：

1. **默认全关** —— `decision.enabled: false` 时，各环节行为与改造前**逐字节一致**
   （`ingest_guard` 另有自己的 `enabled`，同样默认关）。已有 100+ 测试覆盖，零回归。
2. **绝不抛出** —— `BaseDecisionProvider._safe_decide` 吞掉所有异常并返回空；  
   单条决策失败时，重排沿用初检分、路由回落规则、校验沿用规则结果。
3. **保守改判** —— 决策路由要把 `fact` 改判为 `chitchat`（会跳过检索）时，  
   要求置信度 ≥ 0.75；一般改判只需 ≥ 0.6。**宁可多跑一次检索，不可吞掉真问题。**
4. **升级难、降级易** —— 校验融合时，d1 说"无依据"倾向于下调评级；  
   只有规则层无任何硬缺陷 + d1 明确有依据 + 质量高，才允许从 `partial` 升到 `verified`。
5. **成本护栏** —— 重排用 `max_candidates` 限制决策次数；决策输出用  
   `max_new_tokens` 限制长度。

---

## 6. 取舍与效果

**收益**

- 重排：无 GPU 的 CPU 机器上，决策打分通常快于 bge 交叉编码器，且可用  
  `reranker.backend: d1` 一键切换；
- 路由：正则的"一刀切"升级为语义判断，能识别正则覆盖不到的寒暄/聚合表达；
- Agent：小模型最易失败的"输出合法工具名"环节，由决策预选兜底，减少空转；
- 校验：补齐规则抓不到的"换个说法的无据表述"，`status` 更保守可信；
- 充分性：明显答不了的问题直接拒答，**省下一次完整生成**，也少了硬编的机会；
- 分流：差评自动带上归因标签，`/feedback/stats` 的"按类型归因"从摆设变成可用；
- 入库：语义级质量分级 + 合规判定，补上规则式 `DataGate` 抓不到的乱码 / 违规内容。

**代价**

- `llm` 后端每次决策都是一次 LLM 调用，会增加延迟；建议只开启真正需要的环节；
- 入库门禁对**每个分块**都要决策一次（`score` + `noul` 在同一次调用里问完），
  大文件建议配 `ingest_guard.max_chunks` 限制检查量；
- `d1` 后端需要额外权重与 torch（但它是**可选**依赖，不影响 core 安装）。

**验证**

```bash
pytest tests/test_decision_provider.py -v     # 37 个用例：原语归一化 + 四处集成
pytest tests/test_decision_extensions.py -v   # 21 个用例：分流 / 充分性 / 入库门禁
pytest tests/ -q                              # 全量回归
```

---

## 7. 授权提醒

d1 使用 Liquid AI 自家的 **LFM Open License v1.0**：  
**开放权重 ≠ 无条件免费商用**。

- 年营收 **< 1000 万美元**：个人 / 科研 / 中小团队可免费商用与衍生开发；
- 年营收 **≥ 1000 万美元**：需联系官方取得商业许可。

选型时既要看到 8ms 的性能红利，也要看清团队的合规边界。
