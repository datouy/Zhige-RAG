# Evidence 检索标注模板

`evidence_retrieval.jsonl` 是针对当前本地语料可直接运行的启动集。每行一个
query；可回答问题必须带至少一条 `relevance >= 2` 的引文，无答案题使用
`expect_refusal: true`。当前启动集不少于 20 条，生产稳定版本目标为 50 条以上。

## 字段

- `id`：数据集内唯一的问题 ID。
- `dataset_version`：标注版本，修改金标时递增。
- `question`：用户真实问法。
- `query_type`：例如 `proper_noun`、`numeric`、`multi_fact`、`section_lookup`。
- `difficulty`：`easy`、`medium`、`hard`。
- `expect_refusal`：无答案题设为 `true`。此类样本不进入检索命中率分母，单独
  统计正确拒答率；不能填写 `expected_sources` 或 `expected_evidence`。
- `expected_sources`：兼容旧文件级指标，也用于来源覆盖检查。
- `expected_evidence`：evidence/chunk 级金标。
- `expected_evidence[].source`：当前 collection metadata 中的来源文件名。
- `expected_evidence[].page`：可选；填写后只在同页 chunk 中映射。
- `expected_evidence[].quote`：必须是对应 chunk `text` 中连续出现的原文，不能取
  `parent_text` 中不可直接召回的段落。
- `expected_evidence[].relevance`：`3` 为直接支持答案，`2` 为相关证据，
  `<2` 为干扰项，不进入 Recall 分母。

评测时会先对引文和 chunk 做 NFKC、英文大小写折叠和空白折叠，再按
`source + page + quote` 映射到当前 chunk ID。重新切块后不需要手改数据库 ID。

## 运行

先确认所有引文都能映射到当前 collection。`coverage` 小于 1 时，未映射 query
不会按模型漏召计零分，而是进入 `excluded_samples`；CI 应同时设置
`--min-evidence-coverage 1` 防止过期标注集。

```powershell
python scripts/evaluate.py `
  --retrieval-only `
  --dataset data/eval/templates/evidence_retrieval.jsonl `
  --min-samples 20 `
  --min-answerable-samples 12 `
  --min-hard-samples 4 `
  --min-refusal-samples 4 `
  --min-evidence-coverage 1 `
  --min-recall-at-5 0.8 `
  --min-recall-at-10 0.9 `
  --min-hit-rate-at-10 0.9 `
  --min-mrr-at-10 0.7 `
  --max-p95-latency-ms 1500 `
  --no-archive
```

阈值是启动值。积累至少 20 条标注后，用连续两次未改动代码的基线运行取较差的
一次作为回归下限；业务目标可按实际 SLA 上调。
