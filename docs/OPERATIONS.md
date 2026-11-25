# 运维与恢复

## Chroma 健康检查

启动服务后，`/api/ready` 会对全部 collection 执行只读探测。文件存在但
collection 无法解码时返回 `503`，不会再报告“已就绪”。

停机维护时可在命令行检查：

```bash
python scripts/chroma_admin.py health
```

升级 `chromadb`、迁移机器或批量重建前，先停止 API 并备份整个持久化目录：

```bash
python scripts/chroma_admin.py backup --output data/backups/chroma.zip
```

只有单个 collection 损坏、其余 collection 正常时，可先导出健康数据，再删除
损坏 collection 并重新入库：

```bash
python scripts/chroma_admin.py export --collection <name> --output data/export.jsonl
python scripts/chroma_admin.py drop-collection --collection <broken-name> --confirm
```

整个目录无法打开时，可将其整体隔离并创建空目录，随后从 `data/raw` 重新入库：

```bash
python scripts/chroma_admin.py quarantine --confirm --create-empty
```

Chroma 依赖已固定到 `1.5.0`。升级前必须先备份并检查，不能直接替换版本。

## 工作区备份与恢复

管理员可在“管理后台 → 备份恢复”执行完整工作区备份。命令行或自动化也可以使用：

```bash
curl -b cookies.txt -X POST http://localhost:8000/api/v1/admin/backup
curl -b cookies.txt http://localhost:8000/api/v1/admin/backup/<filename> -o backup.zip
```

备份包包含：

- SQLite 一致性快照，或 PostgreSQL `pg_dump` 生成的自定义格式转储；
- `data/chroma_db/` 向量索引；
- `data/raw/` 原始资料；
- `data/eval/` 评估集与报告；
- `data/kg_*.db*` 知识图谱文件；
- `data/ingest_jobs.json` 入库任务状态；
- 审计日志和去掉秘密字段的 `config/local_overrides.yaml`。

备份不单独打包 `.env`、模型 API Key、Cookie 或浏览器数据。数据库中的账号
口令摘要会随工作区保留，以便恢复后继续使用原团队账号；因此备份包应按敏感
数据保管，并建议存放在加密磁盘或加密对象存储中。
本地配置覆盖也会移除模型密钥或仅含凭证的配置项。PostgreSQL 环境必须把
`pg_dump` 与 `pg_restore` 加入 `PATH`。

恢复流程：

1. 管理员在后台选择 ZIP，并输入大写 `RESTORE` 二次确认；
2. 服务检查清单、SHA-256、路径穿越、软链接、文件数量和解压后总大小；
3. 校验通过后只在 `data/.restore_staging/` 隔离暂存，当前工作区不变；
4. 重启知阁。恢复器会在数据库初始化前自动再生成一份当前数据安全备份，然后应用恢复包；
5. 升级或迁移完成后检查登录、知识库、检索、MCP 审批和入库任务。

恢复结果写入 `data/.restore_status.json`。成功时 `status=applied`，并记录
`safety_backup`；失败时 `status=failed`，待恢复包会保留，管理员可以修复环境后
重启重试，或在后台取消。恢复前安全备份位于 `data/.restore_safety/`。

恢复包尚未应用时，可通过以下方式取消：

```bash
curl -b cookies.txt -X DELETE http://localhost:8000/api/v1/admin/restore
```

## 部署边界

当前 K8s 清单固定为单副本，因为 Chroma 与用户知识图谱使用本地文件库，
多个 Pod 并发写同一 PVC 会损坏索引。需要横向扩容时，应先迁移到集中式向量
数据库，并为知识图谱、限流和任务状态提供共享存储。

默认 `ZHIGE_AUTH_MODE=local`，适合单机或个人工作区，不应直接公开到公网。
多人使用时切换为 `team`，完成管理员初始化并配置 `ZHIGE_BOOTSTRAP_TOKEN`、
`ZHIGE_COOKIE_SECURE=1`。公网部署仍建议在反向代理层配置 HTTPS、来源限制和
请求体大小限制，并只授予必要角色。

## 测试隔离

`pytest` 会在系统临时目录创建独立数据库、Chroma 目录和知识图谱目录，不再
读写 `data/users.db` 或 `data/chroma_db`。

## 评估阈值

评估脚本支持文件级和 evidence/chunk 级两种口径。带 `expected_evidence` 的
数据集使用 Recall@K、HitRate@K、MRR@K；旧数据集继续使用 `expected_sources`
文件级口径。标注格式和启动集见
`data/eval/templates/README.md` 与 `data/eval/templates/evidence_retrieval.jsonl`。

evidence 级 CI 门禁示例：

```bash
python scripts/evaluate.py --retrieval-only \
  --dataset data/eval/templates/evidence_retrieval.jsonl --top-k 10 \
  --min-samples 20 --min-answerable-samples 12 \
  --min-hard-samples 4 --min-refusal-samples 4 \
  --min-evidence-coverage 1 \
  --min-recall-at-5 0.8 --min-recall-at-10 0.9 \
  --min-hit-rate-at-10 0.9 --min-mrr-at-10 0.7 \
  --max-p95-latency-ms 1500 --no-archive
```

`--min-evidence-coverage 1` 会阻止过期标注集静默生成低分；引文无法映射时，
该 query 会进入 `retrieval_evidence.excluded_samples`，不会被当成模型漏召。
`expect_refusal: true` 的无答案题不进入检索命中率分母，生成式评估会单独输出
正确拒答率和应答题误拒率。数据集审计同时检查重复 ID/问题、缺失难度、无金标
可答题和错误标注的无答案题，并可用最少困难题/拒答题数量卡住发布门禁。
生成式评估还可增加 `--min-verified-rate 0.8`。报告仍会完整写出，未达到阈值时
进程以非零状态退出。

## 本地数据与部署边界

本地模式无需登录，所有数据归属固定用户
`00000000-0000-0000-0000-000000000001`。团队模式使用 `/api/v1/team/*`
登录入口，知识库仍按 owner 隔离，通过成员表授予只读或可编辑共享。

数据位置：

- 向量库：`data/chroma_db/`
- 关系库：`data/users.db`（SQLite，生产可切换 PostgreSQL）
- 原始文档：`data/raw/`
- 知识图谱：`data/kg.db`

单副本 K8s 清单固定沿用本地文件存储，工作区恢复也应按单实例停机窗口执行。
生产部署参考 `docker-compose.yml` 与 `k8s/`。

