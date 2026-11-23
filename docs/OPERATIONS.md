# 运维、认证与恢复

## 认证边界

- 默认 `AUTH_MODE=enterprise`，受保护 API 必须携带有效会话；
- `AUTH_MODE=local` 只允许回环地址，本地脚本已绑定 `127.0.0.1`；
- 首次企业启动通过 `BOOTSTRAP_ADMIN_USERNAME / BOOTSTRAP_ADMIN_PASSWORD` 创建管理员，
  不提供默认弱口令；
- 管理员停用账号或修改密码后，相关服务端会话立即撤销；
- 会话 Cookie 为 HttpOnly + SameSite=Strict；HTTPS 部署设置
  `AUTH_COOKIE_SECURE=1`。

## 企业多副本

原 `k8s/` 清单仍适用于 Chroma/SQLite 单写文件存储，必须保持单副本。
需要多副本时使用 `k8s/enterprise/stack.yaml`：

- 业务库与向量表进入 PostgreSQL/pgvector；
- 图谱进入带 tenant_id 隔离的 Neo4j；
- 限流计数进入 Redis；
- API 3～6 副本，PDB 至少保留 2 个可用实例；
- Ingress 配置 Cookie 亲和以保持异步入库任务粘滞。

任一部分仍使用本地文件存储时，不得提高原清单副本数或 HPA 上限。

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

## 部署边界

原 K8s 清单固定为单副本，因为 Chroma 与用户知识图谱使用本地文件库，
多个 Pod 并发写同一 PVC 会损坏索引。需要横向扩容时，应先迁移到集中式向量
数据库，并为知识图谱、限流和任务状态提供共享存储。

企业模式默认强制认证。反向代理仍需传递正确的 Origin/Host，并启用 HTTPS；
local 模式禁止暴露到非回环地址。

## 完整备份与恢复

优先使用完整工具：

```bash
python scripts/backup_restore.py backup --allow-live
python scripts/backup_restore.py restore <archive.tar.gz> --force \
  --database-url "<postgres-dsn>"
```

归档包含 manifest、SHA256、数据库、向量、原始文档、图谱和脱敏配置，不包含
`.env`、模型 Key 或 Vault Token。恢复 PostgreSQL、Neo4j 和共享文件卷时应在维护
窗口执行，并在预生产环境完成一次演练。恢复 SQLite 前会自动生成
`.pre-restore` 快照。

## 测试隔离

`pytest` 会在系统临时目录创建独立数据库、Chroma 目录和知识图谱目录，不再
读写 `data/users.db` 或 `data/chroma_db`。

## 评估阈值

评估脚本支持 CI 阈值，例如：

```bash
python scripts/evaluate.py --retrieval-only \
  --min-samples 30 --min-hit-rate 0.8 --min-ndcg 0.7 --max-p95-latency-ms 1500
```

生成式评估还可增加 `--min-verified-rate 0.8`。报告仍会完整写出，未达到阈值时
进程以非零状态退出。

## 本地数据与部署边界

local 模式使用固定本地用户 `00000000-0000-0000-0000-000000000001`；
enterprise 模式使用真实账号、部门和角色。

数据位置：

- 向量库：`data/chroma_db/`
- 关系库：`data/users.db`（SQLite，生产可切换 PostgreSQL）
- 原始文档：`data/raw/`
- 知识图谱：`data/kg.db`

local 模式**不要暴露到公网**。enterprise 模式也必须通过 TLS Ingress/反向代理访问，
并配置备份、监控和告警。生产部署参考 `docker-compose.yml`、`k8s/` 与
`k8s/enterprise/`。

