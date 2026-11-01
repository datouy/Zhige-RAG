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

## 部署边界

当前 K8s 清单固定为单副本，因为 Chroma 与用户知识图谱使用本地文件库，
多个 Pod 并发写同一 PVC 会损坏索引。需要横向扩容时，应先迁移到集中式向量
数据库，并为知识图谱、限流和任务状态提供共享存储。

本项目默认无登录，Ingress 只提供集群内部域名。接入公网前必须增加认证代理
和多用户隔离。

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

服务无需登录，所有数据归属固定本地用户 `00000000-0000-0000-0000-000000000001`。

数据位置：

- 向量库：`data/chroma_db/`
- 关系库：`data/users.db`（SQLite，生产可切换 PostgreSQL）
- 原始文档：`data/raw/`
- 知识图谱：`data/kg.db`

因为默认无认证，**不要直接把服务暴露到公网**；需要公网访问时，请在前面加一层
自己的认证网关或反向代理。生产部署参考 `docker-compose.yml` 与 `k8s/`。

