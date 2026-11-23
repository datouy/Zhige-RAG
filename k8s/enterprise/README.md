# 知阁企业多副本部署

这套清单用于中型企业内部的多副本部署，补齐原始单副本清单无法提供的横向扩容：

- PostgreSQL 同时承载业务表和 pgvector 向量表；
- Neo4j 承载共享知识图谱，所有节点和关系按 tenant_id 隔离；
- Redis 提供跨 Pod 限流计数；
- API 固定 3 副本，HPA 可在 3～6 之间伸缩；
- Ingress 使用 Cookie 亲和，避免异步入库任务轮询落到不同 Pod；
- 原始文档使用 ReadWriteMany 卷，应用本身不再写入本地 Chroma/KG 文件。

## 使用顺序

1. 先复制 secrets.example.yaml，替换全部占位符并创建 secrets.yaml。
2. 应用 stack.yaml。
3. 应用 secrets.yaml。
4. 等待 deploy/zhige-api 完成 rollout。

实际项目应在 Secret Manager、External Secrets 或 Sealed Secrets 中管理密钥，
不要把填好的 secrets.yaml 提交到 Git。

## 重要边界

- VECTOR_STORE_BACKEND=pgvector 与 KG_BACKEND=neo4j 是扩容前提；任一后端不可用时
  应让 Pod 不就绪，不能静默回退到每 Pod 本地文件。
- 异步入库任务仍是进程内存队列。Ingress 已配置 Cookie 亲和；若企业不使用该 Ingress，
  必须提供等价会话粘滞，或将任务队列迁移到 Redis/Celery 后再去掉粘滞。
- 首次启动必须提供 BOOTSTRAP_ADMIN_PASSWORD。创建管理员后建议从 Secret 中移除该值。
- HTTPS 由 Ingress 终止；AUTH_COOKIE_SECURE=1 要求外部入口必须使用 HTTPS。
