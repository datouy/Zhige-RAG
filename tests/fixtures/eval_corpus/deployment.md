# 多副本部署

多副本部署使用 PostgreSQL 作为业务数据库和 pgvector 向量库，使用 Neo4j 存储共享知识图谱，使用 Redis 保存跨实例限流计数。

API 默认运行三个副本，HPA 最多扩展到六个副本。异步入库任务通过 Ingress Cookie 亲和保持在同一副本。
