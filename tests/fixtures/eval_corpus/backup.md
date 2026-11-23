# 备份恢复

企业备份必须包含关系数据库、向量数据、原始文档、知识图谱和必要配置。PostgreSQL 使用 pg_dump 备份，pgvector 表包含在同一数据库备份中。

恢复前必须校验归档清单和 SHA256。SQLite 恢复会先生成 pre-restore 文件。生产恢复必须经过演练，并明确 RPO 与 RTO。
