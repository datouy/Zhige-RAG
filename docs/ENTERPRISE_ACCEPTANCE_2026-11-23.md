# 中型企业可用性改造验收记录

日期：2026-11-23

## 结论

六项指定问题均已形成代码、配置、测试或运维产物。当前项目具备企业内部试点所需的
认证、角色、对象级授权、凭据保护、集中式多副本后端、完整备份恢复和硬质量门禁。

仍需在实际上线前完成组织级事项：接入真实 IdP（如企业已部署）、使用托管高可用
PostgreSQL/Redis/Neo4j、在预生产完成备份恢复演练，并按真实语料重新标定 SLA。

## 逐项结果

| 问题 | 当前结果 | 主要证据 |
|---|---|---|
| 固定本地管理员、无登录与停用 | 已修复：enterprise 默认真实登录；scrypt 密码；失败锁定；服务端可撤销会话；账号停用即失效；角色职责分离；local 零登录只允许回环地址 | src/security.py、src/middleware/identity.py、api/routes/auth.py、tests/test_enterprise_auth.py |
| 知识库只有单用户隔离 | 已修复：owner + admin + member + auditor；知识库 private/department/company；显式成员 read/write；文档与检索 ACL；任务、会话和对象 ID 越权返回 404/403 | src/kb_service.py、api/routes/kb.py、api/routes/chat.py、tests/test_kb_object_acl.py |
| API Key 可能写入 YAML | 已修复：环境变量/Secret 优先；可选 Fernet 加密文件；覆盖 YAML 禁写 api_key；旧明文自动清除和迁移；设置接口不再回显密钥 | src/secrets_store.py、src/settings_service.py、tests/test_secret_store.py |
| Chroma/KG 本地单写导致只能单副本 | 已修复：统一向量接口 + pgvector；Neo4j tenant_id 隔离；Redis 跨 Pod 限流；企业 K8s 3～6 副本、HPA、PDB、Cookie 亲和 Ingress | src/vector_store_pgvector.py、src/factories.py、src/kg/store.py、k8s/enterprise/ |
| 备份覆盖不完整 | 已修复：PostgreSQL/SQLite、pgvector、Chroma、原始文档、SQLite/Neo4j 图谱、脱敏配置；manifest + SHA256；恢复前 SQLite 快照；恢复演练测试 | scripts/backup_restore.py、tests/test_backup_restore.py、docs/OPERATIONS.md |
| CI 质量门禁只告警、压测不足 | 已修复：固定 8 文档/12 问题语料，真实 ONNX + BM25/RRF 硬门禁；真实 Uvicorn HTTP 并发压测脚本；限流阈值可通过 RATE_LIMIT_QUERY 配置 | scripts/ci_eval.py、tests/fixtures/、scripts/load_test.py、.github/workflows/ci.yml |

## 验证证据

| 检查 | 结果 |
|---|---|
| 全量测试 | 501 passed |
| 静态检查 | ruff 对 src、api 及新增脚本全部通过 |
| JS 语法 | node --check 通过 |
| Alembic 全链路 | 从空库升级到 007_kb_object_acl 成功，含 auth 与 KB 成员表 |
| 固定语料质量门禁 | 12 样本；hit rate 1.000；NDCG@5 0.965；P95 222.16 ms |
| 真实 HTTP 搜索压测 | 100 请求 / 10 并发；成功 100；P95 83.94 ms；约 141 req/s（本机测试） |

压测为本机、单 API 进程和固定小语料结果，不代表生产容量。生产容量必须在目标硬件、
真实语料和真实模型后端上重新测试。

## 上线前仍需执行

1. 将 k8s/enterprise 中的 PostgreSQL、Redis、Neo4j 替换为企业托管或已经验证的高可用方案。
2. 若企业已有 OIDC/LDAP/AD，在现有本地账号认证前增加 IdP 对接与部门同步。
3. 在预生产执行完整备份、删除测试库、恢复、启动、检索和权限回归的整套演练。
4. 使用真实语料建立不少于 100 条的评估集，重新确定 hit rate、NDCG、引用正确率、拒答率与 P95。
5. 异步入库任务当前仍为进程内队列，依赖 K8s Ingress Cookie 亲和；若要去掉粘滞，迁移到 Redis 队列。
