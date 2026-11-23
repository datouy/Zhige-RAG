# 变更日志

## 0.4.0（企业可用基线，未发布）

- 恢复企业认证：scrypt 密码、可撤销服务端会话、失败锁定、账号停用。
- 新增 enterprise / local 两种身份模式；local 只允许回环地址。
- 新增 admin / kb_admin / member / auditor 职责分离。
- 知识库支持私有、部门、全员共享和显式 read/write 成员授权。
- 检索、文档、分块、会话和异步入库任务统一对象级授权。
- 模型凭据改为环境变量/Secret 或 Fernet 加密文件，覆盖 YAML 禁止明文 Key。
- 向量层支持 pgvector，图谱支持 tenant_id 隔离 Neo4j，限流支持 Redis。
- 新增企业多副本 K8s 清单、HPA、PDB、Cookie 亲和 Ingress。
- 新增完整备份恢复工具、SHA256 校验和恢复演练测试。
- CI 增加固定语料 RAG 硬门禁；新增真实 HTTP 并发压测脚本。

## 0.3.0

- 去除登录 / 注册与多用户认证，改为免登录本地固定用户（历史版本）。
- 去除订阅 / 套餐与配额限制：无需支付，启动即可使用。
- 新增多知识库、会话管理、网页 URL 导入、扫描件 OCR。
- 新增多模型后端（local / ollama / openai），LLM 与 Embedding 独立配置。
- 新增无 LLM 本地模式（extractive + ONNX embedding），不装 torch 也能跑完整检索链路。
- 新增 d1 决策层（可选）。

## 0.2.0

- 入库分块质量检查，自定义词典扩展到 230+ 词。
- BM25 多粒度分词与混合检索优化。
- reranker 默认关闭、LLM temperature 调低以提升事实型问答精度。

## 0.1.0

- 初始版本：BM25 + 向量混合检索、多格式文档加载、知识图谱、ReAct Agent、版本管理与评估。
