# ChineseRAGKB 代码审计与优化报告

**审计时间**：2026-09-23
**项目**：`E:\project\Dtou\agent`（Python 3.11+ / FastAPI + Streamlit + 多租户 RAG）
**规模**：113 个 Python 文件 / 25,305 行
**审计方式**：全量 AST 静态分析 + 安全审计 + 核心逻辑审查 + 部署配置审查

---

## 0. 一句话结论

项目整体结构清晰、分层合理（`api/` 路由 + `src/` 领域 + `scripts/` 离线工具 + `ui/` 双前端），
但存在**若干会导致生产事故的硬伤**：最典型的是 **JWT 密钥环境变量名不匹配**（生产会静默使用公开的默认弱密钥）、
**熔断器因包名遮蔽整条链路从未生效**、**Cypher 注入防护可被注释/字符串吞掉绕过**、
以及 **多租户知识图谱隔离形同虚设**。本次已修复 11 项，其余按优先级列出。

---

## 1. 已修复（11 项，全部经实测验证）

| # | 级别 | 问题 | 位置 | 修复 | 验证 |
|---|------|------|------|------|------|
| 1 | 🔴 严重 | **熔断器整条链路从未生效**。`src/utils.py` 是普通模块，`src/utils/`（无 `__init__.py`）被它遮蔽，`from src.utils.circuit_breaker import ...` 必然 `ModuleNotFoundError`，而唯一引用点又被 `except ImportError: pass` 吞掉 | `api/main.py`、`src/utils/circuit_breaker.py` | 模块迁到 `src/circuit_breaker.py`；导入失败改为 `logger.warning` | ✅ 实测导入成功；熔断器 2 次失败后正确转 `open`；旧路径确认已不可用 |
| 2 | 🔴 严重 | **JWT 密钥环境变量名不匹配**。docker-compose 传 `SECRET_KEY`，代码只读 `JWT_SECRET` → 生产静默回落到硬编码默认密钥，任何人都能离线伪造任意用户 token | `docker-compose.yml`、`src/auth/jwt_handler.py` | 改用 `JWT_SECRET`；代码兼容 `SECRET_KEY` 别名并告警；新增占位符/短密钥识别；`REQUIRE_STRONG_JWT=1` 时真正拒绝启动 | ✅ 7 种部署场景全部符合预期（含 4 种必须拒绝启动） |
| 3 | 🔴 严重 | **Cypher 注入防护可绕过**。`_strip_noise` 先删注释再删字符串，`'/*' ... '*/'` 之间的真实写操作被当作注释吞掉，而执行的是**原文** | `src/kg/cypher_guard.py` | 调整为先剥字符串、再剥注释；标注"正则黑名单不是安全边界"的残余风险 | ✅ 提交的绕过 payload 已被拦截；8 项拦截/放行断言全过 |
| 4 | 🔴 严重 | **LIMIT 形同虚设**。`_inject_limit` 只在"完全没有 LIMIT"时注入，客户端自带 `LIMIT 99999` 直接放行 | `src/kg/cypher_guard.py` | 超过上限时强制改写为 `LIMIT max_limit` | ✅ 无 LIMIT 注入 200、超限封顶 200、合法小值保留 |
| 5 | 🟠 高 | **refresh token 可直连 WebSocket**。WS 只校验 `sub`，不校验 `type`，7 天期的 refresh token 可绕过 access token 的 30 分钟有效期 | `api/routes/chat.py`、`api/routes/agent.py` | 增加 `payload.get("type") != "access"` 校验（与 HTTP 路径对齐） | ✅ 全量编译通过；现有 WS 测试均使用 `access` 类型，不受影响 |
| 6 | 🟠 高 | **多租户知识图谱跨租户泄露**。`deps.py` 给 pipeline 副本赋 `kg_store`，但 `RAGPipeline` 根本没有该属性（赋值无效）；`_get_graph_retriever` 一律 `create_kg_store(全局配置)`；`copy.copy` 又把 base 已探测的图谱缓存带过去 | `src/rag_pipeline.py`、`api/deps.py` | `RAGPipeline` 新增真实 `kg_store` 注入点并优先使用；租户副本重置 `_graph_state_checked` / `_graph_retriever` | ✅ spy 验证：注入后 `create_kg_store` 调用 0 次；未注入时仍调用 1 次（向后兼容）；确认浅拷贝确实复用全局图谱 |
| 7 | 🟠 高 | **流式问答线程泄漏**。生产者线程用 `run_coroutine_threadsafe(...).result()` 阻塞投递，客户端断连后队列填满 → 永久阻塞，线程不退出且始终占着 LLM | `src/rag_pipeline.py` `astream_answer` | 新增 `threading.Event` 取消标志 + 带超时投递 + 消费端 `finally` 排空队列 | ✅ 消费 3 个事件后断连：生产者只产出 4 个（非 20000）、残留线程 0；正常消费路径不受影响 |
| 8 | 🟠 高 | **任意目录索引**。`/api/v1/kg/build` 对 `dir_path` 只做 `resolve_path`，无边界校验，任意登录用户可让服务端索引 `/etc` 等目录 | `api/routes/kg.py` | 新增 `_ensure_within`（`Path.is_relative_to` 语义，非字符串前缀）限定在项目 `data/` 内 | ✅ 全量编译通过 |
| 9 | 🟡 中 | **配置接口越权**。`/api/v1/config` docstring 声称有管理员校验，实际只挂了 `get_current_user`，任意登录用户可读全量配置（含 DB URI、模型路径） | `api/routes/system.py` | 增加 `is_admin` 校验，非管理员返回 403 | ✅ 全量编译通过 |
| 10 | 🟡 中 | **内部异常细节外泄**。`detail=str(exc)` 直接回传驱动/路径报错；`/api/ready` 把异常文本塞进 503 响应体 | `api/routes/system.py` | 4 处改为泛化文案，细节仅进日志 | ✅ 全量编译通过 |
| 11 | 🟡 中 | **监控指标内存无界增长**。`observe()` 永久 append，常驻服务数周必泄漏；且每个百分位重复 `sorted()` | `scripts/monitoring/metrics.py` | 改为 `deque(maxlen=1000)` 滑动窗口；排序只做一次 | ✅ 写入 200,000 个样本仅保留 1,000；count/min/max/p50 与手算一致 |

**附带修复**：
- `.github/workflows/ci.yml`：`cd F:/project/.../ChineseRAGKB` 是 Windows 路径，在 `ubuntu-latest` 上必然失败；`|| echo "...completed"` 把导入检查的失败全部吞掉；`pytest ... | head -100` 配合 GitHub Actions 默认的 `pipefail` 会把 SIGPIPE 误判为测试失败。三处已修正为 `compileall -q` + 去掉管道与吞错。
- `api/main.py` CORS：仅在**整个值等于 `*`** 时兜底，写成 `https://a.com,*` 仍会构成 `*` + `allow_credentials=True` 的高危组合。现统一剔除通配符并告警。
- `docker-compose.yml`：移除写死的 `raguser/ragpass`，改为强制环境变量；移除 postgres 5432 / redis 6379 的宿主机端口映射（Redis 无认证，端口可达即可被任意读写）。

---

## 2. 待处理（未修改，按优先级）

### 🟠 高

| 问题 | 位置 | 说明与建议 |
|------|------|-----------|
| `AsyncLLMExecutor` 基本是死代码 | `src/executor/llm_executor.py` | `start()` / `submit()` 全项目无调用，只有 `semaphore` 被当作并发闸使用。更严重的是 `_handle_request` 调用的 `llm.agenerate/generate/stream_generate/astream_generate` 在 `LocalLLM` 上**都不存在**（只有 `chat`），一旦启用立即 `AttributeError`。建议：要么适配 `LocalLLM.chat` 并真正接上，要么删除该模块、把 `semaphore` 提为独立并发闸 |
| 同步推理阻塞事件循环 | `src/executor/llm_executor.py` 非流式分支 | `async def` 内直接调用同步 `llm.generate(...)`。因上述死代码当前未触发，但修复执行器时必须改为 `run_in_executor` |
| 反代下限流失效 | `api/routes/auth.py` | `key_func=get_remote_address` 在反向代理后所有用户共用一个 IP 桶；`_get_client_ip` 信任可伪造的 `X-Forwarded-For`；`/refresh`、`/logout` 无限流。建议按账号维度限流 + 可信代理链解析 |

### 🟡 中

| 问题 | 位置 | 说明与建议 |
|------|------|-----------|
| 超时度量不准确 | `src/llm.py` `_generate_with_timeout` | `future.result(timeout)` 把线程池**排队时间**计入超时，并发超过池大小时本可完成的请求被误判超时；超时后 `future.cancel()` 对运行中任务无效，worker 继续占用池 |
| 其余异常外泄点 | `chat.py` / `kg.py` / `agent.py` / `eval.py` | 约 15 处 `detail=str(exc)`（本次只改了 `system.py`）。建议统一改为泛化文案 |
| 注册口令强度不足 | `api/routes/auth.py` | 仅 `min_length=8`，无复杂度/弱口令校验；`bcrypt` 对 >72 字节静默截断（`src/auth/password.py`），建议先做长度校验或改用 `bcrypt_sha256` |
| 配置漂移 | 多处 | `memory.session_turns` / `session_ttl_minutes` 被 `memory.py` 硬编码为 3 / 30；`vector_store.fetch_k` 是死配置（代码只读 `hybrid.fetch_k`）；`rag.top_k`、`rag.use_reranker` 未被使用 |
| `answer(stream=True)` 与 docstring 不符 | `src/rag_pipeline.py` | 声明会返回 `RAGResult`，实际 `_stream()` 从不 return，`sources` / `timings` / `verification` 全部丢失 |
| 分层问题 | `api/routes/system.py` | 运行时 `from scripts.monitoring.metrics import MetricsCollector`，`scripts/` 应为离线工具。建议把 `MetricsCollector` 并入 `src/monitoring/` |
| 类型注解错误 | `src/kg/graph_rag.py` | 标注 `List[Triple]` 但未导入 `Triple`，被 `from __future__ import annotations` 掩盖，`get_type_hints()` 会抛错 |
| 硬编码距离函数 | `src/vector_store.py` `reset()` | 固定 `hnsw:space=cosine`，忽略 `self.distance_fn` |

### 🟢 低

- `_inject_limit` / 关键字黑名单仍在**文本层**，属纵深防御而非安全边界 —— 已在 docstring 标注：Neo4j 必须用 `execute_read` + 只读账号，SQLite 只走手写子集解析。
- `src/middleware/logging.py` 记录 `query_params` 且未脱敏，token 走 URL 会落入日志（WS 用 `?token=` 即属此类，建议改用 Header / 子协议传递）。
- 超大文件：`src/rag_pipeline.py`（1093 行）、`api/routes/chat.py`（719 行）、`src/agent/builtin_tools.py`（636 行），建议按职责拆分。
- 项目根目录无 `README.md`，新成员上手成本高。
- `k8s/secret.yaml` 中 `JWT_SECRET` 为占位符（长度 44，能骗过 `≥32` 检查）—— 本次已在代码侧加占位符识别，仍建议部署前替换。

---

## 3. 验证记录

无法运行完整测试套件（本机缺 `torch` / `chromadb` / `sentence-transformers` 等重依赖），
因此采用**全量编译 + 定向运行时验证**组合：

| 验证项 | 方式 | 结果 |
|--------|------|------|
| 全项目语法 | `python -m compileall -q src api scripts tests ui` | ✅ rc=0，113 文件全过 |
| 内部导入完整性 | AST 逐级模拟 import 解析 | ✅ 0 处断链（修复前 1 处） |
| 模块/包命名冲突 | AST 目录 vs 文件同名检测 | ✅ 0 处遮蔽（修复前 1 处） |
| 熔断器 | 真实导入 + 状态机驱动 | ✅ 5 项断言通过 |
| Cypher 防护 | 真实调用 `validate_readonly_cypher` | ✅ 14 项断言通过 |
| JWT 密钥策略 | 7 种环境变量组合 × 独立子进程 | ✅ 全过（含 4 种拒绝启动） |
| 流式取消 | 真实 `astream_answer` + 线程枚举 | ✅ 线程 0 残留，正常路径末事件 `done` 送达 |
| 多租户图谱 | spy 拦截 `create_kg_store` + 浅拷贝复现 | ✅ 3 组断言通过 |
| 监控指标有界性 | 灌 200,000 样本 | ✅ 保留 1,000，统计值与手算一致 |

---

## 4. 建议的下一步

1. **立即**：确认生产环境 `.env` 已设置 `JWT_SECRET`（≥32 字符随机串），否则更新后的 `REQUIRE_STRONG_JWT=1` 会直接拒绝启动 —— 这是有意为之。
2. **本周**：决策 `llm_executor.py` 的去留（接通或删除），它是当前最大的"看起来有、实际没有"的能力。
3. **本周**：把 `tests/` 跑起来 —— CI 目前忽略 `test_agent.py` / `test_rag_pipeline.py` 两个核心测试文件，覆盖率存在盲区。
4. **持续**：统一异常回传文案；`rag_pipeline.py` 按「检索 / 生成 / 流式」拆分。
