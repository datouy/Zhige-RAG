# 合并收尾与问题修复报告

> 日期：2026-10-15
> 范围：`ChineseRAGKB`（知阁）
> 合并提交：`688e24f`（parents: `663fbb4` ← 本地 main，`d87e341` ← github/main）

---

## 一、合并冲突处理

**状态**：`git merge` 曾停在冲突中（`README.md` 为 `UU`），现已解决并完成合并提交。

**冲突实况**（三方对比）：

| 比较 | 差异 |
| --- | --- |
| base → 本地 ours | **仅 1 行**：给 README 加了 `README_optimization.md` / `CHANGELOG_optimization.md` 两个文档导航链接 |
| base → 远端 theirs | **整体重写**：298 增 / 368 删，README 结构完全换新 |
| 工作区 → theirs | 123 行新增，全部是本轮的 d1 决策层内容 |

**处理方式**：接受工作区版本（= 远端重写版 + 本轮 d1 内容）。

理由：本地那 1 行导航已被整体重写取代，且它指向的 `INSTALL.md` / `DEPLOYMENT.md`
正是本次合并要删除的文件；导航目标已不存在，保留会造成死链。

---

## 二、修复的生产代码缺陷

### 1. `api/routes/agent.py`：残留 `executor` 引用（NameError）

**问题**：P0-1 那轮把 `AsyncLLMExecutor` 删除并改用 `LLMConcurrencyGate`，但漏改了
一处引用 —— 第 255 行仍是 `getattr(executor, "semaphore", None)`，而同函数第 208 行
已经取好的 `gate = get_llm_gate()` 反而从未被使用。

**后果**：`/api/v1/ws/agent` 收到**第一条有效消息**时，`_run()` 之前就抛
`NameError: name 'executor' is not defined`，Agent WebSocket 完全不可用。

**修复**：`executor` → `gate`。

### 2. `api/routes/agent.py`：`/api/v1/ws/agent` 认证策略不一致

**问题**：本地工作把 `/ws/chat` 放宽为「本地单用户模式不校验 token」，但**漏改了 agent WS**。

证据链：

- 前端 `ui/web/js/app.js:558` 明确注释：*「本地模式没有 token，就不带这个参数（后端也不会校验）」*；
- 同文件 `/api/v1/ws/agent` 的两个 REST 兄弟端点用的是 `Depends(get_current_user)`，
  而 `get_current_user` 在本地模式下返回本地用户（放行）；
- 只有 agent WS 依然**无条件**要求 token。

**后果**：本地模式下前端拿不到 token，Agent WebSocket 连接会被 `4401` 直接拒绝 —— 该功能形同不可用。

**修复**：统一为 `auth_enabled()` 分支策略，与 `/ws/chat` 完全一致：

- 未启用认证（默认）：绑定本地用户；
- 启用认证：校验 access token + 用户存在性/启用状态（4401 / 4403）。

**实测验证**：

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| 本地模式连接 `/api/v1/ws/agent` | 被拒绝 | **连接成功** ✓ |
| 认证模式 + 无效 token | 拒绝 | 拒绝 ✓ |

### 3. `api/deps.py`：缺失 `logger` 导入（本轮早先已修）

`api/deps.py` 使用了 `logger.warning`，但全文从未定义 `logger` —— 决策层初始化一旦
失败就会抛 `NameError`，把真实错误吞掉。已补 `get_logger` 导入。

顺带新增 `get_decision_provider()`：轻量获取决策 provider，不必为了拿一个 provider
而构建整条 Pipeline（embedding + Chroma）。

---

## 三、修复的测试

### 根因 A：测试替身签名过期（这是"挂起"的真正原因）

路由早已改为 `get_runtime_pipeline(user_id=..., kb=...)`（多知识库支持），但两个测试文件的
替身仍是 `lambda user_id=None`：

- `tests/test_api.py`
- `tests/test_integration.py`

后果链（**同一个根因造成 500 与卡死两种表象**）：

```
TypeError: unexpected keyword argument 'kb'
  ├─ REST /api/v1/chat  → 500（4 个用例失败）
  └─ WS   /ws/chat      → 服务端回 error 事件，测试一直等 done → 永久阻塞（表现为"超时"）
```

修复：两处替身改为 `lambda user_id=None, kb=None`。

### 根因 B：断言 401 / 空库的用例没考虑本地模式

`auth.enabled: false` 是**默认设计**（单用户本地模式，打开即用），此时匿名请求会落到
本地用户 → 返回 200；而所有请求也共用同一个本地用户，其库中可能已有历史数据。

| 文件 | 用例数 | 处理 |
| --- | --- | --- |
| `tests/test_auth.py` | 2 | 显式 `AUTH_ENABLED=1` |
| `tests/test_websocket.py` | 3 | 显式 `AUTH_ENABLED=1` |
| `tests/test_feedback_api.py` | 1 | 显式 `AUTH_ENABLED=1` |
| `tests/test_api.py` | 2 | 显式 `AUTH_ENABLED=1` |
| `tests/test_integration.py` | 2（空状态断言） | 显式 `AUTH_ENABLED=1`，改用新注册的隔离用户 |

（`auth_enabled()` 是**调用时**读环境变量，因此 `monkeypatch.setenv` 立即生效。）

### 新增回归测试（防止再次退化）

1. `test_websocket_agent_allows_local_mode` —— agent WS 在本地模式下可连接；
2. `test_websocket_agent_streams_without_nameerror` —— 有效消息能正常产出事件，
   不再因 `executor` 残留而 NameError。

---

## 四、验证结果

```
pytest tests/ -q   →   437 passed, 0 failed   (2m28s)
```

对比修复前：

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 失败用例 | 4+ | **0** |
| 挂起/超时的测试文件 | `test_api.py`、`test_integration.py` | **无** |
| 可跑完的全量测试 | 否 | **437 全绿** |

---

## 五、提交结果

```
* 688e24f Merge github/main: 修复 agent WS 认证不一致与残留 executor 引用
|\
| * d87e341 feat: 多知识库 + 会话管理 + OCR/URL导入 + 免登录 + 界面模型设置
* | 663fbb4 perf(chunks): 检索与分块优化 ...
```

- 合并提交含 **140 个文件**变更（+21528 / −5855）；
- 工作区已**完全干净**（`git status` 无输出）；
- 本地 `main` 领先 `origin/main` **7 个提交**。

**远程**：

| 名称 | 地址 |
| --- | --- |
| `origin` | https://gitee.com/tokyi/agent.git |
| `github` | https://github.com/datouy/rag.git |

> 尚未 push —— 需要推送到 `origin`（Gitee）和/或 `github` 请明确告知。
