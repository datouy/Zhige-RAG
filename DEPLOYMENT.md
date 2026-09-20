# 部署手册（Deployment Manual）

> 适用对象：生产环境 / 服务器部署
> 配套系统：中文知识库 RAG 系统（ChineseRAGKB）
> 关联文档：[README.md](README.md) · [INSTALL.md](INSTALL.md)

本手册覆盖三种部署形态：**Docker Compose（推荐单机生产）**、**Kubernetes（集群）**、以及**裸机 / systemd 部署**。并提供配置、反向代理、数据库、安全、监控、备份与故障排查说明。

---

## 1. 系统架构

```
                ┌─────────────────────────────────────────┐
   浏览器 ──────▶│  Streamlit Web UI  (端口 8501)           │
                │     └─ 调用后端 REST / WebSocket          │
                └───────────────┬─────────────────────────┘
                                │  http://localhost:8000
                ┌───────────────▼─────────────────────────┐
                │  FastAPI 后端 (端口 8000)                 │
                │   - /api/health 健康检查                 │
                │   - /api/chat 流式问答 (WebSocket/SSE)    │
                │   - 鉴权 (JWT) / 多租户 / 限流            │
                └───┬───────────┬────────────┬─────────────┘
                    │           │            │
              ┌─────▼────┐ ┌────▼─────┐ ┌───▼────────┐
              │ Chroma   │ │ SQLite / │ │ 可选 Redis │
              │ 向量库   │ │ Postgres │ │ 限流/缓存  │
              └──────────┘ └──────────┘ └────────────┘
                    │
              ┌─────▼───────────────────────────────┐
              │ 本地模型 (./models)                  │
              │  Embedding: bge-small-zh-v1.5        │
              │  LLM:       Qwen2.5-1.5B (4bit)      │
              │  Reranker:   bge-reranker-base(可选) │
              └──────────────────────────────────────┘
```

**服务端口**

| 服务 | 端口 | 说明 |
|------|------|------|
| FastAPI 后端 | 8000 | 业务 API、健康检查、限流、鉴权 |
| Streamlit 前端 | 8501 | Web UI（生产建议置于反向代理后） |
| PostgreSQL | 5432 | 仅 Docker/K8s 编排内使用 |
| Redis | 6379 | 仅 Docker/K8s 编排内使用（限流/缓存） |

---

## 2. 方式一：Docker Compose（推荐）

适用于单机生产，自带 PostgreSQL + Redis + API 服务，模型通过挂载或构建期下载。

### 2.1 准备模型

模型权重（数 GB）默认不进镜像。二选一：

- **构建期下载**：在 `Dockerfile` 后追加 `huggingface-cli download ...`（需构建机可联网）。
- **挂载宿主机模型目录**（推荐）：
  ```yaml
  # docker-compose.yml 的 api 服务 volumes 中取消注释：
  - ${MODEL_CACHE_DIR:-~/.cache}:/root/.cache:ro
  ```
  并确保宿主机已按 INSTALL.md 第 4 节下载好模型。

### 2.2 配置环境变量

创建 `.env`（不要提交）或导出：

```bash
export SECRET_KEY="<至少32位随机串>"
export DATABASE_URL="postgresql+asyncpg://raguser:ragpass@postgres:5432/chineseragkb"
export SYNC_DATABASE_URL="postgresql://raguser:ragpass@postgres:5432/chineseragkb"
export LLM_MODEL_PATH="/models/Qwen2.5-1.5B-Instruct"
export EMBEDDING_MODEL_NAME="BAAI/bge-small-zh-v1.5"
export DEVICE="cuda"          # 无 GPU 改为 cpu
export ALLOWED_ORIGINS="https://your.domain.com"
export RATE_LIMIT_STORAGE_URL="redis://redis:6379/0"
```

### 2.3 启动

```bash
docker compose up -d --build
```

- API：http://localhost:8000 ，健康检查 `GET /api/health`
- 前端：将 8501 经反向代理暴露（见第 5 节）
- 数据持久化：`./data` 与 `./logs` 已挂载为卷；PostgreSQL/Redis 使用命名卷

### 2.4 数据库迁移

```bash
docker compose exec api alembic upgrade head
```

---

## 3. 方式二：Kubernetes

 manifests 位于 `k8s/`：`configmap.yaml` `secret.yaml` `pvc.yaml` `deployment.yaml` `service.yaml` `ingress.yaml` `hpa.yaml`。

### 前置

- Kubernetes 1.24+
- NVIDIA GPU Operator（GPU 场景）
- NGINX Ingress Controller、metrics-server
- 外部 PostgreSQL 15+ 与 Redis 7+（或 operator 提供）

### 步骤

```bash
kubectl create namespace chineseragkb

kubectl -n chineseragkb apply -f configmap.yaml
kubectl -n chineseragkb apply -f secret.yaml      # 含 JWT_SECRET / DB 密码
kubectl -n chineseragkb apply -f pvc.yaml
kubectl -n chineseragkb apply -f deployment.yaml
kubectl -n chineseragkb apply -f service.yaml
kubectl -n chineseragkb apply -f ingress.yaml
kubectl -n chineseragkb apply -f hpa.yaml

# 迁移
kubectl -n chineseragkb exec deploy/chineseragkb-api -- alembic upgrade head

# 查看状态
kubectl -n chineseragkb get pods
kubectl -n chineseragkb logs -l app=chineseragkb -f
```

### 扩缩容

```bash
kubectl -n chineseragkb scale deployment chineseragkb-api --replicas=3
# 或滚动更新
kubectl -n chineseragkb set image deployment/chineseragkb-api api=chineseragkb/api:0.4.0
# 回滚
kubectl -n chineseragkb rollout undo deployment/chineseragkb-api
```

> 注意：多副本时需使用 PostgreSQL + Redis（Stateful 单副本 Chroma 不支持并发写入）。模型建议通过 `emptyDir`/`PVC` 或镜像内预置。

---

## 4. 方式三：裸机 / systemd

适合已有服务器、希望用进程管理器的场景。

```bash
# 1. 按 INSTALL.md 完成安装与环境变量 (.env)
# 2. 以后端为例，创建 /etc/systemd/system/chineseragkb-api.service
```

```ini
[Unit]
Description=ChineseRAGKB API
After=network.target

[Service]
WorkingDirectory=/opt/ChineseRAGKB
EnvironmentFile=/opt/ChineseRAGKB/.env
ExecStart=/opt/ChineseRAGKB/.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
User=appuser

[Install]
WantedBy=multi-user.target
```

前端类似，使用 `streamlit run ui/app.py --server.port 8501 --server.headless=true`。

---

## 5. 反向代理（Nginx 示例）

前端经 8501，后端经 8000，统一域名对外：

```nginx
server {
    listen 80;
    server_name your.domain.com;

    # 前端 UI
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    # 后端 API
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

CORS：生产请将 `ALLOWED_ORIGINS` 设为实际前端域名（逗号分隔），避免 `*`。

---

## 6. 数据库

| 场景 | 配置 |
|------|------|
| 开发 / 演示 | SQLite（`data/users.db`，零配置） |
| 生产 / 多副本 | PostgreSQL（`DATABASE_URL` 环境变量） |

PostgreSQL 连接池（通过环境变量）：

```bash
export DB_POOL_SIZE=20
export DB_MAX_OVERFLOW=40
export DB_POOL_TIMEOUT=30
export DB_POOL_RECYCLE=1800
export DB_POOL_PRE_PING=true
```

迁移：`alembic upgrade head`。回滚：`alembic downgrade -1`。

---

## 7. 安全

1. **密钥**：`JWT_SECRET` / `SECRET_KEY` 必须 ≥32 位随机串，严禁使用默认值。
2. **`.env` 不入库**：已在 `.gitignore` 忽略；生产用 Secret / ConfigMap 注入。
3. **CORS**：`ALLOWED_ORIGINS` 限定域名，不开放 `*`。
4. **网络**：API 与 DB/Redis 仅在内网互通；外部仅暴露 80/443（经反向代理）。
5. **数据本地化**：系统默认完全本地运行，无需上传任何数据到第三方。

---

## 8. 监控与可观测性

- **健康检查**：`GET /api/health`（Docker/K8s 探针均依赖此端点；需安装 `psutil`，缺失时健康检查会 500）。
- **指标**：Prometheus 端点（依赖 `prometheus-client`），可对接 Grafana。
- **审计日志**：`AUDIT_LOG_FILE`（默认 `logs/audit.jsonl`）与可选 DB 审计。
- **请求日志**：`LOG_LEVEL` / `LOG_SAMPLE_RATE` / `SLOW_REQUEST_THRESHOLD_MS` 可配置。

---

## 9. 备份与恢复

所有运行时数据均在 `data/` 目录（向量库 `data/chroma_db`、用户库 `data/users.db`、上传文件 `data/raw`、知识图谱 `data/kg.db` 等）。

```bash
# 备份
tar czf backup-$(date +%F).tgz data/ logs/

# 恢复：停止服务后解压覆盖 data/ 即可
tar xzf backup-YYYY-MM-DD.tgz -C /path/to/ChineseRAGKB
```

> 模型权重（`models/`）体积大且可重新下载，一般不需纳入日常备份。

---

## 10. 故障排查

| 现象 | 排查 |
|------|------|
| 容器一直重启 | 看 `docker compose logs api`；多半是 `SECRET_KEY` 缺省或 DB 未就绪（依赖 `service_healthy`） |
| `/api/health` 500 | 确认 `psutil` 已安装；查看后端日志 |
| 模型加载慢 / OOM | 改用更小模型或关闭 Reranker；GPU 显存不足时 `USE_4BIT=true` |
| K8s Pod `CrashLoopBackOff` | `kubectl -n chineseragkb describe pod <pod>` 与 `--previous` 日志 |
| 前端连不上后端 | 检查 `ALLOWED_ORIGINS` 与反向代理的 `Upgrade` 头（WebSocket） |
| 端口被占用 | 改 `uvicorn --port` 或 `streamlit --server.port` |

更多运行期问题见 [README.md 常见问题](README.md) 与 `logs/` 下日志。
