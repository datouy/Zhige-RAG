"""系统/健康/就绪/指标路由。

这些端点均**不需要认证**，但 ``/api/v1/metrics`` 例外。
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response

from api.deps import (
    _get_health_cache,
    _set_health_cache,
    get_health_cache_ttl,
    get_runtime_config,
    invalidate_health_cache,
)
from src.middleware.auth import get_current_user
from src.utils import get_logger, resolve_path
from sqlalchemy import text as _sa_text

logger = get_logger("api.routes.system")

router = APIRouter()

# 递归脱敏时命中这些键名（不区分大小写）的值会被替换为 "***"
_SENSITIVE_KEY_RE = re.compile(
    r"password|passwd|secret|api_key|apikey|token|credential", re.IGNORECASE
)

# psutil 是可选依赖：缺失时磁盘/内存检查降级为 unavailable，而不是让整个
# /api/health 抛 500（k8s 探针会把服务判死）。requirements.txt 已声明 psutil。
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


def _redact(value: Any) -> Any:
    """递归脱敏：dict/list 深遍历，命中敏感键的标量值替换为 "***"。"""
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if _SENSITIVE_KEY_RE.search(str(k)) and not isinstance(v, (dict, list)):
                out[k] = "***"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


@router.get("/api/health")
async def health_check():
    """增强健康检查（带 TTL 缓存，避免高频探针穿透）。

    检查项：
    - PostgreSQL/SQLite 数据库连接
    - ChromaDB 向量库状态
    - LLM 模型可用性（仅校验配置，不真正加载大模型）
    - 磁盘 / 内存 / Redis（若配置）

    缓存 TTL 由 ``HEALTH_CACHE_TTL`` 环境变量控制（默认 5 秒）。
    """
    # 命中缓存：直接返回，节省 psutil / sqlite 查询
    cached = _get_health_cache()
    if cached is not None:
        return cached

    try:
        cfg = get_runtime_config()
        checks: Dict[str, Any] = {}
        overall_healthy = True

        # ---- Database ----
        db_status = "unknown"
        db_details: Dict[str, Any] = {}
        try:
            from src.db.database import engine

            with engine.connect() as conn:
                conn.execute(_sa_text("SELECT 1"))
            db_status = "healthy"
            db_details["type"] = (
                "postgresql" if str(engine.url).startswith("postgresql") else "sqlite"
            )
        except Exception as e:
            db_status = "unhealthy"
            db_details["error"] = str(e)[:100]
            overall_healthy = False
        checks["database"] = {"status": db_status, "details": db_details}

        # ---- Chroma ----
        chunks_total = 0
        sources_total = 0
        chroma_status = "unknown"
        chroma_details: Dict[str, Any] = {}
        try:
            persist_dir = resolve_path(cfg["vector_store"]["persist_directory"])
            # chromadb >=0.4 持久化文件名为 chroma.sqlite3（旧版为 chroma.sqlite）
            chroma_db = persist_dir / "chroma.sqlite3"
            if not chroma_db.exists():
                chroma_db = persist_dir / "chroma.sqlite"

            if chroma_db.exists():
                conn = sqlite3.connect(str(chroma_db))
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM embeddings")
                chunks_total = cur.fetchone()[0]
                # chromadb >=1.x 把 metadata 拆到 embedding_metadata 表（key/value），
                # 旧版 (<0.6) 则是 embeddings.metadata JSON 列；按 schema 兼容查询
                try:
                    cur.execute(
                        "SELECT COUNT(DISTINCT string_value) FROM embedding_metadata "
                        "WHERE key = 'source'"
                    )
                    sources_total = cur.fetchone()[0] or 0
                except sqlite3.OperationalError:
                    try:
                        cur.execute(
                            "SELECT COUNT(DISTINCT metadata->>'$.source') FROM embeddings"
                        )
                        sources_total = cur.fetchone()[0] or 0
                    except sqlite3.OperationalError:
                        sources_total = 0
                conn.close()
                chroma_status = "healthy"
                chroma_details = {"chunks": chunks_total, "sources": sources_total}
            else:
                chroma_status = "not_initialized"
                chroma_details = {"message": "ChromaDB not yet initialized"}
        except Exception as e:
            chroma_status = "unhealthy"
            chroma_details["error"] = str(e)[:100]
            overall_healthy = False
        checks["chroma"] = {"status": chroma_status, "details": chroma_details}

        # ---- LLM 配置校验 ----
        llm_status = "unknown"
        llm_details: Dict[str, Any] = {}
        try:
            cfg_llm = cfg.get("llm", {})
            model_name = cfg_llm.get("model_name", "unknown")
            cache_dir = cfg_llm.get("cache_dir")
            if cache_dir:
                cache_path = resolve_path(cache_dir)
                llm_details["cache_dir"] = str(cache_path)
                llm_details["cache_exists"] = cache_path.exists()
            llm_status = "available"
            llm_details["model"] = model_name
        except Exception as e:
            llm_status = "not_available"
            llm_details = {"error": str(e)[:100]}
        checks["llm"] = {"status": llm_status, "details": llm_details}

        # ---- Disk ----
        disk_status = "unknown"
        disk_details: Dict[str, Any] = {}
        try:
            if psutil is None:
                disk_status = "unavailable"
                disk_details = {"message": "psutil 未安装，磁盘检查不可用"}
            elif os.name == "nt":
                disk = psutil.disk_usage("C:")
                disk_details = {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent": disk.percent,
                }
                disk_status = "healthy" if disk.percent < 90 else "warning"
            else:
                disk = psutil.disk_usage("/")
                disk_details = {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent": disk.percent,
                }
                disk_status = "healthy" if disk.percent < 90 else "warning"
        except Exception as e:
            disk_status = "error"
            disk_details = {"error": str(e)[:100]}
        checks["disk"] = {"status": disk_status, "details": disk_details}

        # ---- Memory ----
        memory_status = "unknown"
        memory_details: Dict[str, Any] = {}
        try:
            if psutil is None:
                memory_status = "unavailable"
                memory_details = {"message": "psutil 未安装，内存检查不可用"}
            else:
                mem = psutil.virtual_memory()
                memory_details = {
                    "total_gb": round(mem.total / (1024**3), 2),
                    "available_gb": round(mem.available / (1024**3), 2),
                    "percent": mem.percent,
                }
                memory_status = "healthy" if mem.percent < 90 else "warning"
        except Exception as e:
            memory_status = "error"
            memory_details = {"error": str(e)[:100]}
        checks["memory"] = {"status": memory_status, "details": memory_details}

        # ---- Redis ----
        redis_status = "not_configured"
        redis_details: Dict[str, Any] = {}
        try:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                import redis

                r = redis.from_url(redis_url, socket_connect_timeout=2)
                r.ping()
                redis_status = "healthy"
                redis_details = {"connected": True}
            else:
                redis_details = {"message": "Redis not configured"}
        except Exception as e:
            redis_status = "unavailable"
            redis_details = {"error": str(e)[:100]}
        checks["redis"] = {"status": redis_status, "details": redis_details}

        payload: Dict[str, Any] = {
            "status": "ok" if overall_healthy else "degraded",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": "0.3.0",
            "chunks": chunks_total,
            "sources": sources_total,
            "checks": checks,
            "cached_for_seconds": get_health_cache_ttl(),
        }

        _set_health_cache(payload)
        return payload

    except Exception as exc:
        logger.error("健康检查失败: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(exc),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )


@router.get("/api/health/invalidate")
async def health_invalidate(current_user: User = Depends(get_current_user)):
    """主动失效健康检查缓存（运维端点）。

    P5 审计修复：需要认证。之前完全匿名可调用，任何人都能高频刷此端点
    让健康检查反复穿透缓存。
    """
    invalidate_health_cache()
    return {"status": "ok", "message": "health cache invalidated"}


@router.get("/api/ready")
async def readiness_check():
    """Kubernetes 就绪探针 - 检查所有依赖服务状态。"""
    try:
        # 检查数据库连接
        try:
            from src.db.database import engine

            with engine.connect() as conn:
                conn.execute(_sa_text("SELECT 1"))
            db_status = "connected"
        except Exception as e:
            db_status = f"error: {str(e)}"

        # 检查 Chroma 向量库
        try:
            cfg = get_runtime_config()
            persist_dir = resolve_path(cfg["vector_store"]["persist_directory"])
            chroma_db = persist_dir / "chroma.sqlite3"
            if not chroma_db.exists():
                chroma_db = persist_dir / "chroma.sqlite"
            if chroma_db.exists():
                conn = sqlite3.connect(str(chroma_db))
                conn.close()
                chroma_status = "connected"
            else:
                chroma_status = "not_initialized"
        except Exception as e:
            chroma_status = f"error: {str(e)}"

        # 检查 LLM 可用性（仅校验配置）
        try:
            cfg = get_runtime_config()
            cache_dir = cfg.get("llm", {}).get("cache_dir")
            if cache_dir:
                resolve_path(cache_dir)  # 校验路径合法
            llm_status = "available"
        except Exception:
            llm_status = "not_available"

        is_ready = db_status == "connected" and chroma_status in (
            "connected",
            "not_initialized",
        )

        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={
                "status": "ready" if is_ready else "not_ready",
                "checks": {
                    "database": db_status,
                    "chroma": chroma_status,
                    "llm": llm_status,
                },
            },
        )
    except Exception as exc:
        logger.error("就绪检查失败: %s", exc)
        return JSONResponse(
            status_code=503,
            # 不回传 str(exc)：内部路径/驱动报错会泄露给探针调用方
            content={"status": "not_ready", "error": "readiness check failed"},
        )


@router.get("/api/metrics")
async def prometheus_metrics():
    """Prometheus 格式指标端点。"""
    try:
        from src.monitoring.metrics import metrics_collector, PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            return {
                "error": "Prometheus metrics not available",
                "hint": "pip install prometheus-client psutil",
            }

        metrics_output, content_type = metrics_collector.generate_metrics_output()
        return Response(content=metrics_output, media_type=content_type)
    except Exception as exc:
        logger.error("获取 Prometheus 指标失败: %s", exc)
        raise HTTPException(status_code=500, detail="获取指标失败")


@router.get("/api/v1/metrics")
async def get_metrics(current_user: User = Depends(get_current_user)):
    """JSON 格式指标端点。需要认证。

    P5 审计修复：docstring 与 main.py 文档一直声称需要认证，但实现漏掉了
    ``Depends(get_current_user)``，实际向匿名访客暴露系统指标。
    """
    try:
        from src.monitoring.metrics import metrics_collector, PROMETHEUS_AVAILABLE

        system_metrics = metrics_collector.get_system_metrics()

        if PROMETHEUS_AVAILABLE:
            return {
                "prometheus_available": True,
                "system": system_metrics,
                "prometheus_endpoint": "/api/metrics",
            }
        from scripts.monitoring.metrics import MetricsCollector

        custom_metrics = MetricsCollector.get_instance().get_summary()
        return {
            "prometheus_available": False,
            "system": system_metrics,
            "custom": custom_metrics,
        }
    except Exception as exc:
        logger.error("获取指标失败: %s", exc)
        raise HTTPException(status_code=500, detail="获取指标失败")


@router.get("/api/v1/config")
async def get_user_config(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """获取全量运行时配置（递归脱敏）。

    仅管理员可读。此前 docstring 声称"含管理员校验"，但实现只挂了
    ``get_current_user``（仅表示已登录），任何普通用户都能拉取全量配置——
    其中包含 DB URI、模型路径、内部开关等信息。
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="仅管理员可访问")
    try:
        cfg = get_runtime_config()
        return _redact(cfg)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("获取配置失败: %s", exc)
        raise HTTPException(status_code=500, detail="获取配置失败")