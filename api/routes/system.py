"""系统/健康/就绪/指标路由。

这些端点均**不需要认证**，但 ``/api/v1/metrics`` 例外。
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from api.deps import (
    _get_health_cache,
    _set_health_cache,
    get_health_cache_ttl,
    get_runtime_config,
    get_vector_store,
    invalidate_health_cache,
)
from src.db.models import User
from src.middleware.identity import get_current_user
from src.utils import get_logger, resolve_path
from sqlalchemy import text as _sa_text

logger = get_logger("api.routes.system")

router = APIRouter()

_ready_cache_at = 0.0
_ready_cache_payload: Dict[str, Any] | None = None


def _clear_readiness_cache() -> None:
    global _ready_cache_at, _ready_cache_payload
    _ready_cache_at = 0.0
    _ready_cache_payload = None

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
        chunks_total = 0
        sources_total = 0

        # ---- Database ----
        db_status = "unknown"
        db_details: Dict[str, Any] = {}
        try:
            from src.db.database import engine

            with engine.connect() as conn:
                conn.execute(_sa_text("SELECT 1"))
                try:
                    sources_total = int(
                        conn.execute(
                            _sa_text("SELECT COUNT(DISTINCT source) FROM document_records")
                        ).scalar()
                        or 0
                    )
                except Exception:
                    sources_total = 0
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
        chroma_status = "unknown"
        chroma_details: Dict[str, Any] = {}
        try:
            from src.vector_store import probe_collections

            def _probe_vector_store():
                return probe_collections(get_vector_store().client)

            report = await run_in_threadpool(_probe_vector_store)
            chunks_total = int(report.get("chunks", 0))
            if report.get("status") == "healthy":
                chroma_status = "healthy"
                chroma_details = {
                    "collections": report.get("collections", 0),
                    "chunks": chunks_total,
                }
            else:
                chroma_status = "unhealthy"
                chroma_details = {
                    "collections": report.get("collections", 0),
                    "errors": len(report.get("errors", [])),
                }
                overall_healthy = False
        except Exception as e:
            chroma_status = "unhealthy"
            chroma_details["error"] = "vector store probe failed"
            logger.error("健康检查中的向量库探测失败: %s", e)
            overall_healthy = False
        checks["chroma"] = {"status": chroma_status, "details": chroma_details}

        # ---- LLM 配置校验 ----
        llm_status = "unknown"
        llm_details: Dict[str, Any] = {}
        try:
            cfg_llm = cfg.get("llm", {})
            backend = str(cfg_llm.get("backend") or "extractive").strip().lower()
            llm_details["backend"] = backend
            if backend == "extractive":
                llm_status = "not_required"
                llm_details["model"] = "none"
            else:
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

        # ---- Storage 一致性（孤儿 collection / 遗留图谱文件）----
        storage_details: Dict[str, Any] = {}
        try:
            from src.db.database import engine as _engine
            from sqlalchemy import text as _sa

            with _engine.connect() as conn:
                active_names = {
                    str(row[0])
                    for row in conn.execute(
                        _sa("SELECT collection_name FROM knowledge_bases")
                    ).fetchall()
                }
            client = get_vector_store().client
            all_names = [str(getattr(c, "name", c)) for c in client.list_collections()]
            orphaned = sorted(n for n in all_names if n and n not in active_names)
            storage_details["chroma_collections"] = len(all_names)
            storage_details["chroma_collections_active"] = len(active_names)
            storage_details["chroma_collections_orphaned"] = len(orphaned)
            if orphaned:
                storage_details["orphaned_sample"] = orphaned[:10]
            data_dir = resolve_path("data")
            storage_details["kg_files"] = (
                len(list(data_dir.glob("kg_*.db"))) if data_dir.exists() else 0
            )
            if orphaned or storage_details["kg_files"] > 0:
                storage_details["suggestion"] = (
                    "存在孤儿资源，可调用 POST /api/v1/system/maintenance/cleanup"
                    "（dry_run=true 预览，dry_run=false 清理）"
                )
        except Exception as e:
            storage_details["error"] = str(e)[:120]
        checks["storage"] = {
            "status": "ok" if "error" not in storage_details else "degraded",
            "details": storage_details,
        }

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
    _clear_readiness_cache()
    return {"status": "ok", "message": "health cache invalidated"}


@router.post("/api/v1/system/maintenance/cleanup", tags=["系统"])
async def maintenance_cleanup(
    dry_run: bool = Query(True, description="true=仅预览；false=实际删除孤儿资源"),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """清理孤儿资源：未被任何知识库引用的 Chroma collection 与无主 KG 文件。

    安全设计：默认 dry_run=true 只返回预览；只有显式传 dry_run=false
    才真正删除。删除对象仅限：

    - Chroma 中不在 knowledge_bases.collection_name 里的 collection；
    - data/kg_*.db 中不属于任何 users 记录、且非配置默认 kg.db 的文件。
    """
    from sqlalchemy import text as _sa
    from src.db.database import engine as _engine

    with _engine.connect() as conn:
        active_names = {
            str(row[0])
            for row in conn.execute(
                _sa("SELECT collection_name FROM knowledge_bases")
            ).fetchall()
        }
        active_users = {
            str(row[0])
            for row in conn.execute(_sa("SELECT id FROM users")).fetchall()
        }

    client = get_vector_store().client
    all_names = [str(getattr(c, "name", c)) for c in client.list_collections()]
    orphan_collections = sorted(n for n in all_names if n and n not in active_names)

    data_dir = resolve_path("data")
    kg_cfg = get_runtime_config().get("knowledge_graph", {}) or {}
    default_kg_name = Path(str(kg_cfg.get("sqlite_path", "data/kg.db"))).name
    orphan_kg_files = []
    for f in sorted(data_dir.glob("kg_*.db")):
        if f.name == default_kg_name:
            continue
        uid = f.name[len("kg_") : -len(".db")]
        if uid not in active_users:
            orphan_kg_files.append(str(f))

    preview = {
        "dry_run": dry_run,
        "orphaned_collections": len(orphan_collections),
        "orphaned_collections_sample": orphan_collections[:20],
        "orphaned_kg_files": len(orphan_kg_files),
        "orphaned_kg_files_sample": [Path(p).name for p in orphan_kg_files[:20]],
    }
    if dry_run:
        return preview

    removed_collections = 0
    for name in orphan_collections:
        try:
            client.delete_collection(name)
            removed_collections += 1
        except Exception as exc:
            logger.warning("删除孤儿 collection %s 失败: %s", name, exc)

    removed_kg = 0
    for path in orphan_kg_files:
        try:
            Path(path).unlink(missing_ok=True)
            removed_kg += 1
        except Exception as exc:
            logger.warning("删除孤儿 KG 文件 %s 失败: %s", path, exc)

    return {
        **preview,
        "removed_collections": removed_collections,
        "removed_kg_files": removed_kg,
    }


@router.get("/api/ready")
async def readiness_check():
    """Kubernetes 就绪探针 - 检查所有依赖服务状态。"""
    global _ready_cache_at, _ready_cache_payload
    ttl = get_health_cache_ttl()
    if _ready_cache_payload is not None and time.monotonic() - _ready_cache_at < ttl:
        return JSONResponse(status_code=200, content=_ready_cache_payload)
    try:
        # 检查数据库连接
        try:
            from src.db.database import engine

            with engine.connect() as conn:
                conn.execute(_sa_text("SELECT 1"))
            db_status = "connected"
        except Exception as e:
            db_status = f"error: {str(e)}"

        # 检查 Chroma 向量库：文件存在不等于 collection 可读。
        chroma_detail: Dict[str, Any] = {}
        try:
            def _probe_vector_store():
                return get_vector_store().health_check()

            report = await run_in_threadpool(_probe_vector_store)
            chroma_status = "connected"
            chroma_detail = {
                "collections": report.get("collections", 0),
                "chunks": report.get("chunks", 0),
                "query_probe": report.get("query_probe", False),
            }
        except Exception as e:
            logger.error("向量库就绪探测失败: %s", e)
            chroma_status = "error"
            chroma_detail = {"error": "vector store probe failed"}

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

        payload = {
            "status": "ready" if is_ready else "not_ready",
            "checks": {
                "database": db_status,
                "chroma": chroma_status,
                "chroma_detail": chroma_detail,
                "llm": llm_status,
            },
        }
        if is_ready:
            _ready_cache_payload = payload
            _ready_cache_at = time.monotonic()
        return JSONResponse(status_code=200 if is_ready else 503, content=payload)
    except Exception as exc:
        _clear_readiness_cache()
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
    """JSON 格式指标端点。"""
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

    这是本地单用户服务：读取者就是本机使用者，不做额外的权限判定；
    配置里的敏感键（密码 / 密钥 / token）由 ``_redact`` 统一替换为 ``***``。
    """
    try:
        cfg = get_runtime_config()
        return _redact(cfg)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("获取配置失败: %s", exc)
        raise HTTPException(status_code=500, detail="获取配置失败")
