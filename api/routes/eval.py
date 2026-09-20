"""评估路由。"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_runtime_config, get_runtime_pipeline
from src.db.models import User
from src.middleware.auth import get_current_user
from src.utils import get_logger, resolve_path

logger = get_logger("api.routes.eval")

router = APIRouter()


@router.post("/api/v1/eval/run", tags=["评估"])
def run_evaluation(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """运行评估（使用已配置的评估集）。需要认证。

    必须是 ``def``（同步端点）：``evaluate()`` 会循环执行全量
    ``answer_with_timing``，分钟级阻塞。之前写成 ``async def`` 直接调用，
    会把整个事件循环（包括 /api/health 探针）卡死，k8s 会判死重启。
    FastAPI 自动把 ``def`` 端点调度到线程池，不会阻塞事件循环。
    """
    user: User = current_user  # type: ignore[assignment]
    try:
        from scripts.evaluate import evaluate as run_eval, load_eval_dataset

        cfg = get_runtime_config()
        dataset_path = resolve_path(
            cfg.get("evaluation", {}).get("dataset_path", "data/eval/eval_set.jsonl")
        )

        if not dataset_path.exists():
            raise HTTPException(
                status_code=404, detail=f"评估集不存在: {dataset_path}"
            )

        # P1.4：复用单例
        pipeline = get_runtime_pipeline(user_id=user.id)
        items = load_eval_dataset(dataset_path)
        summary = run_eval(pipeline, items)

        return summary
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("评估失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))