"""模型设置路由——让用户在界面上切换模型来源。

端点：

- GET    /api/v1/settings/model         读取当前 LLM / Embedding 配置（Key 脱敏）
- PUT    /api/v1/settings/model         更新配置并**立即生效**（写 local_overrides.yaml）
- POST   /api/v1/settings/model/test    测试某个后端是否可用（不写入配置）
- DELETE /api/v1/settings/model         清除界面覆盖，恢复到 config.yaml

设计说明见 :mod:`src.settings_service`：界面改动写在
``config/local_overrides.yaml``，不碰 ``config/config.yaml``，
因此手写的注释与注释里的说明不会丢。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.db.models import User
from src.middleware.auth import get_current_user
from src.settings_service import (
    SettingsError,
    clear_overrides,
    get_model_settings,
    test_model_connection,
    update_model_settings,
)
from src.utils import get_logger

logger = get_logger("api.routes.settings")

router = APIRouter()


class BackendSection(BaseModel):
    """某个后端的连接参数。"""
    model: Optional[str] = Field(None, description="模型名，如 qwen2.5:7b / deepseek-chat")
    base_url: Optional[str] = Field(None, description="服务地址，如 http://localhost:11434/v1")
    api_key: Optional[str] = Field(None, description="API Key（本地服务可留空）")


class KindSettings(BaseModel):
    """一类组件（llm 或 embedding）的设置。"""
    backend: Optional[str] = Field(None, description="local | ollama | openai")
    ollama: Optional[BackendSection] = None
    openai: Optional[BackendSection] = None


class ModelSettingsBody(BaseModel):
    llm: Optional[KindSettings] = None
    embedding: Optional[KindSettings] = None


class TestBody(BaseModel):
    kind: str = Field("llm", pattern="^(llm|embedding)$")
    backend: str = Field(..., description="local | ollama | openai")
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None


def _require_admin_if_auth(user: User) -> None:
    """启用认证时，改模型配置属于管理员操作。

    本地单用户模式下 get_current_user 返回的本地用户本身就是 admin，直接放行。
    """
    if getattr(user, "is_admin", False):
        return
    raise HTTPException(status_code=403, detail="只有管理员可以修改模型设置")


@router.get("/api/v1/settings/model", tags=["设置"])
async def api_get_model_settings(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """读取当前模型配置（api_key 已脱敏）。"""
    try:
        return get_model_settings()
    except Exception as exc:  # noqa: BLE001
        logger.error("读取模型设置失败: %s", exc)
        raise HTTPException(status_code=500, detail="读取模型设置失败") from exc


@router.put("/api/v1/settings/model", tags=["设置"])
async def api_update_model_settings(
    body: ModelSettingsBody,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """更新模型配置并**立即生效**（无需重启）。

    返回体里的 ``warnings`` 要展示给用户 —— 尤其是切换 Embedding 后端时，
    向量库会切到独立 collection，旧文档需要重新入库。
    """
    _require_admin_if_auth(current_user)
    try:
        payload = body.model_dump(exclude_none=True)
        return update_model_settings(payload)
    except SettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("更新模型设置失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"更新模型设置失败：{exc}") from exc


@router.post("/api/v1/settings/model/test", tags=["设置"])
async def api_test_model(
    body: TestBody,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """测试连接。不写入任何配置，可以放心试。"""
    try:
        return test_model_connection(body.model_dump(exclude_none=True))
    except Exception as exc:  # noqa: BLE001
        logger.error("测试连接失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"测试失败：{exc}") from exc


@router.delete("/api/v1/settings/model", tags=["设置"])
async def api_clear_model_settings(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """清除界面覆盖，恢复到 ``config/config.yaml`` 的配置。"""
    _require_admin_if_auth(current_user)
    from api.deps import _reset_cached_config, reset_all_singletons

    clear_overrides()
    _reset_cached_config()
    reset_all_singletons()
    result = get_model_settings()
    result["warnings"] = ["已恢复为 config/config.yaml 中的配置"]
    return result
