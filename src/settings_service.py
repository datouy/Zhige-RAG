"""模型设置：让用户在**界面上**切换模型来源，不必手改配置文件。

设计取舍
--------
- ``config/config.yaml`` 保持"给人手改"的地位，注释与结构完整保留；
- 界面上的修改写入 ``config/local_overrides.yaml``（优先级更高），
  加载配置时 merge 覆盖上去。

这样做的好处：

1. **不破坏主配置**。用 yaml.dump 直接回写 config.yaml 会把注释全部抹掉，
   而那份注释是理解配置的主要途径。
2. **覆盖文件很小**，随时可以删掉回到默认。
3. 用 git 管理主配置时，不会因为"界面上点了几下"产生噪音 diff。

切换模型后端会改变 Embedding 向量维度 → collection 名带指纹，
因此**切 Embedding 等于换了一个库**：旧文档需要重新入库才能被检索到。
这一点必须在接口响应里明确告知用户，不能让数据"静默消失"。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import get_logger, load_config, merge_dict, resolve_path

logger = get_logger("settings_service")

OVERRIDES_RELPATH = "config/local_overrides.yaml"

SUPPORTED_LLM_BACKENDS = ("local", "ollama", "openai")
SUPPORTED_EMB_BACKENDS = ("local", "ollama", "openai")


class SettingsError(ValueError):
    """设置校验失败（消息面向用户）。"""


# ----------------------------------------------------------------------
#  覆盖文件读写
# ----------------------------------------------------------------------
def overrides_path() -> Path:
    return resolve_path(OVERRIDES_RELPATH)


def load_overrides() -> Dict[str, Any]:
    """读取覆盖文件；不存在或损坏时返回空 dict（不阻塞启动）。"""
    path = overrides_path()
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 %s 失败，已忽略：%s", path.name, exc)
        return {}


def save_overrides(data: Dict[str, Any]) -> None:
    """写入覆盖文件（原子写：先写临时文件再替换）。"""
    import yaml

    path = overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("# 由知阁「模型设置」界面写入，优先级高于 config.yaml\n")
        f.write("# 直接删除本文件即可恢复 config.yaml 的配置。\n\n")
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    tmp.replace(path)
    logger.info("模型设置已写入 %s", path.name)


def clear_overrides() -> None:
    """删除覆盖文件，恢复主配置。"""
    path = overrides_path()
    if path.exists():
        path.unlink()
        logger.info("已清除 %s，恢复 config.yaml 配置", path.name)


def effective_config() -> Dict[str, Any]:
    """主配置 + 覆盖文件的合并结果（覆盖优先）。"""
    base = load_config("config/config.yaml")
    overrides = load_overrides()
    return merge_dict(base, overrides) if overrides else base


# ----------------------------------------------------------------------
#  读取当前模型设置
# ----------------------------------------------------------------------
def _mask(secret: str) -> str:
    if not secret:
        return ""
    secret = str(secret)
    if secret.startswith("${"):     # config 里引用环境变量的占位符，原样返回
        return secret
    return f"{secret[:6]}…" if len(secret) > 8 else "已设置"


def get_model_settings(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """返回当前 LLM / Embedding 配置（api_key 脱敏）。"""
    cfg = cfg or effective_config()
    llm = cfg.get("llm") or {}
    emb = cfg.get("embedding") or {}

    def _section(source: Dict[str, Any], backend: str) -> Dict[str, Any]:
        section = source.get(backend) or {}
        return {
            "model": section.get("model") or source.get("model_name", ""),
            "base_url": section.get("base_url", ""),
            "api_key": _mask(section.get("api_key", "")),
        }

    return {
        "llm": {
            "backend": llm.get("backend", "local"),
            "local": {"model": llm.get("model_name", ""), "device": llm.get("device", "auto")},
            "ollama": _section(llm, "ollama"),
            "openai": _section(llm, "openai"),
        },
        "embedding": {
            "backend": emb.get("backend", "local"),
            "local": {"model": emb.get("model_name", ""), "device": emb.get("device", "auto")},
            "ollama": _section(emb, "ollama"),
            "openai": _section(emb, "openai"),
        },
        "supports": {
            "llm_backends": list(SUPPORTED_LLM_BACKENDS),
            "embedding_backends": list(SUPPORTED_EMB_BACKENDS),
        },
        "overrides_active": bool(load_overrides()),
        "overrides_file": OVERRIDES_RELPATH,
    }


# ----------------------------------------------------------------------
#  更新模型设置
# ----------------------------------------------------------------------
def _clean_section(payload: Dict[str, Any]) -> Dict[str, Any]:
    """过滤掉空值，避免把空字符串写进配置覆盖掉有效默认值。"""
    out: Dict[str, Any] = {}
    for key in ("model", "base_url", "api_key"):
        value = payload.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            out[key] = value
    return out


def update_model_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    """更新模型设置：写覆盖文件 → 清缓存 → 重建单例，**立即生效**。

    Args:
        payload: 形如 ``{"llm": {...}, "embedding": {...}}``，每段可含
            ``backend`` 以及 ``ollama`` / ``openai`` 子段。

    Returns:
        与 :func:`get_model_settings` 相同结构的最新配置，外加 ``warnings``。
    """
    from api.deps import _reset_cached_config, reset_all_singletons

    if not isinstance(payload, dict):
        raise SettingsError("请求体必须是对象")

    llm_in = payload.get("llm") or {}
    emb_in = payload.get("embedding") or {}

    llm_backend = str(llm_in.get("backend") or "").strip().lower()
    emb_backend = str(emb_in.get("backend") or "").strip().lower()
    if llm_backend and llm_backend not in SUPPORTED_LLM_BACKENDS:
        raise SettingsError(f"不支持的 LLM 后端：{llm_backend}")
    if emb_backend and emb_backend not in SUPPORTED_EMB_BACKENDS:
        raise SettingsError(f"不支持的 Embedding 后端：{emb_backend}")

    # 先算出"切换前"的 embedding 后端，用于判断是否需要提醒重新入库
    before = effective_config()
    before_emb = (before.get("embedding") or {}).get("backend", "local")

    overrides = load_overrides()
    llm_over = dict(overrides.get("llm") or {})
    emb_over = dict(overrides.get("embedding") or {})

    if llm_backend:
        llm_over["backend"] = llm_backend
    for backend in ("ollama", "openai"):
        section = _clean_section(llm_in.get(backend) or {})
        if section:
            llm_over[backend] = {**(llm_over.get(backend) or {}), **section}

    if emb_backend:
        emb_over["backend"] = emb_backend
    for backend in ("ollama", "openai"):
        section = _clean_section(emb_in.get(backend) or {})
        if section:
            emb_over[backend] = {**(emb_over.get(backend) or {}), **section}

    # 填入子段后，校验必填项（远程后端缺 base_url / model 直接用不了）
    final_emb_backend = emb_over.get("backend", before_emb)
    for target, over, backend, label in (
        ("LLM", llm_over, llm_over.get("backend", ""), "对话模型"),
        ("Embedding", emb_over, final_emb_backend, "向量模型"),
    ):
        if backend in ("ollama", "openai"):
            section = over.get(backend) or {}
            merged_base = section.get("base_url") or (before.get("llm" if target == "LLM" else "embedding") or {}).get(backend, {}).get("base_url")
            merged_model = section.get("model") or (before.get("llm" if target == "LLM" else "embedding") or {}).get(backend, {}).get("model")
            if not merged_base:
                raise SettingsError(f"{label}选择 {backend} 时必须填写 base_url")
            if not merged_model:
                raise SettingsError(f"{label}选择 {backend} 时必须填写 model")

    overrides["llm"] = llm_over
    overrides["embedding"] = emb_over
    save_overrides(overrides)

    # 立即生效：清配置缓存 + 销毁模型/向量库单例（下次请求按新配置重建）
    _reset_cached_config()
    reset_all_singletons()

    warnings = []
    after = effective_config()
    after_emb = (after.get("embedding") or {}).get("backend", "local")
    if after_emb != before_emb:
        warnings.append(
            f"Embedding 后端已从 {before_emb} 切换为 {after_emb}，"
            "向量库会切换到独立 collection —— 之前入库的文档需要重新上传才能被检索到。"
        )

    # 环境变量优先级高于界面设置（12-factor），冲突时必须说清楚，
    # 否则用户会以为"点了保存却没生效"是 bug。
    import os

    conflicts = [
        label
        for var, label in (
            ("LLM_BACKEND", "对话模型后端"),
            ("EMBEDDING_BACKEND", "向量模型后端"),
            ("OLLAMA_BASE_URL", "Ollama 地址"),
            ("OLLAMA_MODEL", "Ollama 模型"),
            ("OPENAI_BASE_URL", "API 地址"),
            ("OPENAI_MODEL", "API 模型"),
            ("EMBEDDING_OLLAMA_BASE_URL", "Embedding Ollama 地址"),
            ("EMBEDDING_OPENAI_BASE_URL", "Embedding API 地址"),
        )
        if os.getenv(var)
    ]
    if conflicts:
        warnings.append(
            "检测到环境变量已设置：" + "、".join(conflicts)
            + "。环境变量的优先级高于本页设置，若要让它生效请先取消这些环境变量。"
        )

    result = get_model_settings(after)
    result["warnings"] = warnings
    logger.info("模型设置已更新：llm=%s, embedding=%s", llm_over.get("backend"), after_emb)
    return result


# ----------------------------------------------------------------------
#  连通性测试
# ----------------------------------------------------------------------
def test_model_connection(payload: Dict[str, Any]) -> Dict[str, Any]:
    """测试某个后端是否可用。**不写入任何配置**。

    Args:
        payload: ``{"kind": "llm"|"embedding", "backend": "...",
                   "base_url": "...", "api_key": "...", "model": "..."}``
    """
    import importlib.util
    import json
    import urllib.error
    import urllib.request

    from src.http_client import build_headers, is_local_endpoint

    kind = str(payload.get("kind") or "llm").lower()
    backend = str(payload.get("backend") or "").strip().lower()
    base_url = str(payload.get("base_url") or "").strip().rstrip("/")
    api_key = str(payload.get("api_key") or "").strip()
    model = str(payload.get("model") or "").strip()

    # ---- 本地后端：测的是"依赖与权重就位没有"，不是网络 ----
    if backend == "local":
        missing = []
        if importlib.util.find_spec("torch") is None:
            missing.append("torch")
        if kind == "embedding" and importlib.util.find_spec("sentence_transformers") is None:
            missing.append("sentence-transformers")
        if missing:
            return {
                "ok": False,
                "detail": f"本地推理依赖未安装：{'、'.join(missing)}。"
                          f"装法 pip install -r requirements-local.txt —— "
                          f"若不想装 torch，改用 ollama 或 openai 后端即可。",
            }
        return {
            "ok": True,
            "detail": "本地推理依赖已就绪（模型权重会在首次使用时自动加载/下载）",
        }

    if backend not in ("ollama", "openai"):
        return {"ok": False, "detail": f"未知后端：{backend or '(空)'}"}
    if not base_url:
        return {"ok": False, "detail": "请先填写 base_url"}

    # ---- 远程后端：GET /models 探活（本地地址自动绕过代理）----
    url = f"{base_url}/models"
    try:
        opener = (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if is_local_endpoint(url)
            else urllib.request.build_opener()
        )
        request = urllib.request.Request(url, headers=build_headers(api_key), method="GET")
        with opener.open(request, timeout=8.0) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore"))
        models = [str((m or {}).get("id", "")) for m in (data.get("data") or [])]
        hit = (not model) or any(model in m for m in models)
        extra = "" if hit else f"；列表里没有 {model}，请确认模型名（可能需要先 pull/部署）"
        return {
            "ok": True,
            "detail": f"服务可达，可用模型 {len(models)} 个{extra}",
            "models": models[:20],
        }
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return {"ok": False, "detail": f"鉴权失败（HTTP {exc.code}）：请检查 API Key 是否正确/已开通"}
        # 有些服务不开放 /models，只要"有响应"就不算失败
        return {
            "ok": True,
            "detail": f"服务有响应（HTTP {exc.code}）。部分服务不开放 /models 接口，可忽略",
        }
    except Exception as exc:  # noqa: BLE001
        if backend == "ollama":
            return {
                "ok": False,
                "detail": f"连不上 Ollama（{base_url}）：{exc}。"
                          f"请确认已执行 `ollama serve`，并已 `ollama pull {model or '模型名'}`；"
                          f"容器内访问宿主机请用 host.docker.internal。",
            }
        return {"ok": False, "detail": f"连不上 {base_url}：{exc}。请检查地址、网络与代理。"}


__all__ = [
    "OVERRIDES_RELPATH",
    "SettingsError",
    "clear_overrides",
    "effective_config",
    "get_model_settings",
    "load_overrides",
    "save_overrides",
    "test_model_connection",
    "update_model_settings",
]
