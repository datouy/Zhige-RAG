#!/usr/bin/env python
"""环境自检 —— 一条命令回答"为什么跑不起来"。

定位
----
本项目依赖较重（torch / chromadb / sentence-transformers 等），且模型来源
可切换（本地权重 / Ollama / OpenAI 兼容 API）。新手最常卡在三个地方：

1. 依赖没装全（且报错信息指向不到真正的缺失项）；
2. ``llm.local_files_only: true`` 但模型权重根本没下载；
3. 机器上有全局代理，导致 localhost 的 Ollama 永远连不上。

本脚本把这三类问题一次性诊断出来，并给出可直接执行的处理建议。

用法::

    python scripts/doctor.py
    python scripts/doctor.py --config config/config.yaml --port 8000

退出码：0 = 未发现阻塞性问题；1 = 存在必须修复的问题。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OK, WARN, FAIL, INFO = "OK", "WARN", "FAIL", "INFO"
_ICON = {OK: "✅", WARN: "⚠️ ", FAIL: "❌", INFO: "ℹ️ "}

_findings: List[Tuple[str, str, str]] = []


def add(level: str, title: str, hint: str = "") -> None:
    _findings.append((level, title, hint))


def has_module(module_name: str) -> bool:
    """检测模块是否可导入（只看 spec，不真正 import，快且无副作用）。"""
    if not module_name:
        return False
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


# ======================================================================
#  依赖分组：(pip 包名, 导入名, 说明)
# ======================================================================
DEPS_CORE = [
    ("fastapi", "fastapi", "Web 框架"),
    ("uvicorn", "uvicorn", "ASGI 服务器"),
    ("pydantic", "pydantic", "数据校验"),
    ("sqlalchemy", "sqlalchemy", "ORM"),
    ("pyjwt", "jwt", "JWT 认证"),
    ("bcrypt", "bcrypt", "密码哈希"),
    ("PyYAML", "yaml", "配置解析"),
    ("chromadb", "chromadb", "向量库"),
    ("numpy", "numpy", "数值计算"),
]

DEPS_LOCAL_LLM = [
    ("torch", "torch", "推理引擎"),
    ("transformers", "transformers", "模型加载"),
    ("accelerate", "accelerate", "设备编排"),
    ("safetensors", "safetensors", "权重格式"),
    ("sentencepiece", "sentencepiece", "分词"),
]

DEPS_EMBEDDING = [
    ("sentence-transformers", "sentence_transformers", "本地 Embedding"),
]

DEPS_DOCS = [
    ("pdfplumber", "pdfplumber", "PDF 解析"),
    ("pypdf", "pypdf", "PDF 兜底"),
    ("python-docx", "docx", "Word 解析"),
    ("markdown", "markdown", "Markdown 解析"),
    ("beautifulsoup4", "bs4", "HTML 解析"),
    ("jieba", "jieba", "中文分词（BM25）"),
]

DEPS_UI = [
    ("streamlit", "streamlit", "Streamlit 界面"),
]

DEPS_OPS = [
    ("psutil", "psutil", "健康检查"),
    ("prometheus-client", "prometheus_client", "指标导出"),
]


def check_python_version() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 10):
        add(OK, f"Python {major}.{minor}.{sys.version_info[2]}（要求 ≥ 3.10）")
    else:
        add(
            FAIL,
            f"Python {major}.{minor} 过低（代码使用了 `X | None` 等 3.10+ 语法）",
            "请安装 Python 3.10 及以上版本。",
        )


def check_dep_group(title: str, deps: List[Tuple[str, str, str]], blocker: bool) -> None:
    missing = [f"{pip}（{desc}）" for pip, mod, desc in deps if not has_module(mod)]
    if not missing:
        add(OK, f"{title}：{len(deps)} 项依赖齐备")
        return
    level = FAIL if blocker else WARN
    add(
        level,
        f"{title}：缺少 {len(missing)}/{len(deps)} 项 —— " + "、".join(missing),
        (
            "安装：pip install -r requirements-core.txt"
            if blocker
            else "安装（可选能力）：pip install -r requirements-optional.txt"
        ),
    )


def check_config(config_path: Path) -> Optional[Dict[str, Any]]:
    if not config_path.exists():
        add(FAIL, f"配置文件不存在：{config_path}", "请确认在项目根目录执行，或显式指定 --config。")
        return None
    if not has_module("yaml"):
        add(FAIL, "缺少 PyYAML，无法读取配置", "pip install PyYAML")
        return None
    try:
        from src.utils import apply_env_overrides, load_config

        cfg = apply_env_overrides(load_config(str(config_path)))
    except Exception as exc:  # noqa: BLE001
        add(FAIL, f"配置解析失败：{type(exc).__name__}: {exc}", "检查 config.yaml 的缩进与语法。")
        return None

    add(OK, f"配置解析成功：{config_path.name}（{len(cfg)} 个顶级字段）")

    # 配置校验器（pydantic 存在时顺便跑一遍 schema）
    if has_module("pydantic"):
        try:
            from src.config.validator import validate_config

            validate_config(cfg)
            add(OK, "配置 schema 校验通过")
        except Exception as exc:  # noqa: BLE001
            add(WARN, f"配置 schema 校验未通过：{str(exc)[:160]}", "不影响启动，但建议修正。")
    return cfg


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


def _probe(url: str, api_key: str = "", timeout: float = 3.0) -> Tuple[Optional[int], str]:
    """GET 探测，本地地址自动绕过代理。"""
    import urllib.error
    import urllib.request

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        from src.llm_provider import is_local_endpoint

        local = is_local_endpoint(url)
    except Exception:  # noqa: BLE001
        local = False
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if local
        else urllib.request.build_opener()
    )
    try:
        with opener.open(request, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def check_llm_backend(cfg: Dict[str, Any]) -> None:
    llm_cfg = cfg.get("llm") or {}
    backend = str(llm_cfg.get("backend") or "local").strip().lower()

    if backend == "local":
        check_dep_group("本地推理依赖（backend: local）", DEPS_LOCAL_LLM, blocker=True)
        model_name = llm_cfg.get("model_name", "")
        model_path = _resolve(model_name)
        if model_path.exists():
            add(OK, f"本地模型已就位：{model_name}")
        elif llm_cfg.get("local_files_only"):
            add(
                FAIL,
                f"本地模型不存在，但 local_files_only=true：{model_path}",
                "三选一：① 把 llm.local_files_only 改为 false（首次自动下载）；"
                "② 将权重放入上述目录；③ 改用 backend: ollama 或 openai（无需下载、无需 GPU）。",
            )
        else:
            add(
                WARN,
                f"本地模型尚未下载（首次运行会联网拉取）：{model_name}",
                "下载需要数百 MB~数 GB，请确保网络可达 HuggingFace（国内可设 HF_ENDPOINT=https://hf-mirror.com）。",
            )

    elif backend == "ollama":
        section = llm_cfg.get("ollama") or {}
        base_url = section.get("base_url") or "http://localhost:11434/v1"
        model = section.get("model") or "(未配置)"
        status, body = _probe(f"{base_url.rstrip('/')}/models")
        if status == 200:
            add(OK, f"Ollama 可达：{base_url}")
            if model and model != "(未配置)" and model not in body:
                add(
                    WARN,
                    f"Ollama 服务里没有模型 {model}",
                    f"执行：ollama pull {model}",
                )
            elif model and model != "(未配置)":
                add(OK, f"Ollama 已安装模型 {model}")
        elif status is None:
            add(
                FAIL,
                f"Ollama 不可达：{base_url}（{body[:120]}）",
                "① 启动服务：ollama serve；② 若在 Docker 内运行，base_url 不能写 localhost，"
                "应指向宿主机（如 http://host.docker.internal:11434/v1）。",
            )
        else:
            add(WARN, f"Ollama 返回 HTTP {status}：{base_url}", "确认服务版本与路径是否正确。")

    elif backend == "openai":
        section = llm_cfg.get("openai") or {}
        base_url = section.get("base_url") or "https://api.openai.com/v1"
        api_key = section.get("api_key") or os.getenv("OPENAI_API_KEY", "")
        model = section.get("model") or "(未配置)"
        if not api_key or api_key.startswith("${"):
            add(
                FAIL,
                "使用 openai 后端但未配置 api_key",
                "在 .env 中设置 OPENAI_API_KEY=sk-...（config.yaml 的 openai.api_key 已引用该变量）。",
            )
        else:
            add(OK, f"API Key 已配置（{api_key[:6]}…）")
        status, body = _probe(f"{base_url.rstrip('/')}/models", api_key=api_key)
        if status == 200:
            add(OK, f"模型服务可达：{base_url}")
        elif status in (401, 403):
            add(FAIL, f"鉴权失败（HTTP {status}）：{base_url}", "检查 api_key 是否正确/已过期/已开通该模型。")
        elif status is None:
            add(WARN, f"模型服务不可达：{base_url}（{body[:120]}）", "检查网络、base_url 拼写与代理设置。")
        else:
            add(WARN, f"模型服务返回 HTTP {status}：{base_url}", "部分服务未开放 /models，可忽略；若问答报错再排查。")
        add(INFO, f"当前模型：{model}")
    else:
        add(
            FAIL,
            f"未知的 llm.backend：{backend!r}",
            "可选值：local / ollama / openai",
        )

    # 代理提示
    proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or os.getenv("HTTPS_PROXY")
    if proxy and backend in ("ollama", "openai"):
        add(
            INFO,
            f"检测到代理环境变量：{proxy}",
            "本项目对 localhost/内网地址已自动绕过代理；若仍连不上，请把该地址加入 NO_PROXY。",
        )


def check_embedding(cfg: Dict[str, Any]) -> None:
    """按 ``embedding.backend`` 分支检查。

    只有 local 后端才需要 sentence-transformers 与本地缓存；若用户在
    ollama / openai 后端下仍看到"缺少本地 Embedding 依赖"，那是指错了方向。
    """
    emb_cfg = cfg.get("embedding") or {}
    backend = str(emb_cfg.get("backend") or "local").strip().lower()

    if backend == "local":
        check_dep_group("Embedding 依赖（backend: local）", DEPS_EMBEDDING, blocker=True)
        cache_dir = emb_cfg.get("cache_dir")
        if cache_dir and emb_cfg.get("local_files_only"):
            cache_path = _resolve(cache_dir)
            hit = cache_path.exists() and any(cache_path.iterdir())
            if hit:
                add(OK, f"Embedding 缓存目录已存在：{cache_dir}")
            else:
                add(
                    FAIL,
                    f"Embedding 缓存为空，但 local_files_only=true：{cache_path}",
                    "把 embedding.local_files_only 改为 false 让它在首次运行时下载，"
                    "或提前执行：python scripts/preload_llm.py；"
                    "若不想装 torch，可改用 embedding.backend: ollama。",
                )
        return

    if backend == "ollama":
        section = emb_cfg.get("ollama") or {}
        base_url = section.get("base_url") or "http://localhost:11434/v1"
        model = section.get("model") or ""
        status, body = _probe(f"{base_url.rstrip('/')}/models")
        if status == 200:
            add(OK, f"Ollama 可达（Embedding）：{base_url}")
            if model and model not in body:
                add(WARN, f"Ollama 中没有 embedding 模型 {model}", f"执行：ollama pull {model}")
            elif model:
                add(OK, f"Ollama embedding 模型已就绪：{model}")
        elif status is None:
            add(
                FAIL,
                f"Ollama 不可达（Embedding）：{base_url}（{body[:120]}）",
                "① 启动：ollama serve；② 拉取模型：ollama pull "
                f"{model or 'nomic-embed-text'}；③ 容器内请用 host.docker.internal。",
            )
        else:
            add(WARN, f"Ollama 返回 HTTP {status}（Embedding）：{base_url}")
        return

    if backend == "openai":
        section = emb_cfg.get("openai") or {}
        base_url = section.get("base_url") or "https://api.openai.com/v1"
        api_key = section.get("api_key") or os.getenv("OPENAI_API_KEY", "")
        if not api_key or str(api_key).startswith("${"):
            add(
                FAIL,
                "使用 openai Embedding 但没有配置 api_key",
                "在 .env 设置 OPENAI_API_KEY，或改用 embedding.backend: ollama。",
            )
        else:
            add(OK, f"Embedding API Key 已配置（{str(api_key)[:6]}…）")
        status, body = _probe(f"{base_url.rstrip('/')}/models", api_key=str(api_key))
        if status == 200:
            add(OK, f"Embedding 服务可达：{base_url}")
        elif status in (401, 403):
            add(FAIL, f"Embedding 鉴权失败（HTTP {status}）", "检查 api_key 是否有效。")
        elif status is None:
            add(WARN, f"Embedding 服务不可达：{base_url}（{body[:120]}）")
        return

    add(FAIL, f"未知的 embedding.backend：{backend!r}", "可选值：local / ollama / openai")


def check_writable_dirs(cfg: Dict[str, Any]) -> None:
    targets = [("数据目录", ROOT / "data")]
    for key in ("chroma_db", "logs"):
        raw = ((cfg.get("paths") or {}).get(key)) if cfg else None
        if raw:
            targets.append((f"paths.{key}", _resolve(raw)))
    for label, path in targets:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".doctor_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            add(OK, f"{label}可写：{path}")
        except Exception as exc:  # noqa: BLE001
            add(FAIL, f"{label}不可写：{path}（{type(exc).__name__}）", "检查目录权限或磁盘空间。")


def check_ports(ports: List[int]) -> None:
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            busy = sock.connect_ex(("127.0.0.1", port)) == 0
        if busy:
            add(INFO, f"端口 {port} 已被占用", "若启动服务报 Address already in use，请先停止占用进程或换端口。")
        else:
            add(OK, f"端口 {port} 空闲")


def render() -> int:
    print("=" * 74)
    print("知阁 Zhige 环境自检")
    print(f"项目根目录：{ROOT}")
    print("=" * 74)
    order = {FAIL: 0, WARN: 1, OK: 2, INFO: 3}
    for level, title, hint in sorted(_findings, key=lambda f: order[f[0]]):
        print(f"{_ICON[level]} [{level:<4}] {title}")
        if hint:
            for line in hint.split("\n"):
                print(f"          → {line}")

    n_fail = sum(1 for f in _findings if f[0] == FAIL)
    n_warn = sum(1 for f in _findings if f[0] == WARN)
    print("=" * 74)
    print(f"结果：{n_fail} 项阻塞问题，{n_warn} 项警告，共 {len(_findings)} 项检查")

    if n_fail:
        print("\n结论：**当前环境无法正常启动**，请按上面 ❌ 项逐条修复。")
        print("提示：若你只想先跑起来看看，最快路径是把 config/config.yaml 的")
        print("      llm.backend 改为 ollama 或 openai —— 这两条路都不需要 GPU，也不需要下载权重。")
        return 1
    if n_warn:
        print("\n结论：核心链路可用，存在若干非阻塞警告。")
        return 0
    print("\n结论：环境就绪，可以启动服务。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="知阁 Zhige 环境自检")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--port", type=int, action="append", default=None, help="额外检查的端口（可重复）")
    args = parser.parse_args()

    check_python_version()

    cfg = check_config(_resolve(args.config))
    if cfg is not None:
        check_llm_backend(cfg)
        check_embedding(cfg)
        check_doc_group(cfg)
        check_writable_dirs(cfg)

    check_dep_group("核心 Web/存储依赖", DEPS_CORE, blocker=True)
    check_dep_group("文档解析依赖", DEPS_DOCS, blocker=False)
    check_dep_group("UI 依赖", DEPS_UI, blocker=False)
    check_dep_group("运维/监控依赖", DEPS_OPS, blocker=False)

    ports = [8000, 8501] + (args.port or [])
    check_ports(sorted(set(ports)))

    return render()


def check_doc_group(cfg: Dict[str, Any]) -> None:
    """文档解析是 RAG 的入口能力，缺了就没法入库，单独提示一次。"""
    exts = ((cfg.get("document_loader") or {}).get("supported_extensions")) or []
    if exts:
        add(INFO, f"已配置支持的文件类型：{' '.join(exts)}")

    ocr_cfg = (cfg.get("document_loader") or {}).get("ocr") or {}
    if not ocr_cfg.get("enabled", True):
        add(INFO, "扫描件 OCR 已在配置中关闭")
        return
    try:
        from src.document_loader import ocr_available

        if ocr_available():
            add(OK, "扫描件 OCR 可用（RapidOCR）：图片型 PDF 也能提取文字")
        else:
            add(
                WARN,
                "未安装 OCR 依赖：图片型 PDF（扫描件/拍照件）无法提取文字",
                "需要时安装：pip install -r requirements-ocr.txt "
                "（约 100~150 MB，纯 CPU 可跑；不装不影响文本型 PDF 与 Word）。",
            )
    except Exception as exc:  # noqa: BLE001
        add(INFO, f"OCR 能力检测跳过：{type(exc).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
