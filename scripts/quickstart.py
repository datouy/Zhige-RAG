#!/usr/bin/env python
"""知阁 Zhige 一键启动 —— 从 clone 到能用，尽量不让你翻文档。

用法::

    python scripts/quickstart.py                      # 智能检测并启动
    python scripts/quickstart.py --check              # 仅检查环境
    python scripts/quickstart.py --install            # 缺少依赖时自动安装
    python scripts/quickstart.py --backend extractive # 无 LLM 原文摘录
    python scripts/quickstart.py --backend openai     # 强制使用云端 API
    python scripts/quickstart.py --wizard             # 交互式向导

设计说明：
- 自动检测最佳可用的模型后端
- 提供清晰的配置指导
- 默认使用 extractive + ONNX，无需 GPU、torch 或 LLM
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECOMMENDED_OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"

# Windows terminals often default to GBK. ASCII status labels keep the startup
# checks readable and prevent a UnicodeEncodeError before the service starts.
OK, WARN, FAIL, INFO = "[OK]", "[WARN]", "[FAIL]", "[INFO]"


def say(icon: str, message: str) -> None:
    print(f"{icon} {message}")


def header(title: str) -> None:
    print()
    print("=" * 66)
    print(f"  {title}")
    print("=" * 66)


def _reexec_in_project_venv() -> Optional[int]:
    """Use the project's virtualenv even when quickstart is called by system Python."""
    if os.environ.get("ZHIGE_VENV_REEXEC") == "1":
        return None
    existing = venv_python()
    if not existing:
        return None
    current = Path(sys.executable).resolve()
    if current == existing.resolve():
        return None
    print(f"[INFO] 切换到项目虚拟环境：{existing}")
    env = dict(os.environ, ZHIGE_VENV_REEXEC="1")
    return subprocess.call(
        [str(existing), str(Path(__file__).resolve()), *sys.argv[1:]],
        env=env,
        cwd=str(ROOT),
    )


# ----------------------------------------------------------------------
#  1. Python 与虚拟环境
# ----------------------------------------------------------------------
def check_python() -> bool:
    v = sys.version_info
    if (v.major, v.minor) >= (3, 10):
        say(OK, f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    say(FAIL, f"Python {v.major}.{v.minor} 过低 —— 代码使用了 `X | None` 等 3.10+ 语法")
    say("  ", "请安装 Python 3.10+ 后重试")
    return False


def venv_python() -> Optional[Path]:
    """返回项目内 .venv 的解释器路径（不存在则 None）。"""
    candidates = [
        ROOT / ".venv" / "Scripts" / "python.exe",   # Windows
        ROOT / ".venv" / "bin" / "python",           # macOS / Linux
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def check_venv() -> Tuple[bool, Optional[Path]]:
    """检查虚拟环境。返回 (是否可直接继续, .venv 解释器路径)。"""
    if sys.prefix != sys.base_prefix:
        say(OK, "当前已在虚拟环境中运行")
        return True, venv_python() or Path(sys.executable)

    existing = venv_python()
    if existing:
        say(WARN, f"当前不在虚拟环境，但项目内已有 .venv：{existing}")
        say("  ", "建议用它运行：.venv\\Scripts\\activate  （Windows）")
        say("  ", "                  source .venv/bin/activate （macOS/Linux）")
        return True, existing

    say(WARN, "当前不在虚拟环境，且项目内没有 .venv")
    say("  ", "建议先创建（避免污染系统 Python）：")
    say("  ", "  python -m venv .venv")
    say("  ", "  .venv\\Scripts\\activate            # Windows")
    say("  ", "  source .venv/bin/activate          # macOS / Linux")
    return True, Path(sys.executable)


# ----------------------------------------------------------------------
#  2. 依赖检查
# ----------------------------------------------------------------------
def missing_core() -> List[Tuple[str, str, str]]:
    try:
        from scripts.doctor import DEPS_CORE, has_module
    except Exception:
        return []
    return [(pip, mod, desc) for pip, mod, desc in DEPS_CORE if not has_module(mod)]


def install_requirements(files: List[Path]) -> bool:
    targets = [str(f) for f in files if f.exists()]
    if not targets:
        say(FAIL, "找不到依赖清单文件")
        return False
    cmd = [sys.executable, "-m", "pip", "install", "-r"] + targets
    print(f"   $ {' '.join(cmd)}")
    return subprocess.call(cmd) == 0


def check_deps(interpreter: Path, auto_install: bool, assume_yes: bool) -> bool:
    missing = missing_core()
    if not missing:
        say(OK, "核心依赖齐备")
        return True

    names = "、".join(f"{pip}" for pip, _, _ in missing)
    say(FAIL, f"缺少 {len(missing)} 项核心依赖：{names}")

    if not auto_install:
        say("  ", "处理方式：")
        say("  ", f"  ① 自动安装：python scripts\\quickstart.py --install")
        say("  ", "  ② 手动安装：pip install -r requirements-core.txt")
        return False

    if not assume_yes and not _confirm("现在自动安装核心依赖？（不含 torch，通常几分钟）"):
        return False

    say("  ", "开始安装核心依赖 ...")
    ok = install_requirements([ROOT / "requirements-core.txt"])
    if ok:
        say(OK, "核心依赖安装完成")
    else:
        say(FAIL, "核心依赖安装失败，请检查网络或使用国内镜像（pip -i ...）")
    return ok


def _confirm(question: str) -> bool:
    """交互确认；非 tty 环境（CI / 管道）直接返回 False，不阻塞。"""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        return input(f"   {question} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ----------------------------------------------------------------------
#  3. 模型后端检测
# ----------------------------------------------------------------------
def load_cfg() -> Dict[str, Any]:
    try:
        from src.utils import apply_env_overrides, load_config

        return apply_env_overrides(load_config(str(ROOT / "config" / "config.yaml")))
    except Exception:
        return {}


def _probe(url: str, api_key: str = "", timeout: float = 3.0) -> Tuple[Optional[int], str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        from src.http_client import is_local_endpoint
        local = is_local_endpoint(url)
    except Exception:
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
    except Exception as exc:
        return None, str(exc)[:80]


def check_ollama() -> Tuple[bool, str, str]:
    """检查 Ollama 是否可用，返回 (是否可用, 版本信息, 模型列表)。"""
    base = "http://localhost:11434"
    try:
        status, body = _probe(f"{base}/api/version")
        if status == 200:
            # 获取模型列表
            _, models = _probe(f"{base}/api/tags")
            return True, "运行中", models[:500] if models else ""
        return False, f"HTTP {status}", ""
    except Exception as exc:
        return False, str(exc)[:50], ""


def _ollama_root(base: str) -> str:
    """Accept both native and OpenAI-compatible Ollama base URLs."""
    root = str(base or "http://localhost:11434").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


def check_api_key() -> Optional[str]:
    """检查是否有可用的 API Key。"""
    # 尝试从多个位置读取
    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key and not env_key.startswith("${"):
        return env_key
    
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY=") and not line.endswith("="):
                key = line.split("=", 1)[1].strip()
                if key and not key.startswith("${"):
                    return key
    return None


def auto_detect_backend() -> str:
    """Keep the configured LLM-free backend unless the user opts in."""
    say(INFO, "使用配置中的无 LLM 后端（extractive + ONNX）")
    cfg = load_cfg()
    return str((cfg.get("llm") or {}).get("backend") or "extractive").lower()


def check_llm_backend(cfg: Dict[str, Any], force_backend: Optional[str] = None) -> bool:
    llm = cfg.get("llm") or {}
    backend = force_backend or str(os.getenv("LLM_BACKEND") or llm.get("backend") or "extractive").lower()
    say("  ", f"对话模型后端：{backend}")

    if backend == "extractive":
        say(OK, "无 LLM 模式：直接返回带引用的知识库原文")
        return True

    if backend == "ollama":
        section = llm.get("ollama") or {}
        base = os.getenv("OLLAMA_BASE_URL") or section.get("base_url") or "http://localhost:11434"
        model = os.getenv("OLLAMA_MODEL") or section.get("model") or ""
        
        status, body = _probe(f"{_ollama_root(base)}/api/tags")
        if status == 200:
            say(OK, f"Ollama 可达（{base}）")
            if model and model not in body:
                say(WARN, f"Ollama 中没有模型 {model}")
                say("  ", f"请先拉取：ollama pull {model}")
                say("  ", f"推荐模型：ollama pull {RECOMMENDED_OLLAMA_MODEL}")
                return False
            if model:
                say(OK, f"模型已就绪：{model}")
            return True
        
        say(FAIL, f"Ollama 不可达：{base}")
        say("  ", "① 安装 Ollama：https://ollama.com/download")
        say("  ", "② 启动服务：ollama serve")
        say("  ", f"③ 拉取模型：ollama pull {RECOMMENDED_OLLAMA_MODEL}")
        if os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY"):
            say("  ", "[WARN] 检测到代理环境变量，可能需要加入 NO_PROXY")
        return False

    if backend == "openai":
        section = llm.get("openai") or {}
        base = os.getenv("OPENAI_BASE_URL") or section.get("base_url") or ""
        api_key = section.get("api_key") or os.getenv("OPENAI_API_KEY", "") or check_api_key() or ""
        
        if not api_key or str(api_key).startswith("${"):
            say(FAIL, "使用 openai 后端，但没有配置 API Key")
            say("  ", "设置方式：")
            say("  ", "  - 环境变量：OPENAI_API_KEY=sk-...")
            say("  ", "  - 或在界面「模型设置」中配置")
            say("  ", "  - 或改用 Ollama：--backend ollama")
            return False
        say(OK, f"API Key 已配置（{str(api_key)[:6]}…）")
        status, _ = _probe(f"{base.rstrip('/')}/models", api_key=str(api_key))
        if status == 200:
            say(OK, f"模型服务可达（{base}）")
            return True
        if status in (401, 403):
            say(FAIL, f"鉴权失败（HTTP {status}），请检查 API Key")
            return False
        say(WARN, f"模型服务返回 {status}，部分服务不开放 /models 接口，可忽略")
        return True

    if backend == "local":
        model_name = llm.get("model_name", "")
        model_path = Path(model_name)
        if not model_path.is_absolute():
            model_path = ROOT / model_name
        if model_path.exists():
            say(OK, f"本地模型已就位：{model_name}")
            return True
        if llm.get("local_files_only"):
            say(FAIL, f"本地模型不存在，但 local_files_only=true：{model_path}")
            say("  ", "建议改用 Ollama 或云端 API：")
            say("  ", "  python scripts\\quickstart.py --backend ollama")
            return False
        say(WARN, f"本地模型尚未下载，首次使用会联网拉取：{model_name}")
        return True

    say(FAIL, f"未知后端：{backend}（可选 extractive / local / ollama / openai）")
    return False


def check_embedding_backend(cfg: Dict[str, Any], force_backend: Optional[str] = None) -> bool:
    emb = cfg.get("embedding") or {}
    backend = force_backend or str(os.getenv("EMBEDDING_BACKEND") or emb.get("backend") or "onnx").lower()
    say("  ", f"向量模型后端：{backend}")

    if backend == "onnx":
        section = emb.get("onnx") or {}
        model_dir = Path(section.get("model_path") or emb.get("model_name") or "models/bge-small-zh-v1.5-onnx")
        if not model_dir.is_absolute():
            model_dir = ROOT / model_dir
        required = [
            model_dir / section.get("model_file", "onnx/model_quantized.onnx"),
            model_dir / section.get("tokenizer_file", "tokenizer.json"),
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            say(FAIL, "ONNX Embedding 文件缺失：" + "、".join(missing))
            say("  ", "运行：python scripts/setup_onnx_embedding.py")
            return False
        say(OK, f"ONNX Embedding 已就位：{model_dir}")
        return True

    if backend == "ollama":
        section = emb.get("ollama") or {}
        base = (
            os.getenv("OLLAMA_BASE_URL")
            or os.getenv("EMBEDDING_OLLAMA_BASE_URL")
            or section.get("base_url")
            or "http://localhost:11434"
        )
        model = os.getenv("EMBEDDING_OLLAMA_MODEL") or section.get("model") or ""
        status, body = _probe(f"{_ollama_root(base)}/api/tags")
        if status != 200:
            say(FAIL, f"Ollama 不可达（{base}），Embedding 无法工作")
            return False
        say(OK, "Ollama 可达")
        if model and model not in body:
            say(WARN, f"缺少 embedding 模型 {model}")
            say("  ", f"请先拉取：ollama pull {model or 'nomic-embed-text'}")
            say("  ", "推荐：ollama pull nomic-embed-text")
            return False
        return True

    if backend == "openai":
        section = emb.get("openai") or {}
        api_key = section.get("api_key") or os.getenv("OPENAI_API_KEY", "") or check_api_key() or ""
        if not api_key or str(api_key).startswith("${"):
            say(WARN, "使用 openai Embedding 但没有 API Key（可能未配置）")
            return False
        say(OK, "Embedding API Key 已配置")
        return True

    if backend == "local":
        try:
            import importlib.util
            if importlib.util.find_spec("sentence_transformers") is None:
                say(FAIL, "本地 Embedding 需要 sentence-transformers（含 torch，约 2~3 GB）")
                say("  ", "不想装 torch？改用：--backend ollama")
                return False
        except Exception:
            pass
        cache_dir = emb.get("cache_dir")
        if cache_dir and emb.get("local_files_only"):
            cache_path = Path(cache_dir)
            if not cache_path.is_absolute():
                cache_path = ROOT / cache_dir
            if not (cache_path.exists() and any(cache_path.iterdir())):
                say(FAIL, f"Embedding 缓存为空且 local_files_only=true：{cache_path}")
                say("  ", "建议改用 --backend ollama")
                return False
        say(OK, "本地 Embedding 配置可用")
        return True

    say(FAIL, f"未知 Embedding 后端：{backend}")
    return False


# ----------------------------------------------------------------------
#  4. 启动
# ----------------------------------------------------------------------
def launch(interpreter: Path, extra_env: Dict[str, str]) -> int:
    script = ROOT / "scripts" / "start_all.py"
    env = dict(os.environ, **extra_env)
    say("  ", f"启动命令：{interpreter} {script}")
    return subprocess.call([str(interpreter), str(script)], env=env, cwd=str(ROOT))


def show_recommendations():
    """显示配置建议。"""
    print()
    print("推荐配置（按推荐程度排序）")
    print("  1. 无 LLM（当前默认，无需 GPU / torch）")
    print("     - 回答：带引用的知识库原文摘录")
    print("     - 向量：ONNX BGE，运行于 CPU")
    print("     - 准备：python scripts\\setup_onnx_embedding.py")
    print("  2. 可选云端 API（按量付费）")
    print("     - DeepSeek: https://platform.deepseek.com/")
    print("     - 通义千问: https://dashscope.console.aliyun.com/")
    print("     - 设置环境变量 OPENAI_API_KEY 后启动")
    print("  3. 可选生成式本地模型（需要 torch；本机已移除）")
    print("     - 安装：pip install -r requirements-core.txt -r requirements-local.txt")


def main() -> int:
    reexec_result = _reexec_in_project_venv()
    if reexec_result is not None:
        return reexec_result

    parser = argparse.ArgumentParser(description="知阁 Zhige 一键启动")
    parser.add_argument("--check", action="store_true", help="只做检查，不启动服务")
    parser.add_argument("--install", action="store_true", help="缺少核心依赖时自动安装")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认")
    parser.add_argument("--wizard", action="store_true", help="交互式配置向导")
    parser.add_argument(
        "--backend",
        choices=["extractive", "local", "ollama", "openai"],
        help="指定模型后端（通过环境变量，不改配置文件）",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["onnx", "local", "ollama", "openai"],
        help="单独指定 Embedding 后端",
    )
    args = parser.parse_args()

    extra_env: Dict[str, str] = {}
    if args.backend:
        extra_env["LLM_BACKEND"] = args.backend
    if args.embedding_backend:
        extra_env["EMBEDDING_BACKEND"] = args.embedding_backend

    for key, value in extra_env.items():
        os.environ[key] = value

    print("=" * 66)
    print("  知阁 Zhige 一键启动")
    print(f"  项目目录：{ROOT}")
    print("=" * 66)

    header("1/4 运行环境")
    if not check_python():
        return 1
    _, interpreter = check_venv()

    header("2/4 依赖检查")
    if not check_deps(interpreter, args.install, args.yes):
        print()
        say(WARN, "依赖不完整，已停止。补全后重新运行本脚本即可。")
        return 1

    header("3/4 模型后端")
    cfg = load_cfg()
    
    # 确定后端
    backend = args.backend or os.getenv("LLM_BACKEND") or cfg.get("llm", {}).get("backend", "extractive")
    emb_backend = args.embedding_backend or os.getenv("EMBEDDING_BACKEND") or cfg.get("embedding", {}).get("backend", "onnx")
    
    if backend == "unknown":
        # 尝试自动检测
        detected = auto_detect_backend()
        if detected != "unknown" and detected != backend:
            say(INFO, f"自动切换到 {detected} 后端")
            backend = detected
            emb_backend = detected
            extra_env["LLM_BACKEND"] = backend
    
    llm_ok = check_llm_backend(cfg, backend)
    emb_ok = check_embedding_backend(cfg, emb_backend)
    
    if not (llm_ok and emb_ok):
        print()
        show_recommendations()
        if not args.yes and not _confirm("仍要继续启动吗？（模型不可用，问答会失败）"):
            return 1

    if args.check:
        header("检查完成")
        say(OK if (llm_ok and emb_ok) else WARN, "已按 --check 要求跳过启动")
        return 0 if (llm_ok and emb_ok) else 1

    header("4/4 启动服务")
    return launch(interpreter, extra_env)


if __name__ == "__main__":
    raise SystemExit(main())
