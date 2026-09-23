#!/usr/bin/env python
"""一键启动 —— 从 clone 到能用，尽量不让你翻文档。

用法::

    python scripts/quickstart.py                      # 检查环境并启动
    python scripts/quickstart.py --check              # 只检查，不启动
    python scripts/quickstart.py --install            # 缺依赖时自动安装
    python scripts/quickstart.py --backend ollama     # 切换模型后端后启动
    python scripts/quickstart.py --backend openai --yes

它按顺序做四件事：

1. 校验 Python 版本与虚拟环境；
2. 检查核心依赖（缺失则提示，``--install`` 时自动安装）；
3. 按当前 ``backend`` 确认模型**真的可用**（Ollama 在不在跑 / API Key 配没配）；
4. 调用 ``scripts/start_all.py`` 拉起服务。

设计说明：``--backend`` 通过**环境变量**传给子进程，不修改 ``config.yaml``，
因此可以随时试不同后端而不用改文件（同一个环境变量在 Docker 里也能用）。
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

OK, WARN, FAIL = "✅", "⚠️ ", "❌"


def say(icon: str, message: str) -> None:
    print(f"{icon} {message}")


def header(title: str) -> None:
    print()
    print("─" * 66)
    print(f"  {title}")
    print("─" * 66)


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
#  2. 依赖
# ----------------------------------------------------------------------
def missing_core() -> List[Tuple[str, str, str]]:
    try:
        from scripts.doctor import DEPS_CORE, has_module
    except Exception:  # noqa: BLE001 - doctor 不可用时不阻塞启动
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
        say("  ", "处理方式二选一：")
        say("  ", f"  ① 自动安装：{sys.executable} scripts/quickstart.py --install")
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
#  3. 模型后端可用性
# ----------------------------------------------------------------------
def load_cfg() -> Dict[str, Any]:
    try:
        from src.utils import apply_env_overrides, load_config

        return apply_env_overrides(load_config(str(ROOT / "config" / "config.yaml")))
    except Exception:  # noqa: BLE001
        return {}


def _probe(url: str, api_key: str = "", timeout: float = 3.0) -> Tuple[Optional[int], str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        from src.http_client import is_local_endpoint

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


def check_llm_backend(cfg: Dict[str, Any]) -> bool:
    llm = cfg.get("llm") or {}
    backend = str(os.getenv("LLM_BACKEND") or llm.get("backend") or "local").lower()
    say("  ", f"LLM 后端：{backend}")

    if backend == "ollama":
        section = llm.get("ollama") or {}
        base = os.getenv("OLLAMA_BASE_URL") or section.get("base_url") or "http://localhost:11434/v1"
        model = os.getenv("OLLAMA_MODEL") or section.get("model") or ""
        status, body = _probe(f"{base.rstrip('/')}/models")
        if status == 200:
            say(OK, f"Ollama 可达（{base}）")
            if model and model not in body:
                say(WARN, f"Ollama 中没有模型 {model}")
                say("  ", f"请先拉取：ollama pull {model}")
                return False
            if model:
                say(OK, f"模型已就绪：{model}")
            return True
        say(FAIL, f"Ollama 不可达：{base}")
        say("  ", "① 安装并启动 Ollama：https://ollama.com/download → `ollama serve`")
        if model:
            say("  ", f"② 拉取模型：ollama pull {model}")
        if os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY"):
            say("  ", "③ 检测到代理环境变量：若连不上，请把该地址加入 NO_PROXY")
        return False

    if backend == "openai":
        section = llm.get("openai") or {}
        base = os.getenv("OPENAI_BASE_URL") or section.get("base_url") or ""
        api_key = section.get("api_key") or os.getenv("OPENAI_API_KEY", "")
        if not api_key or str(api_key).startswith("${"):
            say(FAIL, "使用 openai 后端，但没有配置 API Key")
            say("  ", "在 .env 中设置 OPENAI_API_KEY=sk-... ，或改用 --backend ollama")
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
            say("  ", "三选一：① 把 llm.local_files_only 改为 false；② 放入权重；")
            say("  ", "        ③ 改用更省事的路 —— python scripts/quickstart.py --backend ollama")
            return False
        say(WARN, f"本地模型尚未下载，首次使用会联网拉取：{model_name}")
        return True

    say(FAIL, f"未知后端：{backend}（可选 local / ollama / openai）")
    return False


def check_embedding_backend(cfg: Dict[str, Any]) -> bool:
    emb = cfg.get("embedding") or {}
    backend = str(os.getenv("EMBEDDING_BACKEND") or emb.get("backend") or "local").lower()
    say("  ", f"Embedding 后端：{backend}")

    if backend == "ollama":
        section = emb.get("ollama") or {}
        base = (
            os.getenv("EMBEDDING_OLLAMA_BASE_URL")
            or section.get("base_url")
            or "http://localhost:11434/v1"
        )
        model = os.getenv("EMBEDDING_OLLAMA_MODEL") or section.get("model") or ""
        status, body = _probe(f"{base.rstrip('/')}/models")
        if status != 200:
            say(FAIL, f"Ollama 不可达（{base}），Embedding 无法工作")
            return False
        say(OK, "Ollama 可达")
        if model and model not in body:
            say(WARN, f"缺少 embedding 模型 {model}")
            say("  ", f"请先拉取：ollama pull {model}")
            return False
        return True

    if backend == "openai":
        section = emb.get("openai") or {}
        api_key = section.get("api_key") or os.getenv("OPENAI_API_KEY", "")
        if not api_key or str(api_key).startswith("${"):
            say(FAIL, "使用 openai Embedding 但没有 API Key")
            return False
        say(OK, "Embedding API Key 已配置")
        return True

    if backend == "local":
        try:
            import importlib.util

            if importlib.util.find_spec("sentence_transformers") is None:
                say(FAIL, "本地 Embedding 需要 sentence-transformers（含 torch，约 2~3 GB）")
                say("  ", "不想装 torch？改用：--backend ollama（同时会切 Embedding 到 Ollama）")
                return False
        except Exception:  # noqa: BLE001
            pass
        cache_dir = emb.get("cache_dir")
        if cache_dir and emb.get("local_files_only"):
            cache_path = Path(cache_dir)
            if not cache_path.is_absolute():
                cache_path = ROOT / cache_dir
            if not (cache_path.exists() and any(cache_path.iterdir())):
                say(FAIL, f"Embedding 缓存为空且 local_files_only=true：{cache_path}")
                say("  ", "把 embedding.local_files_only 改为 false，或改用 --backend ollama")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="知阁 Zhige 一键启动")
    parser.add_argument("--check", action="store_true", help="只做检查，不启动服务")
    parser.add_argument("--install", action="store_true", help="缺少核心依赖时自动安装")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认")
    parser.add_argument(
        "--backend",
        choices=["local", "ollama", "openai"],
        help="临时切换 LLM 与 Embedding 后端（通过环境变量，不改配置文件）",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["local", "ollama", "openai"],
        help="单独指定 Embedding 后端（默认跟随 --backend 中可复用的一方）",
    )
    args = parser.parse_args()

    extra_env: Dict[str, str] = {}
    if args.backend:
        extra_env["LLM_BACKEND"] = args.backend
        # embedding 默认跟随：ollama / openai 两边都支持，local 也一致
        extra_env["EMBEDDING_BACKEND"] = args.embedding_backend or args.backend
    elif args.embedding_backend:
        extra_env["EMBEDDING_BACKEND"] = args.embedding_backend

    # 通过环境变量把覆盖透传给本进程，后续检查与子进程才会一致
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
    llm_ok = check_llm_backend(cfg)
    emb_ok = check_embedding_backend(cfg)
    if not (llm_ok and emb_ok):
        print()
        say(WARN, "模型后端未就绪 —— 服务能启动，但问答会失败。")
        if not args.yes and not _confirm("仍要继续启动吗？"):
            return 1

    if args.check:
        header("检查完成")
        say(OK if (llm_ok and emb_ok) else WARN, "已按 --check 要求跳过启动")
        return 0 if (llm_ok and emb_ok) else 1

    header("4/4 启动服务")
    return launch(interpreter, extra_env)


if __name__ == "__main__":
    raise SystemExit(main())
