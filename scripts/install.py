#!/usr/bin/env python
"""知阁 Zhige - 一键安装脚本 (Windows)
提供两种安装模式：
  1. 快速模式（推荐）：不安装 torch，使用 Ollama 或云端 API，无需 GPU
  2. 完整模式：安装 torch + 本地模型，支持完全离线运行

用法:
    python scripts/install.py              # 快速安装（推荐）
    python scripts/install.py --full       # 完整安装（需要 torch + 本地模型）
    python scripts/install.py --ollama      # 安装 Ollama 后端依赖（推荐）
    python scripts/install.py --check       # 仅检查环境，不安装
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = ROOT / ".venv" / "bin" / "python"
RECOMMENDED_OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"


def run(cmd: list[str], check: bool = False, env: dict | None = None, 
        capture: bool = True) -> subprocess.CompletedProcess:
    merged = dict(os.environ, **(env or {}))
    return subprocess.run(cmd, env=merged, check=check,
                         text=True, capture_output=capture)


def step(n: int, total: int, msg: str) -> None:
    print(f"\n[{n}/{total}] {msg}")


def check_ollama_installed() -> bool:
    """检查 Ollama 是否已安装"""
    try:
        r = run(["ollama", "--version"], capture=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def check_ollama_running() -> bool:
    """检查 Ollama 服务是否在运行"""
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/version")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def main() -> int:
    global VENV_PY

    parser = argparse.ArgumentParser(description="知阁 Zhige 一键安装")
    parser.add_argument("--full", action="store_true", help="完整安装（包含 torch + 本地模型）")
    parser.add_argument("--ollama", action="store_true", help="安装 Ollama 后端依赖（推荐，无需 GPU）")
    parser.add_argument("--check", action="store_true", help="仅检查环境，不执行安装")
    parser.add_argument("--yes", action="store_true", help="跳过确认提示")
    args = parser.parse_args()

    print("=" * 60)
    print("  知阁 Zhige 一键安装")
    print("=" * 60)

    # 检测环境
    step(1, 5, "检测运行环境")
    r = run(["python", "--version"])
    ver = (r.stdout.strip() or r.stderr.strip()).split("\n")[0]
    print(f"  Python: {ver}")

    # 检测 Ollama
    ollama_installed = check_ollama_installed()
    ollama_running = check_ollama_running() if ollama_installed else False
    if ollama_installed:
        print(f"  Ollama: 已安装 {'(运行中)' if ollama_running else '(未运行)'}")
    else:
        print("  Ollama: 未安装")

    # 检测 GPU
    step(2, 5, "检测 GPU")
    r = run([str(VENV_PY), "-c",
             "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'CPU_ONLY')"
             ] if VENV_PY.exists() else ["python", "-c", 
             "import sys; print('NEED_VENV' if sys.prefix == sys.base_prefix else 'IN_VENV')"])
    
    has_cuda = "CUDA_OK" in r.stdout if VENV_PY.exists() else False
    in_venv = "IN_VENV" in r.stdout if VENV_PY.exists() else (sys.prefix != sys.base_prefix)
    
    if has_cuda:
        r2 = run([str(VENV_PY), "-c", "import torch; print(torch.cuda.get_device_name(0))"] 
                 if VENV_PY.exists() else ["echo", "GPU"])
        print(f"  GPU: {r2.stdout.strip()}")
    elif not VENV_PY.exists() or not in_venv:
        print("  GPU: 未检测（需要先创建虚拟环境）")
    else:
        print("  GPU: 未检测到，将以 CPU 模式运行")

    if args.check:
        venv_python = VENV_PY if VENV_PY.exists() else None
        if venv_python is None:
            print("\n  [FAIL] 未找到项目虚拟环境，请运行 python scripts\\install.py")
            return 1
        deps_ok = run(
            [str(venv_python), "-c", "import fastapi, uvicorn, chromadb, yaml"],
            capture=True,
        ).returncode == 0
        print(f"\n  {'[OK]' if deps_ok else '[FAIL]'} 核心依赖{'齐备' if deps_ok else '不完整'}")
        return 0 if deps_ok else 1

    # 创建虚拟环境
    step(3, 5, "创建虚拟环境")
    venv_dir = ROOT / ".venv"
    if not venv_dir.exists():
        run([sys.executable, "-m", "venv", str(venv_dir)])
        print("  [OK] 虚拟环境已创建")
    else:
        print("  [OK] 虚拟环境已存在")
    VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
    if not VENV_PY.exists():
        VENV_PY = ROOT / ".venv" / "bin" / "python"
    if not VENV_PY.exists():
        print("  [FAIL] 虚拟环境创建失败")
        return 1
    print(f"  路径: {VENV_PY}")

    # 升级 pip
    step(4, 5, "升级 pip")
    run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip",
         "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
        check=True, capture=False)
    print("  [OK] pip 已升级")

    step(5, 5, "安装依赖")
    
    # 确定安装模式
    install_mode = "quick"  # 默认快速模式
    
    if args.full:
        install_mode = "full"
        req = ROOT / "requirements.txt"
    elif args.ollama:
        install_mode = "ollama"
        req = ROOT / "requirements-quick.txt"
    elif has_cuda:
        # 有 GPU 但没指定模式，询问用户
        install_mode = "full"
        req = ROOT / "requirements.txt"
        print("  [i] 检测到 GPU，自动选择完整安装模式")
    else:
        # 无 GPU 或无虚拟环境，默认快速模式
        install_mode = "quick"
        req = ROOT / "requirements-quick.txt"
        print("  [i] 自动选择快速安装模式（使用 Ollama，无需 GPU）")

    # 安装依赖
    print(f"  安装模式: {'快速(推荐)' if install_mode in ('quick', 'ollama') else '完整'}")
    print(f"  依赖文件: {req.name}")

    if install_mode in ('quick', 'ollama'):
        # 快速安装：核心服务 + 文档解析，不安装 torch / Streamlit / 重排器。
        extra_files = []
    else:
        # 完整安装
        extra_files = [str(ROOT / "requirements.txt")]

    if install_mode == "full":
        # Keep the local-inference toolchain aligned only for full installs.
        print("  - 对齐本地推理依赖版本...")
        pinned = [
            "numpy==1.26.4",
            "transformers==4.46.1",
            "huggingface-hub==0.27.1",
            "tokenizers==0.20.3",
            "sentence-transformers==3.3.1",
            "scikit-learn==1.5.2",
        ]
        run([str(VENV_PY), "-m", "pip", "install"] + pinned +
            ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
            check=False, capture=False)

    # 安装主依赖
    pkgs = []
    for req_file in [str(req)] + extra_files:
        if Path(req_file).exists():
            pkgs.extend(["-r", req_file])
    
    if pkgs:
        r = run([str(VENV_PY), "-m", "pip", "install"] + pkgs +
                ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
                capture=False)
        if r.returncode != 0:
            print("  [!] 国内镜像安装失败，改用 PyPI 官方源重试...")
            r = run([str(VENV_PY), "-m", "pip", "install"] + pkgs,
                    capture=False)
        if r.returncode != 0:
            print("  [FAIL] 依赖安装失败，请检查网络和上方 pip 日志。")
            return 1

    print("  [OK] 依赖安装完成")

    # 完成后提示
    print()
    print("=" * 60)
    print("  [完成] 安装完成！")
    print("=" * 60)
    print()

    if install_mode in ('quick', 'ollama') and not ollama_installed:
        print("【下一步】安装 Ollama（推荐，无需 GPU）：")
        print()
        print("  1. 下载 Ollama: https://ollama.com/download")
        print("  2. 安装并启动 Ollama")
        print("  3. 运行以下命令下载模型（约 3 GB）：")
        print(f"     ollama pull {RECOMMENDED_OLLAMA_MODEL}")
        print("     ollama pull nomic-embed-text")
        print("  4. 启动知阁：")
        print("     python scripts\\quickstart.py")
        print()
        print("  或者，你也可以使用云端 API（完全不需要安装任何东西）：")
        print("  - DeepSeek: https://platform.deepseek.com/")
        print("  - 阿里云通义: https://dashscope.console.aliyun.com/")
        print("  设置环境变量后直接运行 python scripts\\quickstart.py")
    elif install_mode in ('quick', 'ollama') and not ollama_running:
        print("【下一步】启动 Ollama 并下载模型：")
        print()
        print("  1. 启动 Ollama:")
        print("     ollama serve")
        print()
        print("  2. 下载模型（约 3 GB）：")
        print(f"     ollama pull {RECOMMENDED_OLLAMA_MODEL}")
        print("     ollama pull nomic-embed-text")
        print()
        print("  3. 启动知阁：")
        print("     python scripts\\quickstart.py")
    elif install_mode in ('quick', 'ollama') and ollama_running:
        print("【下一步】Ollama 已就绪！")
        print()
        print("  1. 确保模型已下载（如果没有，运行）：")
        print(f"     ollama pull {RECOMMENDED_OLLAMA_MODEL}")
        print("     ollama pull nomic-embed-text")
        print()
        print("  2. 启动知阁：")
        print("     python scripts\\quickstart.py")
    else:
        print("【下一步】启动知阁：")
        print()
        print("  python scripts\\quickstart.py")
    
    print()
    if not args.yes:
        try:
            input("按 Enter 键退出 ... ")
        except (EOFError, KeyboardInterrupt):
            pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消。")
        sys.exit(1)
