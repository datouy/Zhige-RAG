"""ChineseRAGKB - 一键安装脚本 (Windows)
纯 Python 实现，兼容 CMD / PowerShell / Git Bash / VS Code 终端
用法: python scripts/install.py
"""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = ROOT / ".venv" / "bin" / "python"
VENV_PIP = VENV_PY


def run(cmd: list[str], check: bool = False, env: dict | None = None) -> subprocess.CompletedProcess:
    merged = dict(os.environ, **(env or {}))
    return subprocess.run(cmd, env=merged, check=check,
                         text=True, capture_output=True)


def step(n: int, total: int, msg: str) -> None:
    print(f"\n[{n}/{total}] {msg}")


def main() -> int:
    print("=" * 56)
    print("  知阁 Zhige 一键安装 (Windows)")
    print("=" * 56)

    # 1. Python
    step(1, 7, "检查 Python 环境")
    r = run(["python", "--version"])
    ver = (r.stdout.strip() or r.stderr.strip()).split("\n")[0]
    print(f"  Python: {ver}")

    # 2. venv
    step(2, 7, "创建虚拟环境 .venv")
    venv_dir = ROOT / ".venv"
    if not venv_dir.exists():
        run([sys.executable, "-m", "venv", str(venv_dir)])
        print("  [OK] 虚拟环境已创建")
    else:
        print("  [OK] 虚拟环境已存在")
    print(f"  路径: {VENV_PY}")

    # 3. 升级 pip
    step(3, 7, "升级 pip")
    run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip",
         "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=True)
    print("  [OK] pip 已升级")

    # 4. 检测 CUDA
    step(4, 7, "检测 GPU / CUDA")
    r = run([str(VENV_PY), "-c",
             "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'CPU_ONLY')"])
    has_cuda = r.stdout.strip() == "CUDA_OK"
    if has_cuda:
        r2 = run([str(VENV_PY), "-c", "import torch; print(torch.cuda.get_device_name(0))"])
        print(f"  [OK] GPU: {r2.stdout.strip()}")
    else:
        print("  [!] 未检测到 CUDA，将以 CPU 模式运行")

    # 5. 安装依赖
    step(5, 7, "安装依赖 (PyTorch + requirements.txt)")
    req = ROOT / "requirements.txt"

    torch_extra: list[str] = []
    if has_cuda:
        print("  - 安装 PyTorch CUDA 11.8 版本 ...")
        r = run([str(VENV_PY), "-m", "pip", "install", "torch",
                 "--index-url", "https://download.pytorch.org/whl/cu118",
                 "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
        if r.returncode != 0:
            print("  [!] CUDA 版 PyTorch 安装失败，回退 CPU 版")
        else:
            torch_extra = ["torch"]
            print("  [OK] PyTorch CUDA 版安装成功")

    # ----- 关键：先把核心三方库对齐到稳定版本组合 -----
    # 解决 chromadb / numpy / transformers / huggingface-hub 之间的版本冲突
    print("  - 对齐核心三方库版本 (numpy/transformers/chromadb) ...")
    pinned = [
        "numpy==1.26.4",          # chromadb 0.4.x 唯一兼容的 numpy
        "transformers==4.46.1",   # 最后一个稳定 4.x
        "huggingface-hub==0.27.1",  # 与 transformers 4.46 匹配
        "tokenizers==0.20.3",
        "sentence-transformers==3.3.1",
        "scikit-learn==1.5.2",
    ]
    r = run([str(VENV_PY), "-m", "pip", "install"] + pinned +
            ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=False)
    if r.returncode != 0:
        print("  [!] 核心库对齐失败（可忽略，后续会重试）")

    print("  - 安装 requirements.txt ...")
    pkgs = torch_extra + ["-r", str(req)]
    r = run([str(VENV_PY), "-m", "pip", "install"] + pkgs +
            ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
    if r.returncode != 0:
        # ----- chroma-hnswlib 特殊处理 -----
        # Windows 上 chroma-hnswlib 新版没有预编译 wheel，会触发
        # "Microsoft Visual C++ 14.0 or greater is required"。
        # 把 hnswlib 固定到带 wheel 的最后一个版本 0.7.5。
        print("  [!] 依赖安装失败，尝试修复 chromadb 兼容问题 ...")
        if sys.platform == "win32":
            run([str(VENV_PY), "-m", "pip", "uninstall", "-y",
                 "chroma-hnswlib", "chromadb"], check=False)
            ok1 = run([str(VENV_PY), "-m", "pip", "install",
                       "chroma-hnswlib==0.7.5", "--only-binary=:all:",
                       "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
                      check=False).returncode == 0
            if ok1:
                ok2 = run([str(VENV_PY), "-m", "pip", "install",
                           "chromadb>=0.4.24,<0.6.0", "--no-deps",
                           "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
                          check=False).returncode == 0
                if ok2:
                    # 再装除 chromadb 外的其他依赖
                    other_pkgs = []
                    if torch_extra:
                        other_pkgs.extend(torch_extra)
                    other_pkgs.extend(["-r", str(req)])
                    # 拆掉 chromadb 行（避免再次依赖失败）
                    try:
                        lines = req.read_text(encoding="utf-8").splitlines()
                        lines = [l for l in lines
                                 if not l.strip().startswith("chromadb")
                                 and not l.strip().startswith("chroma-hnswlib")]
                        tmp_req = ROOT / "logs" / "requirements_no_chroma.txt"
                        tmp_req.parent.mkdir(exist_ok=True)
                        tmp_req.write_text("\n".join(lines),
                                           encoding="utf-8")
                        other_pkgs = [l for l in other_pkgs
                                      if l != str(req)] + [str(tmp_req)]
                    except OSError:
                        pass
                    run([str(VENV_PY), "-m", "pip", "install"] + other_pkgs +
                        ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
                        check=False)
                    print("  [OK] chromadb 已修复，继续安装 ...")
                else:
                    print("  [X] chromadb 安装仍失败，请参见 README FAQ Q1 安装 Visual C++ Build Tools")
                    return 1
            else:
                print("  [X] chroma-hnswlib 0.7.5 安装失败，请参见 README FAQ Q1 安装 Visual C++ Build Tools")
                return 1
        else:
            print("  [X] 依赖安装失败，请手动排查")
            return 1
    print("  [OK] 依赖安装完成")

    # 6. huggingface_hub
    step(6, 7, "安装 huggingface_hub")
    run([str(VENV_PY), "-m", "pip", "install", "-U", "huggingface_hub",
         "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=True)
    print("  [OK] huggingface_hub 已安装")

    # 7. 下载模型
    step(7, 7, "下载默认模型 (首次较慢，请耐心等待)")
    models = [
        "BAAI/bge-small-zh-v1.5",
        "Qwen/Qwen2.5-1.5B-Instruct",
    ]
    for m in models:
        print(f"  - {m} ...", end=" ", flush=True)
        r = run([str(VENV_PY), "-m", "huggingface_cli", "download", m],
                env=dict(os.environ, HF_ENDPOINT="https://hf-mirror.com"))
        if r.returncode == 0:
            print("OK")
        else:
            print("失败 (网络问题可稍后重试)")

    print()
    print("=" * 56)
    print("  [完成] 安装完成！")
    print("=" * 56)
    print()
    print("下一步：")
    print("  python scripts\\start_all.py")
    print()
    input("按 Enter 键退出 ... ")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消。")
        sys.exit(1)
