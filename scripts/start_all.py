"""ChineseRAGKB - 一键启动脚本 (Windows)
同时启动后端 FastAPI + 前端 Streamlit + LLM 预加载
用法: python scripts/start_all.py
"""
from __future__ import annotations

import subprocess
import sys
import os
import time
import socket
import webbrowser
import threading
from pathlib import Path

# 强制 stdout 立即刷新（避免在后台运行时输出堆积在 buffer）
import functools
print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = ROOT / ".venv" / "bin" / "python"
LOG_DIR = ROOT / "logs"


def run(cmd: list[str], env: dict | None = None, **kwargs) -> subprocess.Popen:
    merged = dict(os.environ, **(env or {}))
    merged["HF_ENDPOINT"] = "https://hf-mirror.com"
    return subprocess.Popen(cmd, env=merged, **kwargs)


def wait_port(host: str, port: int, timeout: int = 40) -> bool:
    """等待端口变为可用，返回 True 表示服务就绪。"""
    import socket as _socket

    deadline = time.time() + timeout
    last_err = None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((host, port))
            s.close()
            return True
        except OSError as exc:
            last_err = exc
            if attempt <= 3 or attempt % 10 == 0:
                print(f"     [wait {host}:{port}] attempt {attempt}: {exc}", flush=True)
            time.sleep(1)
    if last_err:
        print(f"     (last error: {last_err})", flush=True)
    return False


def get_local_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def kill_port(port: int) -> None:
    """尝试杀掉占用端口的进程（Windows）。"""
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | "
         f"ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"],
        capture_output=True
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="知阁 Zhige 一键启动")
    parser.add_argument(
        "--with-streamlit",
        action="store_true",
        help="额外启动 Streamlit 控制台（8501）。默认不启动 —— Web 界面已由 "
             "FastAPI 在 8000 端口直接提供，Streamlit 已降级为开发调试工具。",
    )
    args = parser.parse_args()

    print("=" * 56)
    print("  知阁 Zhige 一键启动")
    print("=" * 56)

    # 确认 venv
    if not VENV_PY.exists():
        print("[X] 未找到虚拟环境 .venv，请先运行:")
        print("    python scripts\\install.py")
        return 1

    LOG_DIR.mkdir(exist_ok=True)

    # 杀掉旧进程
    print("\n[0/4] 关闭残留旧服务 ...")
    for port in [8000, 8501]:
        kill_port(port)
    subprocess.run(["taskkill", "/F", "/IM", "uvicorn.exe"],
                   capture_output=True)
    subprocess.run(["taskkill", "/F", "/IM", "streamlit.exe"],
                   capture_output=True)
    time.sleep(2)

    backend_env = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com")
    frontend_env = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com",
                        STREAMLIT_SERVER_PORT="8501",
                        STREAMLIT_SERVER_ADDRESS="0.0.0.0",
                        STREAMLIT_SERVER_HEADLESS="true")

    # 1. 后端
    print("\n[1/4] 启动后端 FastAPI (端口 8000) ...")
    backend_log = LOG_DIR / "backend.log"
    backend_proc = run(
        [str(VENV_PY), "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", "8000"],
        env=backend_env,
        cwd=str(ROOT),
        stdout=open(backend_log, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    print("  等待后端就绪 ...", end=" ", flush=True)
    if wait_port("127.0.0.1", 8000):
        print("OK")
    else:
        print("超时！请查看 logs\\backend.log")

    # 2. 前端
    #    默认只跑 Web 界面 —— 它由 FastAPI 直接挂载在 8000 端口（ui/web），
    #    不需要额外进程。Streamlit 已降级为开发调试工具，需要时显式开启。
    frontend_proc = None
    if args.with_streamlit:
        print("\n[2/4] 启动 Streamlit 控制台 (端口 8501) ...")
        frontend_log = LOG_DIR / "frontend.log"
        frontend_proc = run(
            [str(VENV_PY), "-m", "streamlit", "run", "ui/app.py",
             "--server.port", "8501",
             "--server.address", "0.0.0.0",
             "--server.headless=true",
             "--browser.gatherUsageStats=false"],
            env=frontend_env,
            cwd=str(ROOT),
            stdout=open(frontend_log, "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        print("  等待前端就绪 ...", end=" ", flush=True)
        if wait_port("127.0.0.1", 8501, timeout=60):
            print("OK")
        else:
            print("超时！请查看 logs\\frontend.log")
    else:
        print("\n[2/4] Web 界面已随 FastAPI 提供 → http://localhost:8000")
        print("      跳过 Streamlit（需要调试界面时加 --with-streamlit）")

    # 3. LLM 预加载（可选：仅当 scripts/preload_llm.py 存在时执行；失败不影响主服务）
    preload_proc = None
    preload_script = ROOT / "scripts" / "preload_llm.py"
    if preload_script.exists():
        print("\n[3/4] 后台预加载 LLM 模型 (首次约 30-120 秒) ...")
        preload_log = LOG_DIR / "preload.log"
        try:
            preload_proc = run(
                [str(VENV_PY), str(preload_script)],
                env=dict(os.environ, HF_ENDPOINT="https://hf-mirror.com"),
                cwd=str(ROOT),
                stdout=open(preload_log, "w", encoding="utf-8"),
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] LLM 预加载启动失败，已跳过: {exc}")
            preload_proc = None
    else:
        print("\n[3/4] 跳过 LLM 预加载（未找到 scripts/preload_llm.py，"
              "首次问答时会自动懒加载）")

    # 4. 完成
    print("\n[4/4] 最终检查 ...")
    lan_ip = get_local_ip()

    print()
    print("=" * 56)
    print("  [完成] 全部启动成功！")
    print("=" * 56)
    print()
    print("  本机访问 (推荐):")
    print("    打开知阁 : http://localhost:8000")
    print("    API 文档 : http://localhost:8000/docs")
    if args.with_streamlit:
        print("    Streamlit: http://localhost:8501（调试用）")
    if lan_ip:
        print()
        print("  局域网访问 (同 WiFi / 内网):")
        print(f"    打开知阁 : http://{lan_ip}:8000")
        if args.with_streamlit:
            print(f"    Streamlit: http://{lan_ip}:8501")
    print()
    print("  日志文件:")
    print("    logs\\backend.log")
    if args.with_streamlit:
        print("    logs\\frontend.log")
    print("    logs\\preload.log")
    print()
    print("  关闭方式:")
    print("    关闭此窗口 = 停止所有服务")
    print()

    # 自动开浏览器
    def _open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8000")
    threading.Thread(target=_open_browser, daemon=True).start()

    print("  3 秒后自动打开浏览器 ...")
    print()

    try:
        print("  按 Ctrl+C 停止所有服务 ...")
        # 等待子进程。preload 是 best-effort，失败不影响主服务；
        # 只有 backend/frontend 才是关键服务。
        # frontend 可能未启动（默认不跑 Streamlit），必须过滤掉 None，
        # 否则下面的 p.poll() 会直接抛 AttributeError。
        critical = [p for p in (backend_proc, frontend_proc) if p is not None]
        while True:
            time.sleep(5)
            for p in critical:
                if p.poll() is not None:
                    name = "Backend" if p is backend_proc else "Frontend"
                    print(f"\n  [!] {name} 进程已退出")
                    raise SystemExit(1)
    except (KeyboardInterrupt, SystemExit) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("\n\n  正在停止所有服务 ...")

    stop_list = [("Backend", backend_proc), ("Frontend", frontend_proc)]
    if preload_proc is not None:
        stop_list.append(("Preload", preload_proc))
    for name, proc in stop_list:
        if proc.poll() is None:
            proc.terminate()
        print(f"  - {name} 已停止")

    print("\n  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
