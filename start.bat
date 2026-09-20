@echo off
set "ROOT=%~dp0"
if not exist "%ROOT%.venv\Scripts\python.exe" (
    echo [ERROR] 未找到虚拟环境 .venv，请先创建并安装依赖。
    echo   例如: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
"%ROOT%.venv\Scripts\python.exe" "%ROOT%scripts\start_all.py" %*
