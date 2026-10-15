@echo off
chcp 65001 >nul
title 知阁 Zhige - 一键启动

echo.
echo  ╔════════════════════════════════════════════════════════════╗
echo  ║           知阁 Zhige - 开箱即用的本地知识库              ║
echo  ╚════════════════════════════════════════════════════════════╝
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查虚拟环境
if exist ".venv\Scripts\python.exe" (
    echo [OK] 检测到虚拟环境
    set PYTHON=.venv\Scripts\python.exe
) else (
    echo [提示] 未检测到虚拟环境，将使用系统 Python
    set PYTHON=python
)

REM 检查依赖
echo.
echo [检查] 核心依赖...
%PYTHON% -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [警告] 依赖未安装，是否自动安装？
    echo.
    echo   1) 快速安装（推荐，使用 Ollama，不需要 GPU）
    echo   2) 完整安装（包含 torch + 本地模型）
    echo   3) 退出
    echo.
    choice /c 123 /n /m "请选择 [1-3]: "
    
    if errorlevel 3 exit /b 0
    if errorlevel 2 (
        echo 正在安装完整依赖...
        %PYTHON% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    ) else (
        echo 正在安装快速依赖...
        %PYTHON% -m pip install -r requirements-core.txt -r requirements-optional.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    )
)

REM 检查 Ollama
echo.
echo [检查] Ollama 状态...
curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo [提示] Ollama 未运行
    echo.
    echo   推荐安装 Ollama（无需 GPU，完全免费）：
    echo   - 下载地址：https://ollama.com/download
    echo   - 模型命令：ollama pull qwen2.5:7b
    echo   - 模型命令：ollama pull nomic-embed-text
    echo.
    echo   或者使用云端 API（需要 API Key）：
    echo   - 设置环境变量 OPENAI_API_KEY=sk-your-key
    echo.
    echo   按任意键继续启动...
    pause >nul
)

REM 启动服务
echo.
echo [启动] 知阁服务...
echo.
echo   访问地址：http://localhost:8000
echo   API 文档：http://localhost:8000/docs
echo.

REM 使用后台启动并打开浏览器
start "" "http://localhost:8000"
%PYTHON% -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

pause
