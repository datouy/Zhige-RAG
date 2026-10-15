@echo off
chcp 65001 >nul
title 知阁 Zhige - 首次安装向导

echo.
echo  ╔════════════════════════════════════════════════════════════╗
echo  ║        知阁 Zhige - 首次安装向导                          ║
echo  ╚════════════════════════════════════════════════════════════╝
echo.

REM 检查 Python 版本
python --version 2>nul
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo 检测到 Python 版本: %PYVER%
echo.

REM 解析主版本号
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

if %PYMAJOR% LSS 3 (
    echo [错误] Python 版本过低，需要 Python 3.10+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)
if %PYMAJOR% EQU 3 if %PYMINOR% LSS 10 (
    echo [错误] Python 版本过低，需要 Python 3.10+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python 版本符合要求
echo.

REM 选择安装模式
echo ┌─────────────────────────────────────────────────────────┐
echo │  请选择安装模式：                                        │
echo ├─────────────────────────────────────────────────────────┤
echo │                                                         │
echo │  1) 快速安装（推荐，使用 Ollama，无需 GPU）              │
echo │     - 约 500MB 依赖                                      │
echo │     - 需要安装 Ollama（约 200MB）                        │
echo │                                                         │
echo │  2) 完整安装（包含 torch + 本地模型）                   │
echo │     - 约 3-5GB 依赖                                      │
echo │     - 不需要 Ollama                                      │
echo │     - 需要 GPU 才能流畅运行                              │
echo │                                                         │
echo │  3) 仅检查环境                                          │
echo │                                                         │
echo │  4) 退出                                                │
echo │                                                         │
echo └─────────────────────────────────────────────────────────┘
echo.
choice /c 1234 /n /m "请选择 [1-4]: "

if errorlevel 4 exit /b 0
if errorlevel 3 goto :check_env
if errorlevel 2 goto :full_install
if errorlevel 1 goto :quick_install

:quick_install
echo.
echo [安装] 快速安装模式
echo.

REM 检查 Ollama
where ollama >nul 2>&1
if errorlevel 1 (
    echo [提示] Ollama 未安装
    echo.
    echo   Ollama 是推荐使用的模型运行方式：
    echo   - 无需 GPU，CPU 即可运行
    echo   - 完全免费开源
    echo   - 支持 qwen、llama 等多种模型
    echo.
    echo   下载地址：https://ollama.com/download
    echo.
    set OLLAMA_INSTALL=
    choice /c yn /n /m "是否现在跳过 Ollama 安装？[y/n]: "
    if errorlevel 2 goto :skip_ollama
    echo   请先安装 Ollama，然后重新运行本脚本
    pause
    exit /b 0
    :skip_ollama
) else (
    echo [OK] Ollama 已安装
    REM 检查 Ollama 是否运行
    curl -s http://localhost:11434/api/version >nul 2>&1
    if errorlevel 1 (
        echo [提示] Ollama 未运行，建议启动：ollama serve
    ) else (
        echo [OK] Ollama 服务正在运行
    )
)

REM 创建虚拟环境
echo.
echo [安装] 创建虚拟环境...
if exist ".venv" (
    echo [跳过] 虚拟环境已存在
) else (
    python -m venv .venv
    echo [OK] 虚拟环境已创建
)

REM 升级 pip
echo.
echo [安装] 升级 pip...
.venv\Scripts\python.exe -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>&1
echo [OK] pip 已升级

REM 安装核心依赖
echo.
echo [安装] 安装核心依赖（可能需要几分钟）...
.venv\Scripts\python.exe -m pip install -r requirements-core.txt -r requirements-optional.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [错误] 依赖安装失败
    echo.
    echo   常见问题：
    echo   - 网络问题：尝试使用 VPN 或更换镜像源
    echo   - 权限问题：以管理员身份运行
    echo.
    pause
    exit /b 1
)
echo [OK] 核心依赖安装完成

REM 复制 .env 文件
if not exist ".env" (
    echo.
    echo [配置] 创建环境配置文件...
    copy .env.example .env >nul 2>&1
    echo [OK] .env 文件已创建
)

echo.
echo ═════════════════════════════════════════════════════════════
echo.
echo   安装完成！
echo.
echo   下一步：
if not exist ".venv\Scripts\ollama.exe" (
    if not where ollama >nul 2>&1 (
        echo   1) 安装 Ollama: https://ollama.com/download
    )
)
echo   1) 下载模型（如果使用 Ollama）：
echo      ollama pull qwen2.5:7b
echo      ollama pull nomic-embed-text
echo.
echo   2) 启动知阁：
echo      双击 start.bat
echo      或运行：python scripts\quickstart.py
echo.
echo   或使用云端 API：
echo   - 编辑 .env 文件，填入 OPENAI_API_KEY
echo   - 设置 LLM_BACKEND=openai
echo.
echo ═════════════════════════════════════════════════════════════
pause
goto :end

:full_install
echo.
echo [安装] 完整安装模式
echo.

REM 创建虚拟环境
echo.
echo [安装] 创建虚拟环境...
if exist ".venv" (
    echo [跳过] 虚拟环境已存在
) else (
    python -m venv .venv
    echo [OK] 虚拟环境已创建
)

REM 升级 pip
echo.
echo [安装] 升级 pip...
.venv\Scripts\python.exe -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>&1
echo [OK] pip 已升级

REM 安装 torch
echo.
echo [安装] 安装 PyTorch（可能需要较长时间）...
.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
echo [OK] PyTorch 安装完成

REM 安装完整依赖
echo.
echo [安装] 安装完整依赖...
.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo [OK] 完整依赖安装完成

REM 复制 .env 文件
if not exist ".env" (
    echo.
    echo [配置] 创建环境配置文件...
    copy .env.example .env >nul 2>&1
    echo [OK] .env 文件已创建
)

echo.
echo ═════════════════════════════════════════════════════════════
echo.
echo   安装完成！
echo.
echo   下一步：
echo   1) 双击 start.bat 或运行：python scripts\quickstart.py
echo   2) 系统会自动下载所需的模型权重
echo.
echo ═════════════════════════════════════════════════════════════
pause
goto :end

:check_env
echo.
echo [检查] 环境检查
echo.

echo [检查] Python 版本...
echo [OK] %PYVER%

echo.
echo [检查] 虚拟环境...
if exist ".venv" (
    echo [OK] 虚拟环境存在
) else (
    echo [提示] 虚拟环境不存在
)

echo.
echo [检查] Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [提示] Ollama 未安装
) else (
    echo [OK] Ollama 已安装
    curl -s http://localhost:11434/api/version >nul 2>&1
    if errorlevel 1 (
        echo [提示] Ollama 未运行（运行 ollama serve 启动）
    ) else (
        echo [OK] Ollama 服务运行中
    )
)

echo.
echo [检查] 核心依赖...
.venv\Scripts\python.exe -c "import fastapi; import uvicorn; import chromadb; print('OK')" >nul 2>&1
if errorlevel 1 (
    echo [警告] 核心依赖可能不完整
    echo   运行 python scripts\doctor.py 查看详情
) else (
    echo [OK] 核心依赖完整
)

echo.
echo [完成] 环境检查结束
pause

:end
