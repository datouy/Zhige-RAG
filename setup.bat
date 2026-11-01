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
echo │  1) 轻量安装（推荐：无 LLM，CPU ONNX，无需 GPU）        │
echo │     - 含 Web/API、文档解析、中文 BM25 与向量检索         │
echo │     - 不需要 Ollama / torch                              │
echo │                                                         │
echo │  2) 完整安装（增加 OCR 与可选界面）                      │
echo │     - 仍然不需要 LLM / torch                             │
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
.venv\Scripts\python.exe -m pip install -r requirements-quick.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
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

echo.
echo [模型] 准备本地 ONNX 向量模型...
.venv\Scripts\python.exe scripts\setup_onnx_embedding.py

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
echo   1) 启动知阁：
echo      双击 start.bat
echo      或运行：python scripts\quickstart.py
echo.
echo   默认回答模式为原文摘录，不加载 LLM。
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

REM 安装完整依赖
echo.
echo [安装] 安装完整依赖...
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-ocr.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo [OK] 完整依赖安装完成

echo.
echo [模型] 准备本地 ONNX 向量模型...
.venv\Scripts\python.exe scripts\setup_onnx_embedding.py

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
echo   2) 默认使用本地 ONNX 向量模型，不需要生成式 LLM
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
echo [检查] 本地 ONNX 向量模型...
if exist "models\bge-small-zh-v1.5-onnx\onnx\model_quantized.onnx" (
    echo [OK] 模型文件已就绪
) else (
    echo [提示] 模型文件缺失，运行 scripts\setup_onnx_embedding.py
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
