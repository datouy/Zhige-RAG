@echo off
chcp 65001 >nul

echo ===================================================
echo   ChineseRAGKB 一键停止
echo ===================================================
echo.

echo [1/3] 关闭窗口 ...
taskkill /FI "WINDOWTITLE: ChineseRAGKB-*" /T /F >nul 2>nul

echo [2/3] 结束 uvicorn 进程 ...
taskkill /IM "uvicorn.exe" /F >nul 2>nul

echo [3/3] 结束 streamlit 进程 ...
taskkill /IM "streamlit.exe" /F >nul 2>nul

REM 清掉端口占用（用 PowerShell 更稳）
powershell -NoProfile -Command "
$conns = Get-NetTCPConnection -LocalPort 8000,8501 -ErrorAction SilentlyContinue
foreach (\$c in \$conns) {
    try { Stop-Process -Id \$c.OwningProcess -Force -ErrorAction SilentlyContinue } catch {}
}
" >nul 2>nul

echo.
echo [OK] 所有 ChineseRAGKB 服务已停止。
echo.
pause
