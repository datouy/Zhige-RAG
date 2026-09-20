#!/usr/bin/env bash
# ============================================================
#   ChineseRAGKB 一键启动 (Linux / macOS)
#   用法： bash scripts/start_all.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==================================================="
echo "  ChineseRAGKB 一键启动"
echo "==================================================="
echo ""

# ---------- 检查虚拟环境 ----------
if [ ! -f ".venv/bin/activate" ]; then
    echo "[X] 未找到虚拟环境，请先运行 bash scripts/install.sh"
    exit 1
fi

source .venv/bin/activate

# ---------- HuggingFace 镜像 ----------
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p logs

# ---------- 启动后端 ----------
echo "[1/3] 启动后端 API (端口 8000) ..."
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000 \
    > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "      后端 PID: $BACKEND_PID"

sleep 5

# ---------- 启动前端 ----------
echo "[2/3] 启动前端 UI (端口 8501) ..."
python -m streamlit run ui/app.py --server.port 8501 \
    > logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "      前端 PID: $FRONTEND_PID"

# ---------- LLM 预加载 ----------
echo "[3/3] 后台预加载 LLM 模型 ..."
python scripts/preload_llm.py > logs/preload.log 2>&1 &
PRELOAD_PID=$!
echo "      预加载 PID: $PRELOAD_PID"

echo ""
echo "==================================================="
echo "  [完成] 全部启动成功！"
echo "==================================================="
echo ""
echo "  后端 API: http://localhost:8000"
echo "  API 文档: http://localhost:8000/docs"
echo "  前端 UI : http://localhost:8501"
echo ""
echo "日志："
echo "  tail -f logs/backend.log"
echo "  tail -f logs/frontend.log"
echo "  tail -f logs/preload.log"
echo ""

# 保存 PID 方便停止
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid
echo $PRELOAD_PID > .preload.pid

# 自动打开浏览器（Linux 桌面）
if command -v xdg-open &>/dev/null; then
    sleep 3
    xdg-open http://localhost:8501 &>/dev/null || true
elif command -v open &>/dev/null; then
    sleep 3
    open http://localhost:8501 &>/dev/null || true
fi

# 等待用户关闭
echo "按 Ctrl+C 关闭所有服务 ..."
trap "echo '正在关闭 ...'; kill $BACKEND_PID $FRONTEND_PID $PRELOAD_PID 2>/dev/null; rm -f .backend.pid .frontend.pid .preload.pid; exit 0" INT TERM

wait
