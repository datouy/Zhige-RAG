#!/usr/bin/env bash
# ============================================================
#   ChineseRAGKB 一键安装脚本 (Linux / macOS)
#   用法： bash scripts/install.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==================================================="
echo "  ChineseRAGKB 一键安装脚本 (Linux/macOS)"
echo "==================================================="
echo ""

# ---------- 1. Python ----------
echo "[1/7] 检查 Python 环境..."
if ! command -v python3 &>/dev/null; then
    echo "[X] 未检测到 python3，请先安装 Python 3.10-3.12"
    exit 1
fi
PYVER=$(python3 --version | awk '{print $2}')
echo "[OK] Python 版本: $PYVER"
echo ""

# ---------- 2. venv ----------
echo "[2/7] 创建虚拟环境 .venv ..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "[OK] 虚拟环境已创建"
else
    echo "[OK] 虚拟环境已存在，跳过创建"
fi
echo ""

# ---------- 3. activate ----------
echo "[3/7] 激活虚拟环境..."
source .venv/bin/activate
echo "[OK] 虚拟环境已激活"
echo ""

# ---------- 4. pip ----------
echo "[4/7] 升级 pip ..."
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ""

# ---------- 5. CUDA ----------
echo "[5/7] 检测 GPU / CUDA ..."
HAS_CUDA=0
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[OK] 检测到 CUDA 加速"
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "      设备: $GPU_NAME"
    HAS_CUDA=1
else
    echo "[!] 未检测到 CUDA，将以 CPU 模式运行"
fi
echo ""

# ---------- 6. 依赖 ----------
echo "[6/7] 安装依赖 ..."
if [ "$HAS_CUDA" = "1" ]; then
    pip install torch --index-url https://download.pytorch.org/whl/cu118 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple || {
        echo "[!] CUDA 版 PyTorch 安装失败，回退到 CPU 版"
        pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
    }
fi
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "[OK] 依赖安装完成"
echo ""

# ---------- 7. 模型 ----------
echo "[7/7] 下载默认模型（首次较慢）..."
export HF_ENDPOINT=https://hf-mirror.com

if ! command -v huggingface-cli &>/dev/null; then
    pip install -U huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

huggingface-cli download BAAI/bge-small-zh-v1.5 || echo "[!] Embedding 下载失败，可稍后重试"
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct || echo "[!] LLM 下载失败，可稍后重试"
echo ""

echo "==================================================="
echo "  [完成] 安装完成！"
echo "==================================================="
echo ""
echo "接下来可以："
echo "  - bash scripts/start_all.sh  同时启动前后端"
echo ""
