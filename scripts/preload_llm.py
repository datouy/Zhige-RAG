"""LLM 预加载脚本：提前把模型加载到内存/显存，减少首次问答的等待。

用法：
    python scripts/preload_llm.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 把项目根目录加入 import 路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    print("=" * 60)
    print("LLM 预加载 - ChineseRAGKB")
    print("=" * 60)

    start_time = time.time()

    # 1. 读配置
    try:
        from src.utils import apply_env_overrides, load_config
    except ImportError as exc:
        print(f"[X] 导入失败: {exc}")
        print("    请确认已激活虚拟环境并安装依赖。")
        return 1

    try:
        config = load_config(str(ROOT / "config" / "config.yaml"))
        config = apply_env_overrides(config)
        print("[OK] 配置加载完成")
    except Exception as exc:
        print(f"[X] 读取配置失败: {exc}")
        return 1

    llm_cfg = config.get("llm", {})
    print(f"     模型: {llm_cfg.get('model_name', '?')}")
    quant = llm_cfg.get("quantization", {})
    if isinstance(quant, dict):
        print(f"     量化: {quant.get('enabled', False)}")
    print(f"     设备: {llm_cfg.get('device', 'auto')}")

    # 2. 构建 RAG 流水线（触发 LLM 加载）
    print()
    print("[*] 正在加载模型，首次约需 30-120 秒 ...")
    try:
        # P3.3: 单飞行锁 — 避免两个进程同时 load LLM 把显存打爆
        from src.llm_singleflight import (
            LlmLoadSkipped,
            acquire_llm_load_lock,
        )
        with acquire_llm_load_lock():
            from src.rag_pipeline import RAGPipeline

            rag = RAGPipeline.from_config(
                str(ROOT / "config" / "config.yaml"),
                overrides=config,
                lazy_llm=False,
            )
            elapsed = time.time() - start_time
            print(f"[OK] LLM 预加载完成，耗时 {elapsed:.1f}s")
    except LlmLoadSkipped:
        print("[SKIP] 另一进程正在加载 LLM，本进程 backoff 跳过")
        return 0
    except Exception as exc:
        print(f"[X] 加载失败: {exc}")
        return 1

    # 3. 心跳保持
    print()
    print("[*] 模型已就绪，保持后台运行（关闭此窗口即可释放显存）")
    print()

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n[*] 收到退出信号，正在卸载模型 ...")
        return 0


if __name__ == "__main__":
    sys.exit(main())