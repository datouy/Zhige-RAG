"""命令行查询脚本：直接通过 RAG 流程回答问题。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag_pipeline import RAGPipeline
from src.utils import apply_env_overrides, load_config


def main():
    parser = argparse.ArgumentParser(description="RAG 命令行问答")
    parser.add_argument("question", help="用户问题")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_env_overrides(cfg)

    pipeline = RAGPipeline.from_config(args.config, overrides=cfg)
    if args.no_stream:
        result = pipeline.answer(args.question, top_k=args.top_k, stream=False)
        print("\n=== 回答 ===\n", result.answer)
        if result.sources:
            print("\n=== 来源 ===")
            for s in result.sources:
                print(s["cite"])
    else:
        print("=== 流式输出 ===")
        for ev in pipeline.stream_answer(args.question, top_k=args.top_k):
            ev_type = ev.get("event")
            data = ev.get("data")
            if ev_type == "token":
                print(data, end="", flush=True)
            elif ev_type == "hits":
                pass  # 来源信息在 done 时打印
            elif ev_type == "done":
                print("\n\n=== 来源 ===")
                for s in data or []:
                    print(s["cite"])
            elif ev_type == "error":
                print(f"\n[错误] {data}")


if __name__ == "__main__":
    main()