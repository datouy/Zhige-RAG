"""Download the lightweight CPU embedding model used by LLM-free mode."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_REPO = "Xenova/bge-small-zh-v1.5"
FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "onnx/model_quantized.onnx",
)


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    print(f"下载 {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Zhige-RAG/0.3"})
    with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as out:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)
    temp.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default=os.getenv(
            "ONNX_EMBEDDING_DIR", str(ROOT / "models/bge-small-zh-v1.5-onnx")
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/"),
    )
    args = parser.parse_args()
    target = Path(args.target).resolve()

    for relative in FILES:
        destination = target / relative
        if destination.exists() and destination.stat().st_size > 0:
            print(f"已存在 {destination}")
            continue
        download(f"{args.endpoint}/{MODEL_REPO}/resolve/main/{relative}", destination)

    total = sum((target / item).stat().st_size for item in FILES)
    print(f"完成：{target}（{total / 1024 / 1024:.1f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
