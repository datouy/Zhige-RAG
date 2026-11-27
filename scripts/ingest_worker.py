"""Run the isolated durable ingestion worker.

Examples::

    INGEST_EXECUTION_MODE=worker python scripts/ingest_worker.py --once
    INGEST_EXECUTION_MODE=worker python scripts/ingest_worker.py --poll 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest_worker import run_forever, run_once  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="知阁隔离入库 worker")
    parser.add_argument("--once", action="store_true", help="最多处理一个任务后退出")
    parser.add_argument("--poll", type=float, default=2.0, help="空闲轮询间隔秒数")
    args = parser.parse_args()
    if args.once:
        print(json.dumps({"processed": bool(run_once())}, ensure_ascii=False))
        return 0
    run_forever(args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
