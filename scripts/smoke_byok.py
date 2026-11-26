#!/usr/bin/env python
"""BYOK 开箱即用冒烟测试（零外部依赖、零真实 Key）。

用一个进程内的 OpenAI 兼容 mock 服务验证“填 Key 即用”的模型接线：
  1. 环境变量 -> config 覆盖（LLM_BACKEND / OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY）
  2. create_llm(backend=openai) 的流式与非流式 chat
  3. create_embedding(backend=openai) 的 encode
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EMBED_DIM = 8
NL = chr(10)


class _MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 静默
        return

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _send_json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = self._read_body()
        if self.path.endswith("/chat/completions"):
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for token in ["你", "好", "！"]:
                    chunk = {"choices": [{"delta": {"content": token}}]}
                    line = f"data: {json.dumps(chunk, ensure_ascii=False)}{NL}{NL}"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(("data: [DONE]" + NL + NL).encode("utf-8"))
                self.wfile.flush()
            else:
                self._send_json({
                    "choices": [{"message": {"content": "你好，这是 mock 回答"}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
                })
        elif self.path.endswith("/embeddings"):
            n = len(body.get("input") or [])
            self._send_json({
                "data": [
                    {"index": i, "embedding": [0.1 * (i + 1)] * EMBED_DIM}
                    for i in range(n)
                ]
            })
        else:
            self._send_json({"error": "not found"})

    def do_GET(self):
        self._send_json({"ok": True})


def _start_mock():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}/v1"


def main() -> int:
    from src.llm_provider import BACKEND_OPENAI, create_llm
    from src.utils import apply_env_overrides, load_config

    server, base = _start_mock()
    os.environ["LLM_BACKEND"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-mock-byok"
    os.environ["OPENAI_BASE_URL"] = base
    os.environ["OPENAI_MODEL"] = "mock-chat"

    cfg = apply_env_overrides(load_config(str(ROOT / "config" / "config.yaml")))
    llm_cfg = cfg["llm"]
    assert llm_cfg["backend"] == "openai", llm_cfg
    assert llm_cfg["openai"]["base_url"] == base, llm_cfg["openai"]
    assert llm_cfg["openai"]["model"] == "mock-chat", llm_cfg["openai"]

    llm = create_llm(llm_cfg)
    assert getattr(llm, "provider_label", "") == BACKEND_OPENAI

    full = llm.chat([{"role": "user", "content": "hi"}], stream=False)
    assert "mock 回答" in str(full), full

    parts = [str(t) for t in llm.chat([{"role": "user", "content": "hi"}], stream=True)]
    assert "".join(parts) == "你好！", parts

    from src.embeddings_provider import create_embedding

    emb = create_embedding({
        "backend": "openai",
        "openai": {"base_url": base, "model": "mock-embed", "api_key": "sk-mock-byok"},
    })
    vec = emb.encode(["你好", "世界"])
    assert vec.shape == (2, EMBED_DIM), vec.shape

    server.shutdown()
    print("BYOK smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
