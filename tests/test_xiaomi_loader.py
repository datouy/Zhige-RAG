"""Xiaomi-OCR-0 / Any2Markdown 文档解析接入测试。

覆盖两部分：
1. ``_load_pdf`` 的回退分支逻辑（mock，不依赖外部服务）；
2. ``_load_pdf_via_xiaomi`` 的 multipart 上传 + 响应解析（本地 mock HTTP 服务）。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest

from src.document_loader import Document, _load_pdf, _load_pdf_via_xiaomi


def _pdf(tmp_path: Path) -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake pdf bytes")
    return p


# ----------------------------------------------------------------------
#  1. _load_pdf 的回退分支逻辑（mock）
# ----------------------------------------------------------------------
def test_load_pdf_xiaomi_enabled_returns_xiaomi_result(tmp_path):
    pdf = _pdf(tmp_path)
    fake = Document(page=1, content="# 标题\n正文", metadata={"source": "doc.pdf"})
    with mock.patch("src.document_loader._load_pdf_via_xiaomi", return_value=[fake]) as m, \
         mock.patch("src.document_loader._load_pdf_with_engine") as eng, \
         mock.patch("src.document_loader._augment_pdf_with_ocr", side_effect=lambda p, d, c: d):
        docs = _load_pdf(pdf, xiaomi_cfg={"enabled": True})
    assert docs == [fake]
    m.assert_called_once()
    eng.assert_not_called()


def test_load_pdf_xiaomi_exception_falls_back(tmp_path):
    pdf = _pdf(tmp_path)
    with mock.patch("src.document_loader._load_pdf_via_xiaomi",
                    side_effect=RuntimeError("down")) as m, \
         mock.patch("src.document_loader._load_pdf_with_engine", return_value=[]) as eng, \
         mock.patch("src.document_loader._augment_pdf_with_ocr", side_effect=lambda p, d, c: d):
        docs = _load_pdf(pdf, xiaomi_cfg={"enabled": True})
    assert docs == []
    m.assert_called_once()
    eng.assert_called()  # 异常后回退到传统引擎


def test_load_pdf_xiaomi_empty_no_fallback_returns_empty(tmp_path):
    pdf = _pdf(tmp_path)
    with mock.patch("src.document_loader._load_pdf_via_xiaomi", return_value=[]) as m, \
         mock.patch("src.document_loader._load_pdf_with_engine") as eng, \
         mock.patch("src.document_loader._augment_pdf_with_ocr", side_effect=lambda p, d, c: d):
        docs = _load_pdf(pdf, xiaomi_cfg={"enabled": True, "fallback_on_error": False})
    assert docs == []
    m.assert_called_once()
    eng.assert_not_called()


def test_load_pdf_xiaomi_disabled_skips(tmp_path):
    pdf = _pdf(tmp_path)
    with mock.patch("src.document_loader._load_pdf_via_xiaomi") as m, \
         mock.patch("src.document_loader._load_pdf_with_engine", return_value=[]) as eng, \
         mock.patch("src.document_loader._augment_pdf_with_ocr", side_effect=lambda p, d, c: d):
        _load_pdf(pdf, xiaomi_cfg={"enabled": False})
    m.assert_not_called()
    eng.assert_called()


# ----------------------------------------------------------------------
#  2. _load_pdf_via_xiaomi 的 multipart 上传 + 响应解析（本地 mock HTTP）
# ----------------------------------------------------------------------
class _Any2MDHandler(BaseHTTPRequestHandler):
    received: dict = {}

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler 约定方法名
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        type(self).received = {
            "path": self.path,
            "content_type": self.headers.get("Content-Type", ""),
            "has_file": b'name="file"' in body and b"%PDF" in body,
            "has_parse_mode": b'name="parse_mode"' in body and b"ocr" in body,
            "has_model": b'name="model"' in body and b"xiaomi-ocr-0" in body,
        }
        payload = json.dumps({"markdown": "# 标题\n\n正文内容"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # 静默请求日志
        pass


@pytest.fixture()
def any2md_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Any2MDHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_load_pdf_via_xiaomi_calls_any2md_and_parses_markdown(tmp_path, any2md_server):
    pdf = _pdf(tmp_path)
    docs = _load_pdf_via_xiaomi(
        pdf,
        {"base_url": any2md_server, "parse_mode": "ocr", "model": "xiaomi-ocr-0"},
    )
    assert len(docs) == 1
    assert "# 标题" in docs[0].content
    assert "正文内容" in docs[0].content
    assert docs[0].metadata.get("xiaomi_engine") is True
    assert docs[0].metadata.get("xiaomi_model") == "xiaomi-ocr-0"
    # 确认 multipart 真的把文件字节与 parse_mode/model 传上去了
    assert _Any2MDHandler.received["path"] == "/convert/file"
    assert _Any2MDHandler.received["has_file"] is True
    assert _Any2MDHandler.received["has_parse_mode"] is True
    assert _Any2MDHandler.received["has_model"] is True


def test_load_pdf_via_xiaomi_plain_text_response(tmp_path, any2md_server):
    """Any2Markdown 若返回纯 Markdown 文本（非 JSON），也应被正确解析。"""
    original = _Any2MDHandler.do_POST

    def _plain(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        payload = "纯文本 Markdown 响应".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    _Any2MDHandler.do_POST = _plain
    try:
        docs = _load_pdf_via_xiaomi(_pdf(tmp_path), {"base_url": any2md_server})
    finally:
        _Any2MDHandler.do_POST = original
    assert len(docs) == 1
    assert "纯文本 Markdown 响应" in docs[0].content
