#!/usr/bin/env python
"""上传一个文件并高频轮询，验证多 worker 是否共享同一任务状态。"""
from __future__ import annotations

import argparse
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _request_json(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 30,
) -> dict:
    headers = {"X-Forwarded-For": "198.51.100.240"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc


def _upload(base_url: str, path: Path, *, background: bool) -> dict:
    boundary = f"----zhige-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    chunks: list[bytes] = []
    for name, value in (
        ("background", "true" if background else "false"),
        ("version", "false"),
    ):
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode("ascii"))
    return _request_json(
        base_url.rstrip("/") + "/api/v1/ingest",
        method="POST",
        body=b"".join(chunks),
        content_type=f"multipart/form-data; boundary={boundary}",
        timeout=60,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18001")
    parser.add_argument("--file", required=True)
    parser.add_argument("--polls", type=int, default=60)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--output")
    args = parser.parse_args()

    path = Path(args.file).resolve()
    uploaded = _upload(args.base_url, path, background=True)
    job_id = str(uploaded.get("job_id") or "")
    if not job_id:
        raise RuntimeError(f"上传响应缺少 job_id: {uploaded}")

    observations: list[dict] = []
    for _ in range(max(1, args.polls)):
        job = _request_json(
            args.base_url.rstrip("/") + f"/api/v1/ingest/jobs/{job_id}",
            timeout=20,
        )
        observations.append(
            {
                "status": job.get("status"),
                "stage": job.get("stage"),
                "progress": job.get("progress"),
            }
        )
        if job.get("status") in {"done", "failed", "interrupted"}:
            # 终态后继续轮询，专门捕捉另一个 worker 的旧快照覆盖。
            pass
        time.sleep(max(0.0, args.interval))

    statuses = [item["status"] for item in observations]
    final_job = _request_json(
        args.base_url.rstrip("/") + f"/api/v1/ingest/jobs/{job_id}",
        timeout=20,
    )
    regression = "done" in statuses and any(
        status == "interrupted" for status in statuses[statuses.index("done") :]
    )
    result = {
        "job_id": job_id,
        "polls": len(observations),
        "status_counts": {
            status: statuses.count(status) for status in sorted(set(statuses))
        },
        "terminal_regression": regression,
        "final_status": final_job.get("status"),
        "final_stage": final_job.get("stage"),
        "final_error": final_job.get("error"),
        "result": final_job.get("result"),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 1 if regression or final_job.get("status") != "done" else 0


if __name__ == "__main__":
    raise SystemExit(main())
