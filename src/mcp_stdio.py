"""最小 MCP stdio 客户端与 Agent 工具适配。

只实现保存 Agent 所需的三步：``initialize``、``tools/list``、``tools/call``。
服务通过参数数组启动，不经过 shell；每次操作使用独立短进程，避免后台子进程
泄漏，也避免把不同用户的连接句柄混用。
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from src.agent.tool import Tool, ToolResult


class MCPError(RuntimeError):
    """MCP 配置、协议或超时错误。"""


def _command_name(value: str) -> str:
    return Path(value).stem.lower()


def _allowed_commands() -> set[str]:
    raw = os.getenv("MCP_ALLOWED_COMMANDS", "python,python3,node,npx,uvx")
    return {_command_name(item.strip()) for item in raw.split(",") if item.strip()}


def resolve_mcp_command(command: str) -> str:
    """按白名单解析可执行文件，拒绝 shell 字符串和未授权命令。"""
    raw = (command or "").strip()
    if not raw or any(char in raw for char in "\r\n|;&><"):
        raise MCPError("MCP 命令必须是单个可执行文件")
    if _command_name(raw) not in _allowed_commands():
        raise MCPError(f"MCP 命令 `{Path(raw).name}` 不在允许列表中")
    resolved = shutil.which(raw)
    if not resolved:
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            resolved = str(candidate.resolve())
    if not resolved:
        raise MCPError(f"找不到 MCP 命令：{raw}")
    return resolved


def mcp_tool_name(server_id: str, remote_name: str) -> str:
    """生成稳定、符合函数工具命名约束的本地工具名。"""
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", remote_name).strip("_")[:40]
    return f"mcp_{server_id[:8]}_{clean or 'tool'}"


class MCPStdioSession:
    """一次 MCP stdio 会话，消息采用逐行 JSON-RPC 2.0。"""

    def __init__(self, command: str, args: List[str], timeout: float = 20.0) -> None:
        self.command = resolve_mcp_command(command)
        self.args = [str(item) for item in args]
        self.timeout = max(1.0, min(float(timeout), 120.0))
        self.process: Optional[subprocess.Popen[str]] = None
        self._messages: queue.Queue[Any] = queue.Queue()
        self._stderr: Deque[str] = deque(maxlen=40)
        self._next_id = 0

    def __enter__(self) -> "MCPStdioSession":
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        try:
            self.process = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
                creationflags=flags,
                env=child_env,
            )
        except OSError as exc:
            raise MCPError(f"无法启动 MCP 服务：{exc}") from exc
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.5)

    def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            self._messages.put(None)
            return
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._messages.put(json.loads(line))
                except json.JSONDecodeError:
                    self._messages.put(MCPError(f"MCP 返回了非 JSON 内容：{line[:160]}"))
        finally:
            self._messages.put(None)

    def _read_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr.append(line.rstrip()[:500])

    def _send(self, payload: Dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            raise MCPError("MCP 服务未运行")
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._next_id += 1
        request_id = self._next_id
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = "\n".join(list(self._stderr)[-3:])
                raise MCPError(f"MCP {method} 超时" + (f"：{detail}" if detail else ""))
            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise MCPError(f"MCP {method} 超时") from exc
            if message is None:
                detail = "\n".join(list(self._stderr)[-3:])
                raise MCPError("MCP 服务提前退出" + (f"：{detail}" if detail else ""))
            if isinstance(message, Exception):
                raise message
            if not isinstance(message, dict) or message.get("id") != request_id:
                continue
            error = message.get("error")
            if error:
                raise MCPError(
                    str(error.get("message") if isinstance(error, dict) else error)
                )
            return message.get("result")

    def initialize(self) -> Dict[str, Any]:
        result = self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "zhige", "version": "1.0"},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result if isinstance(result, dict) else {}

    def list_tools(self) -> List[Dict[str, Any]]:
        result = self._request("tools/list", {})
        tools = result.get("tools") if isinstance(result, dict) else None
        return [item for item in (tools or []) if isinstance(item, dict) and item.get("name")]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = self._request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )
        return result if isinstance(result, dict) else {"content": result}


def discover_mcp_tools(command: str, args: List[str], timeout: float = 20.0) -> List[Dict[str, Any]]:
    with MCPStdioSession(command, args, timeout=timeout) as session:
        return session.list_tools()


class MCPTool(Tool):
    """把一个远端 MCP 工具包装成现有 ReAct Agent 可调用的工具。"""

    def __init__(
        self,
        server_id: str,
        server_name: str,
        remote_name: str,
        description: str,
        parameters: Dict[str, Any],
        command: str,
        args: List[str],
        timeout: float = 20.0,
    ) -> None:
        self.name = mcp_tool_name(server_id, remote_name)
        self.description = f"{server_name}：{description or remote_name}"
        self.parameters = parameters if isinstance(parameters, dict) else {
            "type": "object",
            "properties": {},
        }
        self.server_id = server_id
        self.server_name = server_name
        self.remote_name = remote_name
        self.command = command
        self.args = args
        self.timeout = timeout

    def run(self, **kwargs) -> ToolResult:
        try:
            with MCPStdioSession(self.command, self.args, timeout=self.timeout) as session:
                payload = session.call_tool(self.remote_name, kwargs)
        except MCPError as exc:
            return ToolResult(success=False, content=None, error=str(exc))
        failed = bool(payload.get("isError"))
        return ToolResult(
            success=not failed,
            content=payload,
            error=_mcp_error_text(payload) if failed else None,
            metadata={"tool": self.name, "server": self.server_id},
        )


def _mcp_error_text(payload: Dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, list):
        text = " ".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
        if text:
            return text
    return "MCP 工具执行失败"


__all__ = [
    "MCPError",
    "MCPStdioSession",
    "MCPTool",
    "discover_mcp_tools",
    "mcp_tool_name",
    "resolve_mcp_command",
]
