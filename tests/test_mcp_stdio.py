"""最小 MCP stdio 协议、命令白名单与工具调用。"""
from __future__ import annotations

import sys
import textwrap

from fastapi.testclient import TestClient
import pytest


@pytest.fixture()
def client():
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def mcp_server_script(tmp_path):
    script = tmp_path / "fake_mcp.py"
    script.write_text(
        textwrap.dedent(
            """
            import json
            import sys

            for line in sys.stdin:
                message = json.loads(line)
                if "id" not in message:
                    continue
                method = message.get("method")
                if method == "initialize":
                    result = {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake", "version": "1.0"},
                    }
                elif method == "tools/list":
                    result = {
                        "tools": [{
                            "name": "echo",
                            "description": "回显输入",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        }]
                    }
                elif method == "tools/call":
                    arguments = message.get("params", {}).get("arguments", {})
                    result = {
                        "content": [{"type": "text", "text": json.dumps(arguments, ensure_ascii=False)}],
                        "isError": False,
                    }
                else:
                    result = {}
                print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}, ensure_ascii=False), flush=True)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return script


def test_mcp_stdio_discovery_and_call(mcp_server_script):
    from src.mcp_stdio import MCPTool, discover_mcp_tools

    tools = discover_mcp_tools(sys.executable, [str(mcp_server_script)], timeout=5)
    assert tools[0]["name"] == "echo"
    assert tools[0]["description"] == "回显输入"
    tool = MCPTool(
        server_id="12345678-abcd",
        server_name="本地",
        remote_name="echo",
        description=tools[0]["description"],
        parameters=tools[0]["inputSchema"],
        command=sys.executable,
        args=[str(mcp_server_script)],
        timeout=5,
    )
    result = tool.safe_run(text="你好")
    assert result.success is True
    assert "你好" in result.content["content"][0]["text"]


def test_mcp_server_api_and_command_whitelist(client, mcp_server_script):
    created = client.post(
        "/api/v1/mcp/servers",
        json={"name": "本地回显", "command": sys.executable, "args": [str(mcp_server_script)]},
    )
    assert created.status_code == 201
    server = created.json()

    tested = client.post(f"/api/v1/mcp/servers/{server['id']}/test")
    assert tested.status_code == 200
    assert tested.json()["tools"][0]["remote_name"] == "echo"
    assert tested.json()["server"]["tool_count"] == 1

    rejected = client.post(
        "/api/v1/mcp/servers",
        json={"name": "shell", "command": "powershell", "args": []},
    )
    assert rejected.status_code == 400


def test_mcp_server_rejects_unknown_id(client):
    assert client.post("/api/v1/mcp/servers/not-found/test").status_code == 404
