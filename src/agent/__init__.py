"""Agent / 工具调用模块。

提供：
- :class:`Tool`: 工具抽象基类
- :class:`ToolRegistry`: 工具注册中心
- :class:`ReActAgent`: ReAct 风格 Agent 主类
- :data:`BuiltinTools`: 一组开箱即用的工具集合
"""

from __future__ import annotations

from .tool import Tool, ToolRegistry, ToolResult
from .builtin_tools import BuiltinTools
from .react_agent import ReActAgent

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "BuiltinTools",
    "ReActAgent",
]
