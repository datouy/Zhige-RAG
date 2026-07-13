"""Tool 抽象基类与 ToolRegistry。

定义 Agent 可调用的「工具」最小契约：
- :class:`Tool`: 必须实现 ``run(**kwargs) -> ToolResult``
- :class:`ToolRegistry`: 注册、获取、列举工具
- :class:`ToolResult`: 统一返回值（含成功状态、错误信息、元数据）

任意 ``Tool`` 实现都应该：
1. 参数校验失败时返回 ``ToolResult(success=False, error="...")`` 而不是抛异常；
2. 在 ``parameters`` 字段中提供合法的 JSON Schema（含 ``type=object``）；
3. ``run()`` 内部尽量捕获异常，并改写成 ``ToolResult`` 返回。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """工具执行结果。

    Attributes:
        success: 是否成功执行。
        content: 工具返回的内容（字符串、列表或字典均可，Agent 内部会自动字符串化）。
        error: 错误信息（若 ``success=False``）。
        metadata: 附加元数据，例如耗时、检索命中等。
    """

    success: bool
    content: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - 简单的字符串化
        if self.success:
            return str(self.content) if self.content is not None else ""
        return f"[error] {self.error}"


class Tool(ABC):
    """工具抽象基类。

    子类必须设置以下类属性：
    - ``name``: 工具唯一名称（英文短串）。
    - ``description``: 工具的简短描述，供 LLM 选择调用。
    - ``parameters``: JSON Schema（``type=object``）。
    """

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """执行工具。参数由 Agent 通过 JSON Schema 反序列化后传入。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI function-calling 的 ``tools`` 列表元素。

        Returns:
            ``{"type": "function", "function": {...}}`` 格式的 dict。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }

    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """简单参数校验：检查 schema 中声明的 ``required`` 与字段类型。

        Args:
            params: 调用方传入的参数（已是 dict）。

        Returns:
            校验通过返回 ``None``；失败返回错误信息字符串。
        """
        if not isinstance(params, dict):
            return "参数必须是 JSON object / dict"
        schema = self.parameters or {}
        required = schema.get("required") or []
        for key in required:
            if key not in params:
                return f"缺少必填参数：{key}"
        # 字段类型宽松校验
        properties = schema.get("properties") or {}
        for key, value in params.items():
            prop_schema = properties.get(key)
            if not prop_schema:
                continue
            expected_type = prop_schema.get("type")
            if expected_type is None:
                continue
            py_type = _JSON_TYPE_MAP.get(expected_type)
            if py_type is None:
                continue
            # number / integer 在 python 中均视为 (int, float)，单独处理 integer
            if expected_type == "integer":
                if not isinstance(value, int) or isinstance(value, bool):
                    return f"参数 {key} 应为 integer"
            elif expected_type == "number":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return f"参数 {key} 应为 number"
            elif expected_type == "boolean":
                if not isinstance(value, bool):
                    return f"参数 {key} 应为 boolean"
            elif expected_type == "string":
                if not isinstance(value, str):
                    return f"参数 {key} 应为 string"
            elif expected_type == "array":
                if not isinstance(value, list):
                    return f"参数 {key} 应为 array"
            elif expected_type == "object":
                if not isinstance(value, dict):
                    return f"参数 {key} 应为 object"
        return None

    def safe_run(self, **kwargs) -> ToolResult:
        """带校验的执行入口：先校验参数，再执行；异常会被捕获并返回 ToolResult。

        Args:
            **kwargs: 调用方传入的 kwargs。

        Returns:
            :class:`ToolResult`。
        """
        try:
            err = self.validate_params(kwargs)
            if err is not None:
                return ToolResult(success=False, content=None, error=err)
            result = self.run(**kwargs)
            if not isinstance(result, ToolResult):
                # 防御性：用户实现里忘写 ToolResult，也允许直接返回字符串
                return ToolResult(success=True, content=result)
            return result
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, content=None, error=f"{type(exc).__name__}: {exc}")


_JSON_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class ToolRegistry:
    """工具注册中心。

    示例::

        reg = ToolRegistry()
        reg.register(CalculatorTool())
        names = reg.list_names()
        schema = reg.to_openai_schemas()
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    # ------------------------------------------------------------------
    def register(self, tool: Tool) -> None:
        """注册一个工具实例。同名工具将被覆盖并记日志。"""
        if not tool.name:
            raise ValueError("Tool.name 不能为空")
        if tool.name in self._tools:
            from src.utils import get_logger

            logger = get_logger("agent.tool")
            logger.warning("工具名 '%s' 重复注册，已覆盖", tool.name)
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """移除一个工具。"""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        """按名称获取工具；不存在返回 ``None``。"""
        return self._tools.get(name)

    def list_names(self) -> List[str]:
        """返回所有已注册工具的名称（按字典序）。"""
        return sorted(self._tools.keys())

    def list_tools(self) -> List[Tool]:
        """返回所有已注册工具实例列表。"""
        return [self._tools[n] for n in self.list_names()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def to_openai_schemas(self) -> List[Dict[str, Any]]:
        """返回 OpenAI function-calling 格式的 schema 列表。"""
        return [t.to_openai_schema() for t in self.list_tools()]
