"""Agent 内置工具集。

包括：
- :class:`SearchDocumentsTool`: 在 Chroma 知识库中检索文档
- :class:`ListDocumentsTool`: 列出已入库的文档
- :class:`GetDocumentChunksTool`: 获取某文档的分块
- :class:`CalculatorTool`: 数学表达式求值（仅 ``ast`` + 沙箱）
- :class:`GetCurrentTimeTool`: 获取当前时间
- :class:`TextStatsTool`: 统计文本长度 / 字数
- :class:`PythonEvalTool`: 安全的 Python 表达式求值（沙箱）
- :class:`BuiltinTools`: 上述工具的工厂方法
"""

from __future__ import annotations

import ast
import datetime as _dt
import math
import os
import re
from typing import Any, Dict, List, Optional

from src.utils import get_logger

from .tool import Tool, ToolRegistry, ToolResult

logger = get_logger("agent.builtin_tools")


# ====================================================================
#  知识库相关工具
# ====================================================================
class SearchDocumentsTool(Tool):
    """在 Chroma 知识库中检索文档。

    必传：``vector_store``（:class:`src.vector_store.ChromaStore` 实例）。

    参数：
    - query: 查询字符串
    - top_k: 返回条数（默认 4）
    """

    name = "search_documents"
    description = "在知识库（向量库）中检索与 query 最相关的文档片段。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "查询文本"},
            "top_k": {"type": "integer", "description": "返回条数", "default": 4, "minimum": 1, "maximum": 50},
        },
        "required": ["query"],
    }

    def __init__(self, vector_store: Optional[Any] = None) -> None:
        self.vector_store = vector_store

    def run(self, **kwargs) -> ToolResult:
        query: str = kwargs.get("query", "")
        top_k: int = int(kwargs.get("top_k", 4))
        if not query:
            return ToolResult(success=False, content=None, error="query 不能为空")
        if self.vector_store is None:
            return ToolResult(success=False, content=None, error="未配置 vector_store，无法检索")

        try:
            hits = self.vector_store.query(query_text=query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, content=None, error=f"检索失败: {exc}")

        snippets: List[Dict[str, Any]] = []
        for h in hits:
            snippets.append(
                {
                    "source": (h.metadata or {}).get("source") or "unknown",
                    "score": round(float(h.score), 4),
                    "text": h.text[:500],
                    "page": (h.metadata or {}).get("page", 1),
                }
            )
        return ToolResult(
            success=True,
            content={"query": query, "hits": snippets, "total": len(snippets)},
            metadata={"tool": self.name, "top_k": top_k},
        )


class ListDocumentsTool(Tool):
    """列出已入库的所有文档（按 source 聚合）。"""

    name = "list_documents"
    description = "列出知识库中所有已入库的文档及其分块数。"
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, vector_store: Optional[Any] = None) -> None:
        self.vector_store = vector_store

    def run(self, **kwargs) -> ToolResult:
        if self.vector_store is None:
            return ToolResult(success=False, content=None, error="未配置 vector_store")
        try:
            sources = self.vector_store.list_sources()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, content=None, error=str(exc))
        return ToolResult(
            success=True,
            content={"documents": sources, "total": len(sources)},
            metadata={"tool": self.name},
        )


class GetDocumentChunksTool(Tool):
    """获取某个文档的分块内容（前 N 条）。"""

    name = "get_document_chunks"
    description = "获取指定 source 文档的前若干条分块内容。"
    parameters = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "文档来源标识（文件名或路径）"},
            "limit": {"type": "integer", "description": "返回条数上限", "default": 10, "minimum": 1, "maximum": 100},
        },
        "required": ["source"],
    }

    def __init__(self, vector_store: Optional[Any] = None) -> None:
        self.vector_store = vector_store

    def run(self, **kwargs) -> ToolResult:
        source: str = kwargs.get("source", "")
        limit: int = int(kwargs.get("limit", 10))
        if not source:
            return ToolResult(success=False, content=None, error="source 不能为空")
        if self.vector_store is None:
            return ToolResult(success=False, content=None, error="未配置 vector_store")
        try:
            res = self.vector_store.collection.get(
                where={"source": {"$eq": source}},
                include=["documents", "metadatas"],
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, content=None, error=str(exc))
        chunks: List[Dict[str, Any]] = []
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        for i, doc in enumerate(docs):
            chunks.append(
                {
                    "id": ids[i] if i < len(ids) else f"chunk_{i}",
                    "text": doc[:500],
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return ToolResult(
            success=True,
            content={"source": source, "chunks": chunks, "total": len(chunks)},
            metadata={"tool": self.name},
        )


# ====================================================================
#  通用工具
# ====================================================================
class CalculatorTool(Tool):
    """数学计算器：基于 AST 安全求值。"""

    name = "calculator"
    description = "对数学表达式求值，例如 '2+3*4'、'sqrt(2)+1'。仅支持纯运算。"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式"},
        },
        "required": ["expression"],
    }

    _ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    _ALLOWED_UNARY = (ast.UAdd, ast.USub)
    _ALLOWED_FUNCS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
    }
    _ALLOWED_NAMES = {**{k: v for k, v in math.__dict__.items() if not k.startswith("_") and isinstance(v, (int, float))}, **{
        "pi": math.pi,
        "e": math.e,
    }}

    def run(self, **kwargs) -> ToolResult:
        expr: str = kwargs.get("expression", "")
        if not expr or not isinstance(expr, str):
            return ToolResult(success=False, content=None, error="expression 不能为空")
        try:
            value = self._safe_eval(expr)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, content=None, error=f"表达式非法: {exc}")
        return ToolResult(success=True, content={"expression": expr, "result": value}, metadata={"tool": self.name})

    def _safe_eval(self, expr: str) -> float:
        """使用 AST 白名单求值，禁止任意名字查找与函数调用（除白名单外）。"""
        tree = ast.parse(expr, mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST):
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"不支持的常量: {type(node.value).__name__}")
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, self._ALLOWED_BINOPS):
            return self._apply_binop(node.op, self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self._ALLOWED_UNARY):
            v = self._eval_node(node.operand)
            return +v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("仅支持函数名直接调用")
            fn = self._ALLOWED_FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"函数 {node.func.id} 不在白名单")
            args = [self._eval_node(a) for a in node.args]
            return fn(*args)
        if isinstance(node, ast.Name):
            if node.id in self._ALLOWED_NAMES:
                return self._ALLOWED_NAMES[node.id]
            raise ValueError(f"未授权的名字: {node.id}")
        raise ValueError(f"不支持的语法: {type(node).__name__}")

    @staticmethod
    def _apply_binop(op, a, b):
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            return a / b
        if isinstance(op, ast.FloorDiv):
            return a // b
        if isinstance(op, ast.Mod):
            return a % b
        if isinstance(op, ast.Pow):
            return a ** b
        raise ValueError(f"不支持的二元运算 {type(op).__name__}")


class GetCurrentTimeTool(Tool):
    """获取当前时间，默认为 ``Asia/Shanghai``。"""

    name = "get_current_time"
    description = "获取当前时间，可指定 IANA 时区。"
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "时区名（IANA，例如 Asia/Shanghai）", "default": "Asia/Shanghai"},
        },
        "required": [],
    }

    _NAME_MAP = {
        "Asia/Shanghai": 8,
        "Asia/Beijing": 8,
        "Asia/Hong_Kong": 8,
        "Asia/Tokyo": 9,
        "Asia/Singapore": 8,
        "Asia/Seoul": 9,
        "UTC": 0,
        "Etc/UTC": 0,
        "Europe/London": 0,
        "America/New_York": -5,
        "America/Los_Angeles": -8,
    }

    def run(self, **kwargs) -> ToolResult:
        tz_name: str = kwargs.get("timezone", "Asia/Shanghai") or "Asia/Shanghai"
        offset_h = self._NAME_MAP.get(tz_name)
        if offset_h is None:
            # 回退到固定偏移解析
            m = re.match(r"^UTC([+-]\d{1,2})(:?\d{2})?$", tz_name)
            if m:
                try:
                    offset_h = int(m.group(1))
                except Exception:
                    offset_h = 0
            else:
                offset_h = 8  # fallback to Asia/Shanghai
        try:
            from zoneinfo import ZoneInfo  # Python 3.9+

            dt = _dt.datetime.now(ZoneInfo(tz_name))
            tz_used = tz_name
        except Exception:
            tz_used = f"UTC{offset_h:+d}"
            dt = _dt.datetime.utcnow() + _dt.timedelta(hours=offset_h)
        return ToolResult(
            success=True,
            content={
                "timezone": tz_used,
                "iso": dt.isoformat(timespec="seconds"),
                "timestamp": int(dt.timestamp()),
                "human": dt.strftime("%Y-%m-%d %H:%M:%S"),
            },
            metadata={"tool": self.name},
        )


class TextStatsTool(Tool):
    """统计文本长度 / 字数 / 行数。"""

    name = "text_stats"
    description = "统计文本的字符数、中文字数、词数、行数等基本指标。"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "待统计文本"},
        },
        "required": ["text"],
    }

    def run(self, **kwargs) -> ToolResult:
        text: str = kwargs.get("text", "")
        if not isinstance(text, str):
            return ToolResult(success=False, content=None, error="text 必须是字符串")
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        words = len(text.split())
        return ToolResult(
            success=True,
            content={
                "chars": len(text),
                "cjk_chars": cjk,
                "words": words,
                "lines": lines if text else 0,
            },
            metadata={"tool": self.name},
        )


# ====================================================================
#  Python 沙箱
# ====================================================================
# 任何含这些关键字或字符的表达式都会被拒绝
_PY_BLOCKED_KEYWORDS = (
    "__import__",
    "import ",
    "from ",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "delattr(",
    "builtins",
    "subprocess",
    "os.",
    "sys.",
    "shutil",
    "pickle",
    "marshal",
    "input(",
    "breakpoint",
    "__",
    "lambda ",
)

_PY_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "int": int,
    "float": float,
    "len": len,
    "str": str,
    "range": range,
    "list": list,
    "tuple": tuple,
    "set": set,
    "dict": dict,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}

_PY_NAMES_BASE = {
    "True": True,
    "False": False,
    "None": None,
    "pi": math.pi,
    "e": math.e,
}


def _safe_py_eval(expr: str) -> Any:
    """更严格的 Python 沙箱求值（表达式级）。

    1. 关键字 / 函数名黑名单预筛；
    2. AST 解析后只允许：常量、二元运算、一元运算、函数调用（白名单）、名字（白名单）。
    3. 文件 I/O、import、exec 等高危操作被彻底拒绝。
    """
    if not expr or not isinstance(expr, str):
        raise ValueError("expression 不能为空")
    # 粗筛：一旦命中即拒绝
    lower = expr.lower()
    for kw in _PY_BLOCKED_KEYWORDS:
        if kw in lower:
            raise ValueError(f"检测到禁止的关键字或函数：{kw.strip()}")
    # AST 校验
    tree = ast.parse(expr, mode="eval")
    return _PySandboxVisitor().visit(tree.body)


class _PySandboxVisitor:
    """AST 遍历器：白名单语法节点 + 白名单函数/名字。"""

    _ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    _ALLOWED_UNARY = (ast.UAdd, ast.USub)
    _ALLOWED_COMPARE = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
    _ALLOWED_BOOL = (ast.And, ast.Or, ast.Not)

    def visit(self, node: ast.AST):  # noqa: C901 - 简单分发
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, str, bool)) or node.value is None:
                # 我们允许数字、字符串、布尔 / None；其它拒绝
                raise ValueError(f"不支持的常量类型 {type(node.value).__name__}")
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, self._ALLOWED_BINOPS):
            return self._binop(node.op, self.visit(node.left), self.visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self._ALLOWED_UNARY):
            v = self.visit(node.operand)
            return +v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BoolOp) and isinstance(node.op, self._ALLOWED_BOOL):
            if isinstance(node.op, ast.Not):
                return not self.visit(node.values[0])
            vals = [self.visit(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(vals)
            return any(vals)
        if isinstance(node, ast.Compare) and all(isinstance(c, self._ALLOWED_COMPARE) for c in node.ops):
            left = self.visit(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = self.visit(right_node)
                if isinstance(op, ast.Eq) and not (left == right):
                    return False
                if isinstance(op, ast.NotEq) and not (left != right):
                    return False
                if isinstance(op, ast.Lt) and not (left < right):
                    return False
                if isinstance(op, ast.LtE) and not (left <= right):
                    return False
                if isinstance(op, ast.Gt) and not (left > right):
                    return False
                if isinstance(op, ast.GtE) and not (left >= right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("仅支持直接函数名调用")
            fn = _PY_ALLOWED_FUNCTIONS.get(node.func.id)
            if fn is None:
                raise ValueError(f"函数 {node.func.id} 不在白名单")
            if node.keywords:
                raise ValueError("不支持关键字参数")
            args = [self.visit(a) for a in node.args]
            return fn(*args)
        if isinstance(node, ast.Name):
            if node.id in _PY_NAMES_BASE:
                return _PY_NAMES_BASE[node.id]
            raise ValueError(f"名字 {node.id} 不允许使用")
        if isinstance(node, ast.Tuple):
            return tuple(self.visit(elt) for elt in node.elts)
        if isinstance(node, ast.List):
            return [self.visit(elt) for elt in node.elts]
        if isinstance(node, ast.Subscript):
            # 只支持下标是 UnaryOp(USub, Constant[int]) / Constant[int] / Slice
            base = self.visit(node.value)
            slc = node.slice
            if isinstance(slc, ast.Constant) and isinstance(slc.value, int):
                try:
                    return base[slc.value]
                except Exception as exc:
                    raise ValueError(f"下标访问失败: {exc}")
            if isinstance(slc, ast.UnaryOp) and isinstance(slc.op, ast.USub):
                # -1 / -2 等
                inner = slc.operand
                if isinstance(inner, ast.Constant) and isinstance(inner.value, int):
                    try:
                        return base[-inner.value]
                    except Exception as exc:
                        raise ValueError(f"下标访问失败: {exc}")
                raise ValueError("负下标必须是常量整数")
            if isinstance(slc, ast.Slice):
                lower = slc.lower
                upper = slc.upper
                step = slc.step
                lo = self.visit(lower) if lower is not None else None
                up = self.visit(upper) if upper is not None else None
                st = self.visit(step) if step is not None else None
                return base[lo:up:st]
            raise ValueError("下标必须是整数或简单切片")
        raise ValueError(f"不支持的语法节点 {type(node).__name__}")

    @staticmethod
    def _binop(op, a, b):
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            return a / b
        if isinstance(op, ast.FloorDiv):
            return a // b
        if isinstance(op, ast.Mod):
            return a % b
        if isinstance(op, ast.Pow):
            return a ** b
        raise ValueError(f"不支持的二元运算 {type(op).__name__}")


class PythonEvalTool(Tool):
    """安全执行 Python 表达式（沙箱）。

    - 禁止：``__import__``、``import``、``eval``、``exec``、``open``、文件 I/O 等
    - 仅允许：数学运算、白名单内置函数、常量与简单数据容器
    """

    name = "python_eval"
    description = (
        "在沙箱中安全执行一个 Python 表达式并返回结果。"
        "禁止 import / 文件 I/O / 系统调用等高危操作。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Python 表达式字符串"},
        },
        "required": ["expression"],
    }

    def run(self, **kwargs) -> ToolResult:
        expr: str = kwargs.get("expression", "")
        try:
            value = _safe_py_eval(expr)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, content=None, error=f"沙箱拒绝或执行失败: {exc}")
        return ToolResult(
            success=True,
            content={"expression": expr, "result": value, "repr": repr(value)},
            metadata={"tool": self.name},
        )


# ====================================================================
#  工厂：BuiltinTools
# ====================================================================
class BuiltinTools:
    """内置工具工厂。

    Usage::

        registry = BuiltinTools.create(vector_store=my_store)
    """

    @staticmethod
    def create(
        vector_store: Optional[Any] = None,
        include: Optional[List[str]] = None,
    ) -> ToolRegistry:
        """构造一个包含所有（或子集）内置工具的 :class:`ToolRegistry`。

        Args:
            vector_store: 可选 ChromaStore 实例。
            include: 工具名白名单。``None`` 表示全部内置工具。
        """
        all_tools = {
            "search_documents": SearchDocumentsTool(vector_store=vector_store),
            "list_documents": ListDocumentsTool(vector_store=vector_store),
            "get_document_chunks": GetDocumentChunksTool(vector_store=vector_store),
            "calculator": CalculatorTool(),
            "get_current_time": GetCurrentTimeTool(),
            "text_stats": TextStatsTool(),
            "python_eval": PythonEvalTool(),
        }
        registry = ToolRegistry()
        names = include if include else sorted(all_tools.keys())
        for n in names:
            t = all_tools.get(n)
            if t is None:
                logger.warning("未知内置工具名 '%s'，已忽略", n)
                continue
            registry.register(t)
        return registry


# 重新导出便于 ``from src.agent.builtin_tools import _safe_py_eval`` 在测试中使用
__all__ = [
    "BuiltinTools",
    "CalculatorTool",
    "GetCurrentTimeTool",
    "GetDocumentChunksTool",
    "ListDocumentsTool",
    "PythonEvalTool",
    "SearchDocumentsTool",
    "TextStatsTool",
    "_safe_py_eval",
]
