"""Cypher 注入防护（P2.1）。

背景
----
``SQLiteGraphStore.query_cypher`` 是只识别 ``MATCH ... RETURN`` 子集的极简实现，
但客户端可直接传入任意 Cypher 文本。如果不加以校验，攻击者可借此探测或绕过
内部语义。``Neo4jStore.query_cypher`` 则把 cypher 原样转发给驱动 — 后者虽然可以
使用 ``session.execute_read`` 等只读事务，但仍有副作用风险（事务日志写、AST
探测等）。本模块对所有后端统一做白名单 + 黑名单校验。

设计原则
--------
- **白名单**：必须以 ``MATCH`` 开头；必须包含 ``RETURN``；可选 LIMIT 注入。
- **黑名单**：拒绝任何写关键字（大小写无关、字边界匹配）。
- **剥注释/字符串**：在关键字匹配前先去掉 ``// ...``、``/* ... */``、
  ``'...'``、``"..."``，避免 ``// DELETE me`` 这种伪装。
- **LIMIT 注入**：如果用户没有写 LIMIT，自动追加 ``LIMIT <max_limit>``，作为
  上限（调用方仍可再次切片）。
- **Neo4j 只读事务**：调用方在拿到通过校验的 cypher 后，应当用
  ``session.execute_read(...)`` 包装（本模块只负责文本校验）。

公开 API
--------
- ``validate_readonly_cypher(cypher, max_limit=200)`` 返回 ``ValidatedCypher``。
- ``CypherValidationError`` 用于 ``raise HTTPException(400, ...)``。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# 写关键字黑名单（大小写无关；字边界匹配）。
# 覆盖：CREATE/DELETE/DETACH/SET/REMOVE/MERGE/DROP/CALL/LOAD/IMPORT/FOREACH，
# 以及一个简单启发式：``UNWIND <大数字>`` 多半是批写入。
_WRITE_KEYWORDS = (
    "CREATE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "MERGE",
    "DROP",
    "CALL",
    "LOAD",
    "IMPORT",
    "FOREACH",
)
# 整条正则：用 \b 强制字边界
_WRITE_KEYWORDS_RE = re.compile(
    r"\b(?:" + "|".join(_WRITE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
# 启发式：``UNWIND range(1, N)`` 中 N >= 1000 多半是批写入
_UNWIND_BIG_RANGE_RE = re.compile(
    r"\bUNWIND\s+range\s*\(\s*\d+\s*,\s*\d{3,}\s*\)",
    re.IGNORECASE,
)
_UNWIND_BIG_LITERAL_RE = re.compile(r"\bUNWIND\s*\d{4,}\b", re.IGNORECASE)

# 注释 / 字符串占位符（用于在关键字扫描前剥离可疑输入）
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
_SINGLE_QUOTED_RE = re.compile(r"'[^']*'")
_DOUBLE_QUOTED_RE = re.compile(r'"[^"]*"')


class CypherValidationError(ValueError):
    """Cypher 校验失败 — 调用方应转为 400 响应。"""


@dataclass(frozen=True)
class ValidatedCypher:
    """通过校验的 Cypher。

    Attributes:
        original:  客户端传入的原文（未改动，便于回传给客户端）。
        sanitized: 经过 ``LIMIT`` 注入的安全版本（用于执行）。
        max_limit: 注入的 LIMIT 上限。
    """

    original: str
    sanitized: str
    max_limit: int


def _strip_noise(text: str) -> str:
    """去掉注释与字符串字面量，便于关键字扫描。"""
    text = _LINE_COMMENT_RE.sub(" ", text)
    text = _BLOCK_COMMENT_RE.sub(" ", text)
    text = _SINGLE_QUOTED_RE.sub("''", text)
    text = _DOUBLE_QUOTED_RE.sub('""', text)
    return text


_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def _inject_limit(text: str, max_limit: int) -> str:
    """如果文本里没有 LIMIT 子句，在末尾追加 ``LIMIT <max_limit>``。"""
    if _LIMIT_RE.search(text):
        return text
    sep = "" if text.rstrip().endswith(";") else ""
    return f"{text.rstrip()}{sep} LIMIT {max_limit}".rstrip()


def validate_readonly_cypher(cypher: str, max_limit: int = 200) -> ValidatedCypher:
    """校验并清洗客户端传入的 Cypher。

    Args:
        cypher:    客户端传入的 Cypher 字符串。
        max_limit: 自动注入的 LIMIT 上限（默认 200）。

    Returns:
        :class:`ValidatedCypher`，其中 ``sanitized`` 可直接执行。

    Raises:
        CypherValidationError: 当 cypher 为空、超过长度上限、以非 MATCH 开头、
            缺少 RETURN、命中写关键字、UNWIND 超过 4 位数等情况。
    """
    if not cypher or not cypher.strip():
        raise CypherValidationError("cypher is empty")

    # 长度上限：避免巨型 payload 攻击
    MAX_LEN = 2048
    text = cypher.strip()
    if len(text) > MAX_LEN:
        raise CypherValidationError(f"cypher too long (>{MAX_LEN} chars)")

    # 必须以 MATCH 开头
    if not text.upper().startswith("MATCH"):
        raise CypherValidationError("only MATCH-prefixed read queries are allowed")

    # 必须包含 RETURN
    if "RETURN" not in text.upper():
        raise CypherValidationError("MATCH queries must contain RETURN")

    # 关键字扫描（在剥掉注释 / 字符串之后）
    stripped = _strip_noise(text)
    m = _WRITE_KEYWORDS_RE.search(stripped)
    if m:
        raise CypherValidationError(
            f"write keyword not allowed: {m.group(0).upper()}"
        )
    if _UNWIND_BIG_RANGE_RE.search(stripped) or _UNWIND_BIG_LITERAL_RE.search(stripped):
        raise CypherValidationError(
            "UNWIND with large literal is not allowed (possible batch write)"
        )

    # 自动注入 LIMIT
    sanitized = _inject_limit(text, max_limit)
    return ValidatedCypher(original=text, sanitized=sanitized, max_limit=max_limit)


__all__ = [
    "CypherValidationError",
    "ValidatedCypher",
    "validate_readonly_cypher",
]
