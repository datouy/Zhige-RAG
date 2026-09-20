"""``src.kg.cypher_guard`` 的表驱动测试（P3.2）。

覆盖：
- 合法 MATCH ... RETURN 语句通过校验，并自动注入 LIMIT。
- 用户显式 LIMIT 时不重复注入。
- 拒绝所有写关键字（CREATE/DELETE/MERGE/SET/REMOVE/DETACH/DROP/CALL/LOAD/IMPORT/FOREACH）。
- 拒绝以非 MATCH 开头的语句。
- 拒绝没有 RETURN 的 MATCH。
- 拒绝 UNWIND 大数字（超过 4 位 / range(...) 中超过 3 位）。
- 拒绝空字符串 / 超过 2048 字符。
- 注释与字符串内的关键字不触发黑名单。
"""

import pytest

from src.kg.cypher_guard import (
    CypherValidationError,
    ValidatedCypher,
    validate_readonly_cypher,
)


# ----------------------------------------------------------------------
#  Valid cases
# ----------------------------------------------------------------------


VALID_QUERIES = [
    "MATCH (n) RETURN n",
    "MATCH (n:Person) WHERE n.name = 'Alice' RETURN n.name",
    "MATCH (a)-[r:KNOWS]->(b) RETURN a.name, b.name, type(r)",
    "MATCH (n) RETURN n LIMIT 10",
    "MATCH (n) RETURN count(*)",
]


@pytest.mark.parametrize("cypher", VALID_QUERIES)
def test_valid_cypher_passes(cypher):
    result = validate_readonly_cypher(cypher)
    assert isinstance(result, ValidatedCypher)
    assert result.sanitized
    # 如果原查询没有 LIMIT，自动注入
    if "LIMIT" not in cypher.upper():
        assert "LIMIT 200" in result.sanitized.upper()


def test_existing_limit_is_preserved():
    cypher = "MATCH (n) RETURN n LIMIT 5"
    r = validate_readonly_cypher(cypher)
    assert "LIMIT 5" in r.sanitized
    assert "LIMIT 200" not in r.sanitized


def test_max_limit_is_configurable():
    r = validate_readonly_cypher("MATCH (n) RETURN n", max_limit=10)
    assert "LIMIT 10" in r.sanitized


def test_keyword_inside_string_does_not_trigger_blocklist():
    """'CREATE me' 写在单引号里不应触发。"""
    cypher = "MATCH (n) WHERE n.name = 'CREATE' RETURN n"
    r = validate_readonly_cypher(cypher)
    assert "RETURN n" in r.sanitized


def test_keyword_inside_comment_does_not_trigger_blocklist():
    """``// CREATE`` 写在行注释里不应触发。"""
    cypher = "MATCH (n) // CREATE this\nRETURN n"
    r = validate_readonly_cypher(cypher)
    assert r.sanitized


def test_keyword_inside_block_comment_does_not_trigger_blocklist():
    cypher = "MATCH (n) /* DELETE me */ RETURN n"
    r = validate_readonly_cypher(cypher)
    assert r.sanitized


# ----------------------------------------------------------------------
#  Invalid cases
# ----------------------------------------------------------------------


def test_empty_raises():
    with pytest.raises(CypherValidationError):
        validate_readonly_cypher("")
    with pytest.raises(CypherValidationError):
        validate_readonly_cypher("   \n  ")


def test_too_long_raises():
    big = "MATCH (n) RETURN n " + "x" * 2050
    with pytest.raises(CypherValidationError) as exc:
        validate_readonly_cypher(big)
    assert "too long" in str(exc.value).lower()


def test_non_match_prefix_raises():
    bad_prefixes = [
        "RETURN n",
        "CREATE (n) RETURN n",
        "OPTIONAL MATCH (n) RETURN n",
        "MERGE (n) RETURN n",
    ]
    for q in bad_prefixes:
        with pytest.raises(CypherValidationError):
            validate_readonly_cypher(q)


def test_missing_return_raises():
    with pytest.raises(CypherValidationError) as exc:
        validate_readonly_cypher("MATCH (n)")
    assert "RETURN" in str(exc.value)


# ----------------------------------------------------------------------
#  Write-keyword blocklist
# ----------------------------------------------------------------------


WRITE_KEYWORDS = [
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
]


@pytest.mark.parametrize("kw", WRITE_KEYWORDS)
def test_write_keyword_rejected(kw):
    cypher = f"MATCH (n) {kw} something RETURN n"
    with pytest.raises(CypherValidationError) as exc:
        validate_readonly_cypher(cypher)
    assert kw.lower() in str(exc.value).lower() or kw in str(exc.value)


@pytest.mark.parametrize("kw", WRITE_KEYWORDS)
def test_write_keyword_case_insensitive_rejected(kw):
    cypher = f"MATCH (n) {kw.lower()} something RETURN n"
    with pytest.raises(CypherValidationError):
        validate_readonly_cypher(cypher)


def test_unwind_big_literal_rejected():
    cypher = "MATCH (n) UNWIND 12345 AS x RETURN x"
    with pytest.raises(CypherValidationError):
        validate_readonly_cypher(cypher)


def test_unwind_big_range_rejected():
    cypher = "MATCH (n) UNWIND range(1, 10000) AS x RETURN x"
    with pytest.raises(CypherValidationError):
        validate_readonly_cypher(cypher)


def test_unwind_small_range_allowed():
    cypher = "MATCH (n) UNWIND range(1, 5) AS x RETURN x"
    r = validate_readonly_cypher(cypher)
    assert r.sanitized


# ----------------------------------------------------------------------
#  Sanitized equals original when LIMIT is present
# ----------------------------------------------------------------------


def test_validated_cypher_exposes_original():
    raw = "MATCH (n) RETURN n LIMIT 3"
    r = validate_readonly_cypher(raw)
    assert r.original == "MATCH (n) RETURN n LIMIT 3"
