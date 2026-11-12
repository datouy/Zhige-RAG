"""文档里的默认值必须与 config.yaml 一致。

背景：``docs/answer-quality-tuning.md`` 的「可调参数」表列出了各参数默认值，
但这份表是**手写维护**的 —— 改配置时忘了同步文档，就出现"文档说 6000、
实际是 12000"这种漂移。读者照着文档调参会得到与预期完全不同的结果，
而且很难察觉（数字只差一倍，不算离谱）。

这里从 config.yaml 读真实值，与文档表格逐行比对，漂移即测试失败。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import load_config

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "answer-quality-tuning.md"


@pytest.fixture(scope="module")
def doc_text() -> str:
    assert DOC.exists(), f"找不到文档：{DOC}"
    return DOC.read_text(encoding="utf-8")


def _doc_table_defaults(doc_text: str) -> dict[str, str]:
    """从「可调参数」表格里解析出 参数名 -> 默认值。"""
    out: dict[str, str] = {}
    for line in doc_text.splitlines():
        line = line.strip()
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        name = cells[0].strip("`")
        if not name or name.startswith("-"):
            continue
        out[name] = cells[1].strip("`")
    return out


def _cfg_value(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _same(doc_value: str, cfg_value) -> bool:
    if cfg_value is None:
        return False
    if isinstance(cfg_value, bool):
        return doc_value.lower() == str(cfg_value).lower()
    if isinstance(cfg_value, (int, float)):
        try:
            return float(doc_value) == float(cfg_value)
        except ValueError:
            return False
    return doc_value.strip() == str(cfg_value).strip()


# 文档表格里出现的参数 -> config.yaml 里的路径
DOC_PARAM_TO_CONFIG = {
    "rag.max_context_tokens": "rag.max_context_tokens",
    "rag.fulltext_expand.max_documents": "rag.fulltext_expand.max_documents",
    "rag.fulltext_expand.max_chars": "rag.fulltext_expand.max_chars",
    "rag.mapreduce.soft_limit_chars": "rag.mapreduce.soft_limit_chars",
    "rag.mapreduce.detail_chunks": "rag.mapreduce.detail_chunks",
    "rag.verification.synthesis_aware": "rag.verification.synthesis_aware",
    "retrieval.top_k": "retrieval.top_k",
}


class TestDocDefaultsMatchConfig:
    def test_table_is_parsed(self, doc_text: str):
        defaults = _doc_table_defaults(doc_text)
        assert defaults, "没能从文档里解析出「可调参数」表"

    @pytest.mark.parametrize("doc_name", sorted(DOC_PARAM_TO_CONFIG))
    def test_default_matches_config_yaml(self, doc_text: str, doc_name: str):
        defaults = _doc_table_defaults(doc_text)
        assert doc_name in defaults, (
            f"文档「可调参数」表里没有 {doc_name} —— 参数表与代码脱节了"
        )

        cfg = load_config("config/config.yaml")
        cfg_value = _cfg_value(cfg, DOC_PARAM_TO_CONFIG[doc_name])
        assert cfg_value is not None, (
            f"config.yaml 里找不到 {DOC_PARAM_TO_CONFIG[doc_name]}"
        )

        doc_value = defaults[doc_name]
        assert _same(doc_value, cfg_value), (
            f"文档默认值与 config.yaml 不一致：{doc_name} 文档写 {doc_value!r}，"
            f"配置实际是 {cfg_value!r}。改配置时需同步更新 "
            f"docs/answer-quality-tuning.md 的「可调参数」表。"
        )


class TestRemovedConfigKeysAreNotDocumented:
    """已删除的死配置不该再出现在文档里 —— 否则读者会照着改一个无效项。"""

    @pytest.mark.parametrize("key", ["rag.top_k", "rag.use_reranker", "rag.score_threshold",
                                     "vector_store.top_k", "vector_store.fetch_k"])
    def test_dead_key_not_in_doc(self, doc_text: str, key: str):
        assert key not in doc_text, (
            f"{key} 已从配置中删除（从未被代码读取），不该再出现在文档里"
        )
