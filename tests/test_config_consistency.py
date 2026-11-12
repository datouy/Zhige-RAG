"""配置一致性守护测试。

背景：项目里曾经同时存在**两套默认值** ——
``config/config.yaml`` 里写着的值，和 ``src/config/validator.py`` 里 pydantic
模型的 ``Field(default=...)``。两套默认值互不引用，改了一处忘了另一处，
于是出现"改了 yaml 没生效"或"没改却被默认值覆盖"。

例如 ``retrieval.top_k`` 在 yaml 里是 8、在 validator 里是 4；
``rag.top_k`` / ``vector_store.top_k`` 更是**从未被任何代码读取**的纯死配置。

这里用断言把两份真相钉在一起：yaml 是准的，validator 的默认值必须与它一致。
"""

import copy

import pytest
from pydantic import ValidationError

from src.config.validator import validate_config
from src.utils import load_config


def test_current_config_is_valid():
    assert validate_config(load_config("config/config.yaml"))


def test_unknown_config_field_is_rejected():
    cfg = copy.deepcopy(load_config("config/config.yaml"))
    cfg["vector_store"]["persist_direcotry"] = "typo"
    with pytest.raises(ValidationError, match="persist_direcotry"):
        validate_config(cfg)


# ======================== yaml 与 validator 默认值一致 ========================

# (yaml 路径, validator 模型, 字段名)
DEFAULT_CONSISTENCY_CASES = [
    (("retrieval", "top_k"), "retrieval", "top_k"),
    (("retrieval", "fetch_k"), "retrieval", "fetch_k"),
    (("retrieval", "rerank_candidates"), "retrieval", "rerank_candidates"),
    (("retrieval", "min_score"), "retrieval", "min_score"),
    (("retrieval", "keyword_min_score"), "retrieval", "keyword_min_score"),
    (("retrieval", "score_threshold"), "retrieval", "score_threshold"),
    (("retrieval", "max_query_variants"), "retrieval", "max_query_variants"),
    (("ui", "default_top_k"), "ui", "default_top_k"),
    (("ui", "max_upload_mb"), "ui", "max_upload_mb"),
    (("ui", "max_chat_history"), "ui", "max_chat_history"),
    (("rag", "max_context_tokens"), "rag", "max_context_tokens"),
    (("rag", "max_history_turns"), "rag", "max_history_turns"),
    (("reranker", "backend"), "reranker", "backend"),
]


@pytest.mark.parametrize(
    "yaml_path,model_name,field_name",
    DEFAULT_CONSISTENCY_CASES,
    ids=[f"{'.'.join(p)}" for p, _, _ in DEFAULT_CONSISTENCY_CASES],
)
def test_yaml_value_matches_validator_default(yaml_path, model_name, field_name):
    """yaml 里的值必须与 validator 的默认值一致。

    不一致意味着：把 yaml 里那项删掉（或写错名字）时，validator 会用另一个
    数值兜底，行为与用户预期不符 —— 这正是要防的"配置不生效"假象。
    """
    from src.config import validator as v

    cfg = load_config("config/config.yaml")
    yaml_value = cfg
    for key in yaml_path:
        yaml_value = yaml_value[key]

    model = getattr(v, {
        "retrieval": "RetrievalConfig",
        "ui": "UIConfig",
        "rag": "RAGConfig",
        "reranker": "RerankerConfig",
    }[model_name])
    default_value = model.model_fields[field_name].default

    assert yaml_value == default_value, (
        f"config.yaml 的 {'.'.join(yaml_path)} = {yaml_value!r}，"
        f"但 validator.{model_name}.{field_name} 的默认值是 {default_value!r}。"
        f"两者必须一致，否则 yaml 的该项会被悄悄覆盖。"
    )


def test_dead_top_k_config_fields_are_gone():
    """``rag.top_k`` / ``vector_store.top_k`` 等死配置必须已删除。

    它们从未被任何代码读取；留着会让人以为改它们能调召回条数。
    """
    cfg = load_config("config/config.yaml")
    assert "top_k" not in (cfg.get("rag") or {}), "rag.top_k 是死配置，应删除"
    assert "fetch_k" not in (cfg.get("rag") or {}), "rag.fetch_k 是死配置，应删除"
    assert "use_reranker" not in (cfg.get("rag") or {}), "rag.use_reranker 是死配置，应删除"
    assert "score_threshold" not in (cfg.get("rag") or {}), "rag.score_threshold 是死配置，应删除"
    assert "top_k" not in (cfg.get("vector_store") or {}), "vector_store.top_k 是死配置，应删除"
    assert "fetch_k" not in (cfg.get("vector_store") or {}), "vector_store.fetch_k 是死配置，应删除"


def test_top_k_has_single_source_of_truth():
    """``retrieval.top_k`` 必须是检索条数的唯一真值来源。

    接口请求体 / WebSocket / 前端都应缺省回落��它，而不是各自写死一个数字。
    """
    from api.routes.chat import ChatRequest, SearchBody

    assert ChatRequest.model_fields["top_k"].default is None, (
        "ChatRequest.top_k 的缺省应为 None，由服务端回落到 retrieval.top_k"
    )
    assert SearchBody.model_fields["top_k"].default is None, (
        "SearchBody.top_k 的缺省应为 None，由服务端回落到 retrieval.top_k"
    )

    # 检索层自己也不能再写死
    src = open("src/rag_pipeline.py", encoding="utf-8").read()
    assert 'retrieval_cfg.get("top_k"' in src

def test_retrieval_sufficiency_default_on():
    """拒答前置应默认开启（受 decision.enabled 总开关门控，extractive 部署不触发）。"""
    cfg = load_config("config/config.yaml")
    rs = ((cfg.get("decision") or {}).get("retrieval_sufficiency")) or {}
    assert rs.get("enabled") is True, (
        "decision.retrieval_sufficiency.enabled 应默认开启，"
        "以便打开决策层后自动获得检索不足即拒答保护"
    )
