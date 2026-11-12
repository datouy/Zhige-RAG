"""重排后端自检的回归测试。

背景（真实故障）：FlagEmbedding 与当前 transformers 不兼容时，``compute_score``
会抛 ``XLMRobertaTokenizer has no attribute prepare_for_model``。该异常此前只在
``rerank()`` 里被捕获并**静默降级成初检顺序** —— ``reranker.enabled: true``
看上去开着，实际一次都没重排过。现在加载时会自检一次并切到 CrossEncoder。
"""

from __future__ import annotations

import sys
import types


from src.reranker import BgeReranker


def _bare_reranker(model, backend: str = "flag") -> BgeReranker:
    """不加载任何模型，只搭出 _verify_flag_backend 需要的字段。"""
    r = BgeReranker.__new__(BgeReranker)
    r.model = model
    r._backend = backend
    r.max_length = 256
    r.device = "cpu"
    r.model_name = "stub"
    return r


class _BrokenFlagModel:
    """模拟 FlagEmbedding 与 transformers 版本不合时的行为。"""

    def compute_score(self, pairs, normalize=True):
        raise RuntimeError("XLMRobertaTokenizer has no attribute prepare_for_model")


class _WorkingFlagModel:
    def compute_score(self, pairs, normalize=True):
        return [0.5 for _ in pairs]


def test_switches_to_cross_encoder_when_flag_scoring_breaks(monkeypatch):
    created: dict = {}

    class _StubCrossEncoder:
        def __init__(self, model_name, max_length=None, device=None):
            created["args"] = (model_name, max_length, device)

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = _StubCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    r = _bare_reranker(_BrokenFlagModel())
    r._verify_flag_backend("stub-model")

    assert r._backend == "cross_encoder", "打分不可用就必须换后端，不能继续假装在重排"
    assert isinstance(r.model, _StubCrossEncoder)
    assert created["args"][0] == "stub-model"


def test_keeps_flag_backend_when_scoring_works(monkeypatch):
    def _fail(*_a, **_k):  # pragma: no cover - 正常路径不该走到这里
        raise AssertionError("打分可用时不该构造 CrossEncoder")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = _fail
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    r = _bare_reranker(_WorkingFlagModel())
    r._verify_flag_backend("stub-model")

    assert r._backend == "flag"


def test_threshold_is_calibrated_for_cross_encoder():
    """配置层回归：阈值必须按交叉编码器的分数量程校准。

    实测相关提问 ≥0.087、无关提问 ≤0.001；旧值 0.3 是给 FlagEmbedding 的
    归一化分数用的，换到 CrossEncoder 后会把含糊但相关的候选全部误杀。

    注意这里读的是**仓库里的 config/config.yaml**，而不是
    ``effective_config()``：后者会叠加 ``config/local_overrides.yaml``
    （该文件被 .git 忽略，属于每台机器自己的本地设置）。用 effective_config
    断言"必须关闭重排"就等于把测试绑死在某一台开发机的本地配置上 ——
    别人 clone 下来改了本地覆盖，这个测试就挂了，与代码对错无关。
    """
    from src.utils import load_config

    rcfg = load_config("config/config.yaml").get("reranker") or {}
    # The shipped default is the LLM-free ONNX profile: a 3.2 GB cross
    # encoder is disabled so all RAM/VRAM stays available for retrieval/OCR.
    assert rcfg.get("enabled") is False
    assert float(rcfg.get("min_relevance")) <= 0.1
