"""``src.llm`` 的回归测试（P3.2）。

覆盖：
- 当 ``quant.enabled=True`` 时，``BitsAndBytesConfig`` 必须带
  ``load_in_4bit=True`` 和 ``llm_int8_enable_fp32_cpu_offload=True``，并把
  ``device_map`` 强制为 ``"auto"``。
- 当 ``quant.enabled=False`` 时，不传 ``quantization_config``，
  ``torch_dtype`` 按 ``cuda/cpu`` 解析。
- ``LocalLLM.estimate_memory_gb`` 的判定逻辑。

不实际下载模型权重；用 ``unittest.mock`` 在 ``transformers`` 模块层做 patch。
"""

from unittest.mock import MagicMock, patch

import pytest

from src.llm import LocalLLM, _detect_device


# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------


def _make_fake_transformers(bnb_holder):
    """构造一组会被 ``_load_model`` / ``_load_tokenizer`` 用来调用的 fake。

    通过往 ``sys.modules["transformers"]`` 里塞 MagicMock 来拦截。
    """

    fake_model = MagicMock(name="fake_model")
    fake_model.eval = MagicMock()
    fake_model.config = MagicMock(pad_token_id=None, eos_token_id=99)

    fake_tokenizer = MagicMock(name="fake_tokenizer")
    fake_tokenizer.pad_token_id = 99
    fake_tokenizer.pad_token = "<pad>"
    fake_tokenizer.eos_token = "<eos>"

    def from_pretrained(name, **kwargs):
        # 关键：把 kwargs 暴露到 bnb_holder
        bnb_holder["from_pretrained_kwargs"] = kwargs
        bnb_holder["from_pretrained_called"] = True
        return fake_model

    def tokenizer_from_pretrained(name, **kwargs):
        return fake_tokenizer

    def bnb_config(**kwargs):
        bnb_holder["bnb_kwargs"] = kwargs
        return MagicMock(name="bnb_config")

    return MagicMock(
        AutoModelForCausalLM=MagicMock(from_pretrained=from_pretrained),
        AutoTokenizer=MagicMock(from_pretrained=tokenizer_from_pretrained),
        BitsAndBytesConfig=MagicMock(side_effect=bnb_config),
        TextIteratorStreamer=MagicMock(),
    )


@pytest.fixture
def fake_xf(monkeypatch):
    """在整个测试期间把 ``sys.modules['transformers']`` 替换成 MagicMock，
    并把 ``from_pretrained`` 调用结果存到 ``bnbs`` 字典里供断言。"""
    bnb_holder: dict = {}
    fake = _make_fake_transformers(bnb_holder)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake)
    return bnb_holder


# ----------------------------------------------------------------------
#  Tests
# ----------------------------------------------------------------------


def test_quant_enabled_passes_bitsandbytes_and_device_map(fake_xf):
    """当 ``quant.enabled=True`` 时，必须：
    1. 把 ``BitsAndBytesConfig(load_in_4bit=True,
       llm_int8_enable_fp32_cpu_offload=True, ...)`` 传给 from_pretrained
    2. 把 ``device_map="auto"`` 显式注入
    3. 不传 ``torch_dtype``
    """
    with patch("src.llm._detect_device", return_value="cuda"), \
         patch("src.llm.Timer"):
        LocalLLM(
            model_name="dummy/model",
            quant={"enabled": True, "quant_type": "nf4", "compute_dtype": "float16"},
            torch_dtype="auto",
        )

    assert fake_xf.get("from_pretrained_called"), "from_pretrained was not called"
    bnb_kwargs = fake_xf.get("bnb_kwargs")
    assert bnb_kwargs is not None, "BitsAndBytesConfig was not constructed"
    assert bnb_kwargs.get("load_in_4bit") is True, "load_in_4bit must be True"
    assert bnb_kwargs.get("llm_int8_enable_fp32_cpu_offload") is True, (
        "llm_int8_enable_fp32_cpu_offload must be True (regression for earlier bug)"
    )

    fp_kwargs = fake_xf["from_pretrained_kwargs"]
    assert fp_kwargs.get("device_map") == "auto", (
        "device_map must be forced to 'auto' when quant is enabled"
    )
    assert "quantization_config" in fp_kwargs, (
        "quantization_config must be forwarded to from_pretrained"
    )
    assert "torch_dtype" not in fp_kwargs, (
        "torch_dtype must NOT be passed when quantization is enabled"
    )


def test_quant_disabled_uses_torch_dtype(fake_xf):
    """当 ``quant.enabled=False`` 时，from_pretrained 应该拿到 ``torch_dtype``
    而不带 ``quantization_config``。
    """
    with patch("src.llm._detect_device", return_value="cpu"), \
         patch("src.llm.Timer"):
        LocalLLM(
            model_name="dummy/model",
            quant={"enabled": False},
            torch_dtype="float32",
        )

    fp_kwargs = fake_xf.get("from_pretrained_kwargs", {})
    assert "quantization_config" not in fp_kwargs, (
        "quantization_config must not be passed when quant disabled"
    )
    assert fp_kwargs.get("device_map") == "auto"


def test_detect_device_returns_explicit_value():
    """显式传入 cpu / cuda 不应触发探测。"""
    assert _detect_device("cpu") == "cpu"
    assert _detect_device("cuda:0") == "cuda:0"


def test_detect_device_auto_with_torch(monkeypatch):
    """auto 在 torch.cuda.is_available()==True 时返回 'cuda'，否则 'cpu'。"""
    fake_torch = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    fake_torch.cuda.is_available.return_value = False
    assert _detect_device("auto") == "cpu"

    fake_torch.cuda.is_available.return_value = True
    assert _detect_device("auto") == "cuda"


def test_estimate_memory_gb():
    """简单的 size-map 估算测试。"""
    assert LocalLLM.estimate_memory_gb("Qwen2.5-1.5B-Instruct", quant=False) == 3.0
    assert LocalLLM.estimate_memory_gb("Qwen2.5-1.5B-Instruct", quant=True) == 1.2
    assert LocalLLM.estimate_memory_gb("unknown-model", quant=False) == 8.0
    assert LocalLLM.estimate_memory_gb("unknown-model", quant=True) == 4.0
