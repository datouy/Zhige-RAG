"""在线重排 API（P0-5：Cohere / 硅基流动 / DashScope 兼容 /rerank）回归测试。"""

from __future__ import annotations


from src.reranker import ApiReranker, create_reranker
from src.vector_store import Hit


def _hits():
    return [
        Hit(id="1", text="公司年假制度：员工入职满一年可休五天年假。", score=0.9, metadata={"source": "a.md"}),
        Hit(id="2", text="差旅报销需在出差结束后三十日内提交。", score=0.5, metadata={"source": "a.md"}),
    ]


def test_create_reranker_api_backend_with_base_url(monkeypatch):
    cfg = {
        "reranker": {
            "enabled": True,
            "backend": "api",
            "base_url": "https://api.siliconflow.cn/v1",
            "model_name": "BAAI/bge-reranker-v2-m3",
            "api_key": "",
        }
    }
    r = create_reranker(cfg)
    assert isinstance(r, ApiReranker)
    assert r.base_url == "https://api.siliconflow.cn/v1"


def test_create_reranker_api_backend_without_base_url_disabled():
    cfg = {"reranker": {"enabled": True, "backend": "api", "base_url": ""}}
    assert create_reranker(cfg) is None


def test_create_reranker_auto_without_base_url_falls_through(monkeypatch):
    # auto 且无 base_url / d1 不可用 / 无 model_name → 最终 None（不崩溃）
    cfg = {"reranker": {"enabled": True, "backend": "auto", "base_url": ""}, "decision": {"enabled": False}}
    assert create_reranker(cfg) is None


def test_api_reranker_reranks_and_marks_backend(monkeypatch):
    def _fake_post_json(url, payload, headers=None, timeout=None, label=None, model=None):
        return {
            "results": [
                {"index": 1, "relevance_score": 0.99},
                {"index": 0, "relevance_score": 0.01},
            ]
        }

    r = ApiReranker(base_url="https://api.example.com/v1", api_key="k", model_name="m")

    import src.http_client as hc
    monkeypatch.setattr(hc, "post_json", _fake_post_json)

    out = r.rerank("年假", _hits(), top_n=2)
    assert out[0].id == "2"  # 重排后最相关的排前面
    assert out[0].metadata["_rerank_backend"] == "api"
    assert out[0].score == 0.99


def test_api_reranker_failure_falls_back_to_initial_order(monkeypatch):
    def _fail(*a, **k):
        raise RuntimeError("network down")

    import src.http_client as hc
    monkeypatch.setattr(hc, "post_json", _fail)

    r = ApiReranker(base_url="https://api.example.com/v1", api_key="k", model_name="m")
    out = r.rerank("年假", _hits(), top_n=2)
    # 失败时按初检顺序降级返回，绝不抛异常
    assert [h.id for h in out] == ["1", "2"]
