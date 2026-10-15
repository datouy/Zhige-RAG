"""LLM 实例复用回归测试。

背景（实测过的性能缺陷）：多租户 / 多知识库路径用 ``copy.copy`` 复制全局
pipeline，而 ``get_pipeline()`` 默认懒加载（``llm=None``）。浅拷贝后每个副本
的 ``llm`` 也是 None，``ensure_llm`` 于是各自把权重加载一遍 —— 连续 3 个请求
加载 3 次模型，每个问答凭空多出数秒（本机实测 3.5s/次）。

修复方式：模型实例放在所有副本共享的 ``_llm_holder`` 里。
"""

from __future__ import annotations

import threading
import time

import pytest


@pytest.fixture()
def stubbed_deps(monkeypatch):
    """把 embedding / 向量库打桩，并且每次测试后清空全局单例。"""
    import api.deps as deps
    from src.rag_pipeline import RAGPipeline

    builds: list = []

    def fake_build(cls, llm_cfg):
        builds.append(len(builds) + 1)
        return object()

    monkeypatch.setattr(RAGPipeline, "_build_llm_from_cfg", classmethod(fake_build))
    monkeypatch.setattr(deps, "get_embedding", lambda: object())
    monkeypatch.setattr(deps, "get_vector_store", lambda: object())
    # reranker 默认已开启（配置里 enabled: true）：单元测试不该为此加载 1.1GB
    # 的交叉编码器，打个桩让 pipeline 装配逻辑照常走。
    import src.reranker as reranker_mod

    monkeypatch.setattr(reranker_mod, "create_reranker", lambda *a, **k: None)
    deps.reset_all_singletons()
    try:
        yield deps, builds
    finally:
        deps.reset_all_singletons()


def test_tenant_copies_share_one_llm(stubbed_deps):
    """连续 3 个租户请求只加载一次模型，且全局单例拿到同一实例。"""
    deps, builds = stubbed_deps

    base = deps.get_pipeline()
    assert base.llm is None, "全局单例应保持懒加载，不在启动时就占显存"

    last = None
    for _ in range(3):
        last = deps.get_runtime_pipeline(user_id="u-1", kb=None)
        last.ensure_llm()

    assert len(builds) == 1, f"模型只应加载 1 次，实际 {len(builds)} 次"
    assert last.llm is not None

    base.ensure_llm()
    assert base.llm is last.llm, "单例与副本必须共用同一个模型实例"


def test_ensure_llm_idempotent(stubbed_deps):
    """同一个 pipeline 上多次调用不会重复加载。"""
    deps, builds = stubbed_deps
    pipe = deps.get_runtime_pipeline(user_id="u-1", kb=None)
    for _ in range(5):
        pipe.ensure_llm()
    assert len(builds) == 1


def test_concurrent_first_questions_load_once(stubbed_deps, monkeypatch):
    """并发首问（LLM 闸允许 3 并发）也只允许一次真正的加载。"""
    deps, builds = stubbed_deps
    from src.rag_pipeline import RAGPipeline

    def slow_build(cls, llm_cfg):
        builds.append(len(builds) + 1)
        time.sleep(0.2)  # 放大竞态窗口
        return object()

    monkeypatch.setattr(RAGPipeline, "_build_llm_from_cfg", classmethod(slow_build))

    gates = [threading.Event() for _ in range(3)]
    errors: list = []

    def worker(i):
        try:
            pipe = deps.get_runtime_pipeline(user_id=f"u-{i}", kb=None)
            gates[i].wait(timeout=5)
            pipe.ensure_llm()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for gate in gates:
        gate.set()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors
    assert len(builds) == 1, f"并发首问只应加载 1 次，实际 {len(builds)} 次"


def test_warmup_disabled_under_pytest():
    """测试环境必须关闭启动预热，否则每个 TestClient 都会去加载真实模型。"""
    from api.main import _warmup_enabled

    assert _warmup_enabled() is False
