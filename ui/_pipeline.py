"""共享的 RAG 流水线模块。

把 ``load_pipeline`` 从 ``ui/app.py`` 抽出来，避免 page_modules 引用 ``ui/app``
时引发循环 import，进而触发 ``st.set_page_config()`` 重复调用报错。
"""
from __future__ import annotations

import streamlit as st

from src.rag_pipeline import RAGPipeline
from src.utils import apply_env_overrides, load_config


@st.cache_resource(show_spinner=False)
def load_pipeline(config_path: str = "config/config.yaml") -> RAGPipeline:
    """惰性加载整个 RAG 流水线（避免每次操作都重新加载模型）。"""
    cfg = load_config(config_path)
    cfg = apply_env_overrides(cfg)
    pipeline = RAGPipeline.from_config(config_path, overrides=cfg, lazy_llm=True)
    return pipeline
