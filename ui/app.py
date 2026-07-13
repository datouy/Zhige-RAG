"""Streamlit Web UI - 中文知识库 RAG 系统。

启动命令：
    streamlit run ui/app.py --server.port 8501

注意：如需使用新的 FastAPI 模式，请运行：
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# 让 Streamlit 能从项目根目录导入 src.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# =========================== Streamlit 兼容性检测 ===========================
# 检测是否应在 FastAPI 模式下运行
_streamlit_mode = os.environ.get("STREAMLIT_RUNTIME", "")

def _check_fastapi_mode():
    """检测 FastAPI 模式：如果 ui/web 目录存在且未强制要求 Streamlit，则提示用户。"""
    ui_web_dir = ROOT / "ui" / "web"
    # 如果 ui/web 目录存在，说明 FastAPI 模式已启用
    if ui_web_dir.exists() and not _streamlit_mode.lower() == "enable":
        print("=" * 60)
        print("⚠️  FastAPI 模式已启用！")
        print()
        print("推荐使用新的 Web UI：")
        print("    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
        print()
        print("然后在浏览器中打开 http://localhost:8000")
        print()
        print("如需使用旧版 Streamlit UI，请设置环境变量：")
        print("    STREAMLIT_RUNTIME=enable streamlit run ui/app.py")
        print("=" * 60)
        print()

_check_fastapi_mode()

import streamlit as st

from src.document_loader import load_document
from src.embeddings import EmbeddingModel
from src.llm import LocalLLM
from src.rag_pipeline import RAGPipeline
from src.reranker import BgeReranker
from src.text_splitter import ChineseTextSplitter, RecursiveTextSplitter
from src.utils import (
    apply_env_overrides,
    get_logger,
    load_config,
    merge_dict,
    resolve_path,
)
from src.vector_store import ChromaStore

from ui.page_modules.analytics import render_page as page_analytics
from ui.page_modules.versions import render_page as page_versions
from ui.page_modules.agent import render_page as page_agent
from ui.page_modules.kg import render_page as page_kg

logger = get_logger("ui")

# ============================== 常量与样式 ==============================
st.set_page_config(
    page_title="中文知识库 RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .kb-card { padding: 1rem; border-radius: 0.5rem; background: #f6f8fa; border: 1px solid #e5e7eb; margin-bottom: 0.5rem;}
    .kb-cite  { color: #2563eb; font-weight: 600; }
    .kb-meta  { color: #6b7280; font-size: 0.85rem; }
    .kb-stat  { font-size: 1.6rem; font-weight: 700; color: #111827; }
    .kb-label { color: #6b7280; font-size: 0.85rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================== 全局状态 ==============================
@st.cache_resource(show_spinner=False)
def load_pipeline(config_path: str) -> RAGPipeline:
    """惰性加载整个 RAG 流水线（避免每次操作都重新加载模型）。"""
    cfg = load_config(config_path)
    cfg = apply_env_overrides(cfg)
    pipeline = RAGPipeline.from_config(config_path, overrides=cfg, lazy_llm=True)
    return pipeline


@st.cache_resource(show_spinner=False)
def get_embedding_for_ui(config_path: str) -> EmbeddingModel:
    cfg = load_config(config_path)
    cfg = apply_env_overrides(cfg)
    return EmbeddingModel(
        model_name=cfg["embedding"]["model_name"],
        device=cfg["embedding"].get("device", "auto"),
        batch_size=cfg["embedding"].get("batch_size", 32),
        max_seq_length=cfg["embedding"].get("max_seq_length", 512),
        normalize=cfg["embedding"].get("normalize_embeddings", True),
        cache_dir=cfg["embedding"].get("cache_dir"),
        local_files_only=cfg["embedding"].get("local_files_only", False),
    )


@st.cache_resource(show_spinner=False)
def get_vector_store(config_path: str, _embed: EmbeddingModel) -> ChromaStore:
    cfg = load_config(config_path)
    return ChromaStore(
        persist_directory=cfg["vector_store"]["persist_directory"],
        collection_name=cfg["vector_store"].get("collection_name", "chinese_rag_kb"),
        embedding_model=_embed,
        distance_fn=cfg["vector_store"].get("distance_fn", "cosine"),
    )


@st.cache_resource(show_spinner=False)
def get_reranker(config_path: str):
    cfg = load_config(config_path)
    rcfg = cfg.get("reranker", {})
    if not rcfg.get("enabled"):
        return None
    return BgeReranker(
        model_name=rcfg["model_name"],
        device=rcfg.get("device", "auto"),
        cache_dir=rcfg.get("cache_dir"),
    )


def _raw_docs_dir(cfg: dict) -> Path:
    """兼容多种命名：raw_docs / raw_docs_dir。"""
    paths = cfg.get("paths", {}) or {}
    rel = paths.get("raw_docs") or paths.get("raw_docs_dir") or "data/raw"
    return resolve_path(rel)


def _eval_dataset_path(cfg: dict) -> Path:
    ev = cfg.get("evaluation", {}) or {}
    rel = ev.get("dataset_path") or "data/eval/eval_set.jsonl"
    return resolve_path(rel)


def _top_k(cfg: dict) -> int:
    return (
        (cfg.get("retrieval", {}) or {}).get("top_k")
        or (cfg.get("ui", {}) or {}).get("default_top_k")
        or 4
    )


def get_splitter(cfg: dict):
    sp = cfg.get("text_splitter", {})
    strategy = sp.get("strategy", "chinese")
    if strategy == "chinese":
        return ChineseTextSplitter(
            chunk_size=sp.get("chunk_size", 256),
            chunk_overlap=sp.get("chunk_overlap", 32),
            separators=sp.get("chinese_separators") or sp.get("separators"),
            keep_separator=sp.get("keep_separator", True),
            min_chunk_size=sp.get("min_chunk_size", 32),
        )
    return RecursiveTextSplitter(
        chunk_size=sp.get("chunk_size", 512),
        chunk_overlap=sp.get("chunk_overlap", 64),
    )


# ============================== 侧边栏 ==============================
def render_sidebar(config_path: str):
    cfg = load_config(config_path)
    embed = get_embedding_for_ui(config_path)
    store = get_vector_store(config_path, embed)

    with st.sidebar:
        st.title("📚 中文知识库 RAG")
        app_meta = cfg.get("app", {}) or cfg.get("project", {})
        st.caption(f"v{app_meta.get('version', '0.1.0')} · {app_meta.get('name', 'ChineseRAGKB')}")

        st.markdown("### 📊 知识库状态")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="kb-label">分块总数</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kb-stat">{store.count()}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="kb-label">Embedding</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kb-stat">{embed.dimension}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="kb-meta">模型: <code>{cfg["embedding"]["model_name"]}</code><br/>设备: <code>{embed.device}</code></div>',
            unsafe_allow_html=True,
        )

        st.divider()
        page = st.radio(
            "功能导航",
            ["💬 智能问答", "🤖 Agent", "📤 文档上传", "📁 知识库管理", "📜 文档版本", "🕸 知识图谱", "⚙️ 系统设置", "📈 评估", "📊 指标可视化"],
            index=0,
        )
        st.divider()

        with st.expander("🛠 调试信息", expanded=False):
            llm_cfg = cfg.get("llm", {})
            emb_cfg = cfg.get("embedding", {})
            vs_cfg = cfg.get("vector_store", {})
            ts_cfg = cfg.get("text_splitter", {})
            rr_cfg = cfg.get("reranker", {})
            quant_cfg = llm_cfg.get("quantization", {}) if isinstance(llm_cfg.get("quantization"), dict) else {}
            st.json(
                {
                    "embedding_model": emb_cfg.get("model_name"),
                    "llm_model": llm_cfg.get("model_name"),
                    "reranker_enabled": rr_cfg.get("enabled", False),
                    "vector_store": vs_cfg.get("persist_directory"),
                    "chunk_size": ts_cfg.get("chunk_size"),
                    "top_k": _top_k(cfg),
                    "use_4bit": quant_cfg.get("enabled", False),
                }
            )
        return page, cfg, embed, store


# ============================== 页面：问答 ==============================
def page_chat(pipeline: RAGPipeline, cfg: dict):
    st.header("💬 智能问答")
    st.caption("基于本地知识库的 RAG 问答，支持流式输出与引用展示。")

    # 初始化历史
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 显示历史
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander(f"📎 查看 {len(turn['sources'])} 条引用", expanded=False):
                    for s in turn["sources"]:
                        score_val = s.get('score')
                        score_str = f"{score_val:.3f}" if isinstance(score_val, (int, float)) else "—"
                        st.markdown(
                            f"**{s['cite']}**  \n"
                            f"<span class='kb-meta'>相似度: {score_str}</span>",
                            unsafe_allow_html=True,
                        )
                        st.code(s.get("snippet", ""))

    user_input = st.chat_input("请输入你的问题…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            sources: list = []
            t0 = time.perf_counter()
            try:
                for ev in pipeline.stream_answer(user_input):
                    etype = ev.get("event")
                    data = ev.get("data")
                    if etype == "hits":
                        sources = data or []
                    elif etype == "token":
                        full_answer += data
                        placeholder.markdown(full_answer + "▌")
                    elif etype == "done":
                        sources = data or sources
                    elif etype == "error":
                        placeholder.error(str(data))
                elapsed = (time.perf_counter() - t0) * 1000
                placeholder.markdown(full_answer or "（无回答）")
                if sources:
                    with st.expander(f"📎 查看 {len(sources)} 条引用（耗时 {elapsed:.0f} ms）", expanded=False):
                        for s in sources:
                            score_val = s.get('score')
                            score_str = f"{score_val:.3f}" if isinstance(score_val, (int, float)) else "—"
                            st.markdown(
                                f"**{s['cite']}**  \n"
                                f"<span class='kb-meta'>相似度: {score_str}</span>",
                                unsafe_allow_html=True,
                            )
                            st.code(s.get("snippet", ""))
                else:
                    st.caption(f"耗时 {elapsed:.0f} ms · 未命中知识库")
            except Exception as exc:
                placeholder.error(f"生成失败：{exc}")
                logger.exception("生成失败")

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": full_answer or "（无回答）",
                    "sources": sources,
                }
            )
            # 控制历史长度
            max_h = cfg.get("ui", {}).get("max_chat_history", 20)
            if len(st.session_state.chat_history) > max_h * 2:
                st.session_state.chat_history = st.session_state.chat_history[-max_h * 2 :]

    # 清空对话按钮
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🧹 清空对话"):
            st.session_state.chat_history = []
            st.rerun()


# ============================== 页面：上传 ==============================
def page_upload(cfg: dict, embed: EmbeddingModel, store: ChromaStore):
    st.header("📤 文档上传与入库")
    st.caption("支持 PDF、DOCX、Markdown、TXT。上传后可预览前若干分块。")

    uploaded = st.file_uploader(
        "选择文件",
        type=["pdf", "docx", "md", "markdown", "txt"],
        accept_multiple_files=True,
    )
    if uploaded:
        raw_dir = _raw_docs_dir(cfg)
        raw_dir.mkdir(parents=True, exist_ok=True)

        # 临时保存
        saved_paths = []
        for f in uploaded:
            target = raw_dir / f.name
            with open(target, "wb") as out:
                out.write(f.read())
            saved_paths.append(target)
        st.success(f"已暂存 {len(saved_paths)} 个文件到 {raw_dir}")

        if st.button("🚀 解析并入库", type="primary"):
            splitter = get_splitter(cfg)
            progress = st.progress(0.0, text="开始解析…")
            all_chunks = []
            for i, p in enumerate(saved_paths, 1):
                progress.progress(i / (len(saved_paths) + 1), text=f"解析 {p.name}…")
                docs = load_document(
                    p,
                    pdf_engine=cfg["document_loader"].get("pdf_engine", "pdfplumber"),
                    encoding=cfg["document_loader"].get("encoding", "utf-8"),
                )
                for d in docs:
                    all_chunks.extend(splitter.split_text(d.content, metadata=d.metadata))
            progress.progress(1.0, text=f"生成 {len(all_chunks)} 个分块，开始写入向量库…")
            n = store.add_chunks(all_chunks)
            progress.empty()
            st.success(f"✅ 入库完成，新增/覆盖 {n} 条分块")
            # 预览
            with st.expander("🔍 分块预览（前 10 条）", expanded=False):
                for ck in all_chunks[:10]:
                    st.markdown(
                        f"<div class='kb-card'><b>[{ck.index}]</b> {ck.metadata.get('source', '?')} "
                        f"<span class='kb-meta'>(page {ck.metadata.get('page', 1)})</span><br/>"
                        f"{ck.text[:300]}…</div>",
                        unsafe_allow_html=True,
                    )

    st.divider()
    st.subheader("📂 批量入库已有目录")
    raw_dir = _raw_docs_dir(cfg)
    st.code(str(raw_dir))
    if st.button("📥 将 data/raw 全部入库"):
        from src.document_loader import load_directory

        splitter = get_splitter(cfg)
        docs = load_directory(raw_dir, pdf_engine=cfg["document_loader"].get("pdf_engine", "pdfplumber"))
        chunks = []
        for d in docs:
            chunks.extend(splitter.split_text(d.content, metadata=d.metadata))
        st.info(f"共 {len(docs)} 个文档，{len(chunks)} 个分块")
        if chunks:
            n = store.add_chunks(chunks)
            st.success(f"✅ 入库完成：{n} 条")


# ============================== 页面：管理 ==============================
def page_manage(cfg: dict, store: ChromaStore):
    st.header("📁 知识库管理")
    st.caption("查看来源文档、按 metadata 过滤、删除文档。")

    sources = store.list_sources()
    if not sources:
        st.info("知识库为空，请先上传文档。")
        return

    st.subheader(f"📚 来源文档（共 {len(sources)} 个文件，{store.count()} 个分块）")
    # 过滤搜索
    keyword = st.text_input("🔎 文件名过滤", "")
    if keyword:
        sources = [s for s in sources if keyword.lower() in s["source"].lower()]

    for s in sources:
        cols = st.columns([5, 1, 1])
        cols[0].markdown(f"**📄 {s['source']}**  \n<span class='kb-meta'>{s['chunks']} 个分块</span>", unsafe_allow_html=True)
        if cols[1].button("预览", key=f"prev_{s['source']}"):
            st.session_state[f"show_prev_{s['source']}"] = not st.session_state.get(f"show_prev_{s['source']}", False)
        if cols[2].button("🗑 删除", key=f"del_{s['source']}"):
            n = store.delete_by_metadata({"source": s["source"]})
            st.warning(f"已删除 {n} 条")
            st.rerun()
        if st.session_state.get(f"show_prev_{s['source']}", False):
            hits = store.collection.get(where={"source": s["source"]}, include=["documents", "metadatas"], limit=3)
            for i, (doc, meta) in enumerate(zip(hits.get("documents", []), hits.get("metadatas", []))):
                st.markdown(
                    f"<div class='kb-card'><b>第 {meta.get('page', 1)} 页</b><br/>{doc[:400]}…</div>",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.subheader("🧪 检索测试")
    q = st.text_input("输入测试查询")
    k = st.slider("Top-K", 1, 20, value=_top_k(cfg))
    if st.button("检索") and q:
        t0 = time.perf_counter()
        hits = store.query(q, top_k=k)
        dt = (time.perf_counter() - t0) * 1000
        st.caption(f"耗时 {dt:.1f} ms")
        for i, h in enumerate(hits, 1):
            with st.expander(f"[{i}] 相似度 {h.score:.3f}  ·  {h.metadata.get('source','?')}"):
                st.markdown(f"```text\n{h.text}\n```")


# ============================== 页面：设置 ==============================
def page_settings(cfg: dict):
    st.header("⚙️ 系统设置")
    st.caption("查看与（部分）运行时调整配置。修改配置需要重启 Streamlit 才能生效。")

    st.subheader("Embedding")
    st.json(cfg.get("embedding", {}))
    st.subheader("LLM")
    st.json(cfg.get("llm", {}))
    st.subheader("Text Splitter")
    st.json(cfg.get("text_splitter", {}))
    st.subheader("Retrieval")
    st.json(cfg.get("retrieval", {}))
    st.subheader("Reranker")
    st.json(cfg.get("reranker", {}))
    st.subheader("RAG")
    st.json(cfg.get("rag", {}))

    st.divider()
    st.subheader("📦 显存/内存估算")
    llm_cfg = cfg.get("llm", {})
    quant_cfg = llm_cfg.get("quantization", {}) if isinstance(llm_cfg.get("quantization"), dict) else {}
    quant = bool(quant_cfg.get("enabled", False))
    gb = LocalLLM.estimate_memory_gb(llm_cfg.get("model_name", ""), quant=quant)
    st.metric("LLM 预计占用", f"{gb} GB", help="基于模型参数量粗略估算，未含 KV cache 与上下文")

    st.info(
        "如需切换模型/量化，请修改 `config/config.yaml`，然后重启 Streamlit。"
        "或者在 `.env` 中设置 `LLM_MODEL_PATH`、`USE_4BIT` 等环境变量。"
    )


# ============================== 页面：评估 ==============================
def page_evaluate(pipeline: RAGPipeline, cfg: dict):
    st.header("📈 评估")
    st.caption("在小型评估集上跑检索命中、关键词覆盖、响应时间三项指标。")

    dataset_path = resolve_path(cfg.get("evaluation", {}).get("dataset_path", "data/eval/eval_set.jsonl"))
    st.code(f"评估集路径：{dataset_path}")
    if not dataset_path.exists():
        st.warning("评估集不存在，请创建 jsonl 文件，每行：{\"question\":..., \"expected_sources\":[...], \"expected_keywords\":[...]}")
        st.code(
            '{"question": "什么是 RAG？", "expected_sources": ["intro.pdf"], "expected_keywords": ["检索", "生成"]}\n'
            '{"question": "请简述 RAG 的流程", "expected_keywords": ["向量", "提示"]}\n',
            language="json",
        )
        return

    if st.button("▶️ 运行评估", type="primary"):
        from scripts.evaluate import load_eval_dataset, evaluate as run_eval

        items = load_eval_dataset(dataset_path)
        with st.spinner(f"运行评估：{len(items)} 条样本…"):
            summary = run_eval(pipeline, items)

        col1, col2, col3 = st.columns(3)
        col1.metric("样本数", summary.get("samples", 0))
        col2.metric("检索命中率", f"{summary.get('retrieval_hit_rate', 0):.2%}")
        col3.metric("平均响应时间", f"{summary.get('avg_latency_ms', 0):.0f} ms")
        if summary.get("avg_keyword_coverage") is not None:
            st.metric("平均关键词覆盖率", f"{summary['avg_keyword_coverage']:.2%}")

        with st.expander("📋 详情", expanded=False):
            st.json(summary)


# ============================== 主入口 ==============================
def main():
    config_path = "config/config.yaml"
    page, cfg, embed, store = render_sidebar(config_path)
    pipeline = load_pipeline(config_path)

    if page.startswith("💬"):
        page_chat(pipeline, cfg)
    elif page.startswith("🤖"):
        page_agent(cfg)
    elif page.startswith("📤"):
        page_upload(cfg, embed, store)
    elif page.startswith("📁"):
        page_manage(cfg, store)
    elif page.startswith("📜"):
        page_versions()
    elif page.startswith("🕸"):
        page_kg(cfg)
    elif page.startswith("⚙️"):
        page_settings(cfg)
    elif page.startswith("📈"):
        page_evaluate(pipeline, cfg)
    elif page.startswith("📊"):
        page_analytics(cfg)


if __name__ == "__main__":
    main()