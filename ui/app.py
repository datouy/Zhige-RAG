"""Streamlit Web UI - 知阁 · 本地知识库 系统（简化版）。

启动命令：
    streamlit run ui/app.py --server.port 8501
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

# 让 Streamlit 能从项目根目录导入 src.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ============================== 页面配置 ==============================
st.set_page_config(
    page_title="知阁 · 本地知识库",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.data_quality import run_data_pipeline
from src.document_loader import load_document
from src.document_meta import enrich_directory, enrich_file
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
from src.vector_store import ChromaStore, hybrid_kwargs

# 从独立模块导入，避免 page_modules 反向引用 ui.app 导致循环 import
from ui._pipeline import load_pipeline

logger = get_logger("ui")

# ============================== 样式 ==============================
CUSTOM_CSS = """
<style>
    .kb-card { padding: 1rem; border-radius: 0.5rem; background: #f6f8fa; border: 1px solid #e5e7eb; margin-bottom: 0.5rem;}
    .kb-cite  { color: #2563eb; font-weight: 600; }
    .kb-meta  { color: #6b7280; font-size: 0.85rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================== 全局状态 ==============================
_UI_USER_ID = "local"

_VERIF_BADGE = {
    "verified": "✅ 已通过可验证性校验（回答有引用依据）",
    "partial": "⚠️ 部分内容可验证",
    "unverified": "❌ 未通过可验证性校验（请谨慎采信）",
    "refused": "🚫 未能回答（知识库中没有足够相关内容）",
}


def _write_feedback(rating: str, query: str, answer: str, sources: list, correction: str = "") -> str | None:
    """反馈层落地。"""
    from src.db.database import SessionLocal
    from src.db.models import FeedbackRecord

    db = SessionLocal()
    try:
        rec = FeedbackRecord(
            user_id=_UI_USER_ID,
            query=query,
            answer=answer,
            sources_json=json.dumps([str(s.get("cite", "")) for s in sources], ensure_ascii=False),
            rating=rating,
            correction=correction,
            question_type="ui",
        )
        db.add(rec)
        db.commit()
        return rec.id
    except Exception as exc:
        logger.warning("反馈写库失败: %s", exc)
        db.rollback()
        return None
    finally:
        db.close()


def _render_feedback(turn: dict, key_prefix: str) -> None:
    """单条回答的反馈入口。"""
    if not turn.get("query"):
        return
    if turn.get("feedback"):
        caption = "🙏 已收到反馈：" + ("👍 有帮助" if turn["feedback"] == "helpful" else "👎 没有帮助")
        st.caption(caption)
        return
    c1, c2 = st.columns([0.05, 0.05])
    if c1.button("👍", key=f"{key_prefix}_up", help="有帮助"):
        turn["feedback_id"] = _write_feedback(
            "helpful", turn["query"], turn.get("content", ""), turn.get("sources") or []
        )
        turn["feedback"] = "helpful"
        st.rerun()
    if c2.button("👎", key=f"{key_prefix}_dn", help="没有帮助"):
        turn["feedback_id"] = _write_feedback(
            "not_helpful", turn["query"], turn.get("content", ""), turn.get("sources") or []
        )
        turn["feedback"] = "not_helpful"
        st.rerun()


@st.cache_resource(show_spinner=False)
def get_embedding_for_ui(config_path: str):
    cfg = load_config(config_path)
    cfg = apply_env_overrides(cfg)
    # 后端由 embedding.backend 决定（local / ollama / openai）。
    # 必须走工厂，否则 UI 侧仍会强制加载本地 sentence-transformers（依赖 torch）。
    from src.embeddings_provider import create_embedding

    return create_embedding(cfg.get("embedding", {}))


@st.cache_resource(show_spinner=False)
def get_vector_store(config_path: str, _embed: EmbeddingModel) -> ChromaStore:
    cfg = load_config(config_path)
    return ChromaStore(
        persist_directory=cfg["vector_store"]["persist_directory"],
        collection_name=cfg["vector_store"].get("collection_name", "chinese_rag_kb"),
        embedding_model=_embed,
        distance_fn=cfg["vector_store"].get("distance_fn", "cosine"),
        **hybrid_kwargs(cfg.get("vector_store", {})),
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
    paths = cfg.get("paths", {}) or {}
    rel = paths.get("raw_docs") or paths.get("raw_docs_dir") or "data/raw"
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
            chunk_size=sp.get("chunk_size", 400),
            chunk_overlap=sp.get("chunk_overlap", 50),
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
        st.title("📚 知阁 · 本地知识库")
        
        st.markdown(f"**知识库条目**: {store.count()}")
        
        st.divider()
        page = st.radio(
            "功能导航",
            ["💬 智能问答", "📤 上传文档", "📁 知识库管理"],
            index=0,
        )
        return page, cfg, embed, store


# ============================== 页面：问答 ==============================
def page_chat(pipeline: RAGPipeline, cfg: dict):
    st.header("💬 智能问答")
    st.caption("向你的知识库提问：回答全部来自已上传的文档，并标注出处。")

    # 清空成功提示
    if st.session_state.pop("chat_cleared_msg", False):
        st.success("对话已清空")

    # 初始化历史
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 显示历史
    for i, turn in enumerate(st.session_state.chat_history):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("verification", {}).get("status"):
                st.caption(_VERIF_BADGE.get(turn["verification"]["status"], ""))
            if turn.get("sources"):
                _srcs = turn["sources"]
                with st.expander(f"📎 引用来源（共 {len(_srcs)} 条）", expanded=False):
                    for s in _srcs[:3]:
                        score_val = s.get('score')
                        score_str = f"{score_val:.3f}" if isinstance(score_val, (int, float)) else "—"
                        st.markdown(
                            f"**{s['cite']}**  \n"
                            f"<span class='kb-meta'>相似度 {score_str}</span>",
                            unsafe_allow_html=True,
                        )
                        st.code(s.get("snippet", "")[:200])
            if turn["role"] == "assistant":
                _render_feedback(turn, key_prefix=f"hist_{i}")

    # 输入框
    preset = st.session_state.pop("pending_question", None)
    user_input = preset or st.chat_input("请输入你的问题…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input, "query": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            sources: list = []
            verif: dict = {}
            t0 = time.perf_counter()
            try:
                n_turns = int((cfg.get("memory", {}) or {}).get("session_turns", 3))
                history = [
                    {"role": t["role"], "content": t["content"]}
                    for t in st.session_state.chat_history[:-1]
                    if t.get("role") in ("user", "assistant") and t.get("content")
                ][-(n_turns * 2):]
                
                lts = getattr(pipeline, "long_term_store", None)
                if lts is not None:
                    try:
                        lts.maybe_remember_from_message(_UI_USER_ID, user_input)
                    except Exception as mem_exc:
                        logger.debug("长期记忆写入跳过: %s", mem_exc)
                        
                for ev in pipeline.stream_answer(user_input, history=history, user_id=_UI_USER_ID):
                    etype = ev.get("event")
                    data = ev.get("data")
                    if etype == "hits":
                        sources = data or []
                    elif etype == "token":
                        full_answer += data
                        placeholder.markdown(full_answer + "▌")
                    elif etype == "done":
                        sources = data or sources
                        verif = ev.get("verification") or {}
                    elif etype == "error":
                        placeholder.error(str(data))
                        
                elapsed = (time.perf_counter() - t0) * 1000
                placeholder.markdown(full_answer or "（无回答）")
                if verif.get("status"):
                    st.caption(_VERIF_BADGE.get(verif["status"], ""))
                if sources:
                    with st.expander(f"📎 引用来源（共 {len(sources)} 条，耗时 {elapsed:.0f} ms）", expanded=False):
                        for s in sources:
                            score_val = s.get('score')
                            score_str = f"{score_val:.3f}" if isinstance(score_val, (int, float)) else "—"
                            st.markdown(
                                f"**{s['cite']}**  \n"
                                f"<span class='kb-meta'>相似度 {score_str}</span>",
                                unsafe_allow_html=True,
                            )
                            st.code(s.get("snippet", "")[:200])
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
                    "verification": verif,
                    "query": user_input,
                }
            )
            _render_feedback(st.session_state.chat_history[-1], key_prefix="live_last")
            
            max_h = cfg.get("ui", {}).get("max_chat_history", 20)
            if len(st.session_state.chat_history) > max_h * 2:
                st.session_state.chat_history = st.session_state.chat_history[-max_h * 2:]

    # 清空按钮
    col1, _ = st.columns([1, 5])
    if col1.button("🧹 清空对话"):
        st.session_state.chat_history = []
        st.session_state["chat_cleared_msg"] = True
        st.rerun()


# ============================== 页面：上传 ==============================
def page_upload(cfg: dict, embed: EmbeddingModel, store: ChromaStore):
    st.header("📤 文档上传与入库")
    _max_mb = (cfg.get("ui", {}) or {}).get("max_upload_mb", 200)
    st.caption(f"支持 PDF、DOCX、Markdown、TXT；单个文件不超过 {_max_mb}MB")

    uploaded = st.file_uploader(
        "选择文件",
        type=["pdf", "docx", "md", "markdown", "txt"],
        accept_multiple_files=True,
    )
    if uploaded:
        raw_dir = _raw_docs_dir(cfg)
        raw_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for f in uploaded:
            safe_name = Path(f.name).name
            if not safe_name or ".." in f.name:
                st.warning(f"已跳过非法文件名: {f.name!r}")
                continue
            target = raw_dir / safe_name
            with open(target, "wb") as out:
                out.write(f.read())
            saved_paths.append(target)
        st.success(f"已暂存 {len(saved_paths)} 个文件到 {raw_dir}")

        if st.button("🚀 解析并入库", type="primary"):
            splitter = get_splitter(cfg)
            progress = st.progress(0.0, text="开始解析…")
            all_chunks = []
            dq_cfg = cfg.get("data_quality", {}) or {}
            for i, p in enumerate(saved_paths, 1):
                progress.progress(i / (len(saved_paths) + 1), text=f"解析 {p.name}…")
                ocr_cfg = cfg.get("document_loader", {}).get("ocr") or {}
                docs = enrich_file(
                    p, default_acl=dq_cfg.get("default_acl", "*"), ocr_cfg=ocr_cfg
                )
                if not docs:
                    docs = load_document(
                        p,
                        pdf_engine=cfg["document_loader"].get("pdf_engine", "pdfplumber"),
                        encoding=cfg["document_loader"].get("encoding", "utf-8"),
                        ocr_cfg=ocr_cfg,
                    )
                chunks, _report = run_data_pipeline(docs, cfg, splitter)
                store.delete_by_metadata({"source": p.name})
                all_chunks.extend(chunks)
            progress.progress(1.0, text=f"生成 {len(all_chunks)} 个分块，开始写入…")
            n = store.add_chunks(all_chunks)
            progress.empty()
            st.success(f"✅ 入库完成，新增/覆盖 {n} 条分块")

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
        splitter = get_splitter(cfg)
        docs = enrich_directory(raw_dir, default_acl=(cfg.get("data_quality", {}) or {}).get("default_acl", "*"))
        chunks = []
        sources = set()
        for d in docs:
            sources.add((d.metadata or {}).get("source") or "")
        chunks, _report = run_data_pipeline(docs, cfg, splitter)
        for s in sources:
            if s:
                store.delete_by_metadata({"source": s})
        st.info(f"共 {len(docs)} 个文档小节，{len(chunks)} 个分块")
        if chunks:
            n = store.add_chunks(chunks)
            st.success(f"✅ 入库完成：{n} 条")


# ============================== 页面：管理 ==============================
def page_manage(cfg: dict, store: ChromaStore):
    st.header("📁 知识库管理")
    st.caption("管理你上传的文档：查看、预览、删除")

    sources = store.list_sources()
    if not sources:
        st.info("知识库为空，请先上传文档")
        return

    st.subheader(f"📚 来源文档（共 {len(sources)} 个文件，{store.count()} 个分块）")
    keyword = st.text_input("🔎 按文件名筛选", "")
    if keyword:
        sources = [s for s in sources if keyword.lower() in s["source"].lower()]

    for s in sources:
        cols = st.columns([5, 1, 1])
        cols[0].markdown(f"**📄 {s['source']}**  \n<span class='kb-meta'>{s['chunks']} 个分块</span>", unsafe_allow_html=True)
        if cols[1].button("预览", key=f"prev_{s['source']}"):
            st.session_state[f"show_prev_{s['source']}"] = not st.session_state.get(f"show_prev_{s['source']}", False)
        if cols[2].button("🗑 删除", key=f"del_{s['source']}"):
            st.session_state[f"confirm_del_{s['source']}"] = True
        if st.session_state.get(f"confirm_del_{s['source']}"):
            st.warning(f"⚠️ 即将删除《{s['source']}》的全部 {s['chunks']} 个分块！")
            _ok = st.checkbox("我确认要删除", key=f"delchk_{s['source']}")
            _c1, _c2 = st.columns(2)
            if _c1.button("确认删除", key=f"delyes_{s['source']}", disabled=not _ok, type="primary"):
                n = store.delete_by_metadata({"source": s["source"]})
                st.session_state.pop(f"confirm_del_{s['source']}", None)
                st.success(f"已删除 {n} 条")
                st.rerun()
            if _c2.button("取消", key=f"delno_{s['source']}"):
                st.session_state.pop(f"confirm_del_{s['source']}", None)
                st.rerun()
        if st.session_state.get(f"show_prev_{s['source']}", False):
            hits = store.collection.get(where={"source": s["source"]}, include=["documents", "metadatas"], limit=3)
            for i, (doc, meta) in enumerate(zip(hits.get("documents", []), hits.get("metadatas", []))):
                st.markdown(
                    f"<div class='kb-card'><b>第 {meta.get('page', 1)} 页</b><br/>{doc[:400]}…</div>",
                    unsafe_allow_html=True,
                )


# ============================== 主入口 ==============================
def main():
    config_path = "config/config.yaml"
    page, cfg, embed, store = render_sidebar(config_path)
    pipeline = load_pipeline(config_path)

    if page.startswith("💬"):
        page_chat(pipeline, cfg)
    elif page.startswith("📤"):
        page_upload(cfg, embed, store)
    elif page.startswith("📁"):
        page_manage(cfg, store)


if __name__ == "__main__":
    main()
