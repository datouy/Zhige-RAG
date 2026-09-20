"""🕸 知识图谱 Streamlit 页面。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.kg import KGExtractor, GraphRAG, GraphRetriever, create_kg_store
from src.utils import get_logger, load_config, resolve_path

logger = get_logger("ui.kg")


def _get_kg_store():
    cfg = load_config("config/config.yaml")
    kg_cfg = cfg.get("knowledge_graph", {}) or {}
    if not kg_cfg:
        kg_cfg = {"enabled": True, "backend": "sqlite", "sqlite_path": "data/kg.db"}
    return create_kg_store(kg_cfg)


@st.cache_resource(show_spinner=False)
def _get_kg_store_cached(_cfg_hash: int = 0):
    return _get_kg_store()


@st.cache_resource(show_spinner=False)
def _get_kg_retriever():
    return GraphRetriever(_get_kg_store())


@st.cache_resource(show_spinner=False)
def _get_kg_extractor():
    from src.llm import LocalLLM
    from src.rag_pipeline import RAGPipeline

    try:
        pipeline = RAGPipeline.from_config("config/config.yaml", lazy_llm=False)
        llm = pipeline.llm
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM 加载失败：%s", exc)
        return None
    if llm is None:
        return None
    cfg = load_config("config/config.yaml")
    ex_cfg = (cfg.get("knowledge_graph", {}) or {}).get("extractor", {}) or {}
    return KGExtractor(
        llm=llm,
        max_entities_per_chunk=int(ex_cfg.get("max_entities_per_chunk", 20)),
        max_relations_per_chunk=int(ex_cfg.get("max_relations_per_chunk", 30)),
    )


@st.cache_resource(show_spinner=False)
def _get_graph_rag():
    try:
        from src.embeddings import EmbeddingModel
        from src.rag_pipeline import RAGPipeline
        from src.vector_store import ChromaStore, hybrid_kwargs

        cfg = load_config("config/config.yaml")
        emb_cfg = cfg.get("embedding", {}) or {}
        embedding = EmbeddingModel(
            model_name=emb_cfg.get("model_name", "BAAI/bge-small-zh-v1.5"),
            device=emb_cfg.get("device", "auto"),
            batch_size=emb_cfg.get("batch_size", 16),
            max_seq_length=emb_cfg.get("max_seq_length", 512),
            normalize=emb_cfg.get("normalize_embeddings", True),
            cache_dir=emb_cfg.get("cache_dir"),
            local_files_only=emb_cfg.get("local_files_only", False),
        )
        vs = ChromaStore(
            persist_directory=cfg["vector_store"]["persist_directory"],
            collection_name=cfg["vector_store"].get("collection_name", "chinese_rag_kb"),
            embedding_model=embedding,
            distance_fn=cfg["vector_store"].get("distance_fn", "cosine"),
            **hybrid_kwargs(cfg.get("vector_store", {})),
        )
        kg_store = _get_kg_store()
        retriever = GraphRetriever(kg_store)
        extractor = _get_kg_extractor()
        try:
            pipeline = RAGPipeline.from_config("config/config.yaml", lazy_llm=False)
            llm = pipeline.llm
        except Exception:
            llm = None
        if llm is None:
            return None
        return GraphRAG(vector_store=vs, kg_store=kg_store, extractor=extractor, retriever=retriever, llm=llm)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GraphRAG 初始化失败：%s", exc)
        return None


def _show_stats(store) -> None:
    stats = store.count()
    cols = st.columns(2)
    cols[0].metric("实体数量", stats.get("entities", 0))
    cols[1].metric("关系数量", stats.get("relations", 0))


def _list_entities(store, type_filter: Optional[str] = None, search: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    if search:
        entities = store.find_entities_by_name(search, fuzzy=True, limit=limit)
    else:
        entities = []
        import sqlite3
        conn = sqlite3.connect(store.db_path)
        cur = conn.cursor()
        query = "SELECT name, type, description, aliases FROM entities"
        params: List[Any] = []
        if type_filter:
            query += " WHERE type = ?"
            params.append(type_filter)
        query += " LIMIT ?"
        params.append(limit)
        cur.execute(query, params)
        for row in cur.fetchall():
            entities.append(
                {
                    "name": row[0],
                    "type": row[1] or "Other",
                    "description": row[2] or "",
                    "aliases": json.loads(row[3] or "[]"),
                }
            )
        conn.close()
    return entities


def _entity_detail(store, name: str) -> None:
    ent = store.get_entity(name)
    if ent is None:
        st.warning(f"未找到实体：{name}")
        return
    st.markdown(f"### 🟢 {ent.name}")
    st.caption(f"类型: `{ent.type}`")
    if ent.description:
        st.markdown(f"> {ent.description}")
    if ent.aliases:
        st.markdown(f"**别名：** {', '.join(ent.aliases)}")
    relations = store.get_relations(ent.name, direction="both")
    if relations:
        st.markdown(f"#### 🔗 关系网络（{len(relations)} 条）")
        for r in relations:
            direction_icon = "➡️" if r.source == ent.name else "⬅️"
            st.markdown(
                f"- {direction_icon} `{r.source}` -[{r.type}]-> `{r.target}`"
                + (f" _{r.description}_" if r.description else "")
            )


def render_page(cfg: dict) -> None:  # noqa: D401
    """渲染知识图谱页面。"""
    st.header("🕸 知识图谱")
    st.caption("基于 LLM 的实体/关系抽取与图谱检索。支持 SQLite/Neo4j 两种后端。")

    try:
        store = _get_kg_store()
    except Exception as exc:
        st.error(f"图谱存储初始化失败：{exc}")
        return

    tabs = st.tabs(["📊 统计", "🧬 实体管理", "🔍 图谱检索", "✍️ 抽取器", "💬 GraphRAG 问答"])

    with tabs[0]:
        st.subheader("图谱统计")
        _show_stats(store)
        _kg_counts = {}
        try:
            _kg_counts = store.count() or {}
        except Exception:
            pass
        if int(_kg_counts.get("entities", 0) or 0) <= 0:
            # C3: 空状态引导——小白需要知道"为什么是空的、怎么构建"
            st.info(
                "📖 知识图谱目前是空的。**这不影响正常问答**；若要启用"
                "\"图谱增强问答\"（回答实体之间关系类问题更擅长），请在命令行运行 "
                "`python scripts/build_kg.py` 构建图谱（需要先关闭占用 GPU 的程序）。"
            )
        else:
            if st.button("🗑 清空图谱（谨慎操作，将删除全部实体与关系）"):
                store.clear()
                st.success("已清空")
                st.rerun()

    with tabs[1]:
        st.subheader("实体列表")
        col1, col2 = st.columns(2)
        type_filter = col1.text_input("类型过滤", value="")
        search_query = col2.text_input("搜索（名称/描述）", value="")
        entities = _list_entities(store, type_filter or None, search_query or None, limit=200)
        if not entities:
            st.info("图谱为空。请先在「抽取器」标签抽取文本，或使用 `scripts/build_kg.py` 构建。")
        else:
            selected = st.selectbox(
                "选择实体查看详情",
                options=[e["name"] for e in entities],
                key="kg_entity_select",
            )
            st.markdown(f"共 {len(entities)} 个实体")
            st.dataframe(entities, use_container_width=True, hide_index=True)
            if selected:
                st.divider()
                _entity_detail(store, selected)

    with tabs[2]:
        st.subheader("图谱检索")
        q = st.text_input("查询关键词（实体名/短语）", value="")
        col1, col2 = st.columns(2)
        top_k = col1.slider("Top-K 实体", 1, 20, value=5)
        hops = col2.slider("多跳扩展数", 0, 5, value=2)
        if q and st.button("检索", key="kg_search_btn"):
            retriever = _get_kg_retriever()
            result = retriever.search(query=q, top_k_entities=top_k, hops=hops)
            st.markdown(f"**命中实体：** {len(result.get('entities', []))}")
            st.markdown(f"**命中关系：** {len(result.get('relations', []))}")
            for e in result.get("entities", [])[:50]:
                st.markdown(f"- 🔹 **{e.name}** ({e.type}) {e.description}")
            for r in result.get("relations", [])[:50]:
                st.markdown(
                    f"- 🔗 `{r.source}` -[{r.type}]-> `{r.target}`"
                    + (f" _{r.description}_" if r.description else "")
                )

    with tabs[3]:
        st.subheader("文本抽取器")
        st.caption("输入文本 → LLM 抽取 → 入库。需 LLM 已加载。")
        text = st.text_area("待抽取文本", height=200)
        source_doc = st.text_input("来源文档（可选）", value="")
        if st.button("抽取并入库") and text.strip():
            extractor = _get_kg_extractor()
            if extractor is None:
                st.error("LLM 未就绪，请确认 LLM 已加载。")
            else:
                with st.spinner("抽取中…"):
                    entities, relations = extractor.extract(text=text, source_doc=source_doc)
                    n_e = store.upsert_entities(entities) if entities else 0
                    n_r = store.upsert_relations(relations) if relations else 0
                st.success(f"抽取完成：实体 {len(entities)}（写入 {n_e}），关系 {len(relations)}（写入 {n_r}）")
                if entities:
                    st.markdown("**实体：**")
                    for e in entities:
                        st.markdown(f"- 🔹 {e.name} ({e.type}) {e.description}")
                if relations:
                    st.markdown("**关系：**")
                    for r in relations:
                        st.markdown(f"- 🔗 {r.source} -[{r.type}]-> {r.target}")

    with tabs[4]:
        st.subheader("GraphRAG 问答")
        st.caption("向量检索 + 图谱扩展 → LLM 回答")
        graph_rag = _get_graph_rag()
        if graph_rag is None:
            st.warning("GraphRAG 初始化失败（可能 LLM 未加载）。")
        else:
            q = st.text_input("问题", value="", key="kg_graph_rag_q")
            col1, col2 = st.columns(2)
            top_k = col1.slider("向量 Top-K", 1, 20, value=4, key="gr_topk")
            graph_hops = col2.slider("图谱扩展跳数", 0, 5, value=2, key="gr_hops")
            if st.button("开始 GraphRAG") and q.strip():
                with st.spinner("生成中…"):
                    res = graph_rag.query(question=q, top_k=top_k, graph_hops=graph_hops)
                st.markdown("### 💡 回答")
                st.markdown(res.get("answer", "（无回答）"))
                sources = res.get("sources", []) or []
                if sources:
                    st.markdown("### 📎 文档来源")
                    for s in sources:
                        with st.expander(f"[{s.get('score', 0):.3f}] {s.get('metadata', {}).get('source', '?')}"):
                            st.markdown(s.get("text", ""))
                gc = res.get("graph_context", {}) or {}
                triples = gc.get("triples", []) or []
                if triples:
                    st.markdown("### 🕸 图谱三元组")
                    for t in triples[:30]:
                        st.markdown(f"- `{t.subject}` -[{t.predicate}]-> `{t.object}`")