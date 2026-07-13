"""文档版本管理 UI 页面。

功能：
- 查看所有已版本化文档列表
- 查看某文档的版本历史
- 对比两个版本的差异（chunk-level diff）
- 差异可视化（新增/删除/修改高亮）
- 清理旧版本
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from scripts.version_manager import (
    compute_diff,
    list_all_documents,
    list_versions,
    prune_versions,
)

# ----------------------------------------------------------------------
# 样式
# ----------------------------------------------------------------------
DIFF_CSS = """
<style>
    .diff-added {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
    }
    .diff-removed {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
    }
    .diff-modified {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
    }
    .diff-summary {
        display: flex;
        gap: 16px;
        margin: 16px 0;
        padding: 12px;
        background: #f6f8fa;
        border-radius: 8px;
    }
    .diff-stat {
        text-align: center;
        min-width: 60px;
    }
    .diff-stat-num {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .diff-stat-label {
        font-size: 0.8rem;
        color: #666;
    }
    .diff-added .diff-stat-num { color: #28a745; }
    .diff-removed .diff-stat-num { color: #dc3545; }
    .diff-modified .diff-stat-num { color: #ffc107; }
</style>
"""
st.markdown(DIFF_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _format_timestamp(ts: str) -> str:
    """将 ISO8601 时间戳转为可读格式。"""
    if not ts:
        return "N/A"
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def _truncate_text(text: str, max_len: int = 200) -> str:
    """截断长文本。"""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ----------------------------------------------------------------------
# 页面渲染
# ----------------------------------------------------------------------
def render_page():
    st.header("📜 文档版本管理")
    st.caption("查看文档版本历史、对比差异、可视化变更。")

    # 加载文档列表
    all_docs = list_all_documents()

    if not all_docs:
        st.info("尚无任何已版本化的文档。使用 `python scripts/ingest.py <path> --version` 入库文档以创建版本。")
        st.code("python scripts/ingest.py data/raw --version", language="bash")
        return

    # 文档选择
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_doc = st.selectbox("选择文档", options=all_docs, index=0)
    with col2:
        st.markdown("")  # 占位对齐
        if st.button("🔄 刷新", help="刷新版本列表"):
            st.rerun()

    if not selected_doc:
        return

    # 加载版本列表
    versions = list_versions(selected_doc)

    if not versions:
        st.warning(f"文档 {selected_doc} 没有版本记录。")
        return

    # 版本列表展示
    st.subheader(f"📋 版本历史 ({len(versions)} 个版本)")

    # 表格头
    header_cols = st.columns([1, 2, 1, 1, 1, 1])
    headers = ["版本", "时间", "分块数", "字符数", "来源", "操作"]
    for col, h in zip(header_cols, headers):
        col.markdown(f"**{h}**")

    st.divider()

    # 版本行
    version_options = [v["version"] for v in versions]
    selected_for_compare: list = []

    for v in versions:
        row_cols = st.columns([1, 2, 1, 1, 1, 2])
        row_cols[0].write(v["version"])
        row_cols[1].write(_format_timestamp(v.get("timestamp", "")))
        row_cols[2].write(str(v.get("num_chunks", "")))
        row_cols[3].write(str(v.get("total_chars", "")))
        row_cols[4].write(v.get("created_by", ""))

        # 对比按钮
        if row_cols[5].button("🔍 对比", key=f"compare_{v['version']}"):
            selected_for_compare.append(v["version"])

    # 版本对比区域
    st.divider()
    st.subheader("🔍 版本对比")

    col_a, col_b = st.columns(2)
    with col_a:
        from_ver = st.selectbox("从版本", options=version_options, index=0, key="from_ver")
    with col_b:
        # 默认选择第二个版本
        to_options = version_options
        to_ver = st.selectbox(
            "到版本",
            options=to_options,
            index=min(1, len(to_options) - 1),
            key="to_ver",
        )

    if st.button("📊 执行对比", type="primary"):
        if from_ver == to_ver:
            st.warning("请选择不同的版本进行对比。")
        else:
            with st.spinner("正在计算差异..."):
                result = compute_diff(selected_doc, from_ver, to_ver)

            if result is None:
                st.error("无法计算差异，请检查版本是否存在。")
            else:
                _render_diff_result(result)

    # 清理旧版本
    st.divider()
    st.subheader("🧹 版本清理")

    col_keep, col_btn = st.columns([1, 4])
    with col_keep:
        keep_n = st.number_input("保留最近", min_value=1, max_value=len(versions), value=3, step=1)
    with col_btn:
        st.markdown("")
        if st.button(f"🗑 删除旧版本（保留 {keep_n} 个）", help="删除所有版本，保留最新的 N 个"):
            if len(versions) <= keep_n:
                st.info(f"当前版本数 {len(versions)} <= {keep_n}，无需清理。")
            else:
                deleted = prune_versions(selected_doc, keep=keep_n)
                st.success(f"已删除 {deleted} 个旧版本，保留最近 {keep_n} 个。")
                st.rerun()


def _render_diff_result(result: dict):
    """渲染 diff 结果。"""
    summary = result.get("summary", {})
    changes = result.get("changes", [])

    # 统计摘要
    st.markdown("### 📊 差异摘要")
    summary_cols = st.columns(4)
    summary_data = [
        ("新增", summary.get("added", 0), "🟢"),
        ("删除", summary.get("removed", 0), "🔴"),
        ("修改", summary.get("modified", 0), "🟡"),
        ("不变", summary.get("unchanged", 0), "⚪"),
    ]
    for col, (label, count, icon) in zip(summary_cols, summary_data):
        col.metric(f"{icon} {label}", count)

    st.divider()

    if not changes:
        st.success("两个版本完全相同，无差异。")
        return

    # 变更类型筛选
    change_types = list(set(c["type"] for c in changes))
    type_labels = {"added": "🟢 新增", "removed": "🔴 删除", "modified": "🟡 修改"}
    selected_types = st.multiselect(
        "筛选变更类型",
        options=change_types,
        default=change_types,
        format_func=lambda t: type_labels.get(t, t),
    )

    filtered_changes = [c for c in changes if c["type"] in selected_types]

    st.markdown(f"### 📝 详细变更（共 {len(filtered_changes)} 项）")

    for i, change in enumerate(filtered_changes, 1):
        change_type = change["type"]

        if change_type == "modified":
            with st.expander(f"🟡 修改 - Chunk #{change.get('chunk_id', '?')}", expanded=True):
                col_from, col_to = st.columns(2)
                with col_from:
                    st.markdown("**旧内容**")
                    st.code(_truncate_text(change.get("from_text", ""), 300))
                with col_to:
                    st.markdown("**新内容**")
                    st.code(_truncate_text(change.get("to_text", ""), 300))
                similarity = change.get("similarity", 0)
                st.caption(f"相似度: {similarity:.1%}")

        elif change_type == "added":
            with st.expander(f"🟢 新增", expanded=True):
                st.code(_truncate_text(change.get("to_text", ""), 400))
                st.caption("这是新增的 chunk")

        elif change_type == "removed":
            with st.expander(f"🔴 删除 - Chunk #{change.get('chunk_id', '?')}", expanded=True):
                st.code(_truncate_text(change.get("from_text", ""), 400))
                st.caption("这是被删除的 chunk")

        st.divider()

    # JSON 输出
    with st.expander("📄 原始 JSON", expanded=False):
        st.json(result)
