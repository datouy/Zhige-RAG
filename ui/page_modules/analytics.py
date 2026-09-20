"""📊 指标可视化页面。

从 ``data/eval/report.json`` 与 ``data/eval/reports/*.json`` 读取评估结果，
用 Plotly 渲染顶层指标卡 + 三个图表 + 详情表格。

设计要点：
- 与现有 UI 风格一致（中文文案、单文件多 radio）
- 不引入额外重型依赖，仅 ``plotly``（已在 requirements 中）
- 文件缺失/异常时给出友好提示，而不是抛栈
- ``render_page(cfg)`` 为唯一对外入口，方便 ``ui/app.py`` 侧栏调用
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import get_logger, resolve_path

logger = get_logger("ui.analytics")


# ======================================================================
#  数据加载（容错）
# ======================================================================
def _load_latest_report(report_path: Path) -> Optional[Dict[str, Any]]:
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("读取 %s 失败：%s", report_path, exc)
        return None


def _load_history_index(reports_dir: Path) -> List[Dict[str, Any]]:
    idx_path = reports_dir / "index.json"
    if not idx_path.exists():
        return []
    try:
        data = json.loads(idx_path.read_text(encoding="utf-8") or "[]")
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.warning("读取 %s 失败：%s", idx_path, exc)
    return []


def _load_history_snapshots(reports_dir: Path, max_n: int = 50) -> List[Dict[str, Any]]:
    """读 ``index.json``；若没有则扫描 ``reports/*.json`` 兜底。

    优先用 ``index.json``（轻量），缺失时再读完整历史文件。
    """
    entries = _load_history_index(reports_dir)
    if entries:
        return entries[-max_n:]

    if not reports_dir.exists():
        return []
    snapshots: List[Dict[str, Any]] = []
    for p in sorted(reports_dir.glob("*.json")):
        try:
            snap = json.loads(p.read_text(encoding="utf-8") or "{}")
            snapshots.append(
                {
                    "ts": snap.get("ts") or p.stem,
                    "samples": snap.get("samples", 0),
                    "retrieval_hit_rate": snap.get("retrieval_hit_rate", 0.0),
                    "avg_keyword_coverage": snap.get("avg_keyword_coverage"),
                    "avg_latency_ms": snap.get("avg_latency_ms"),
                    "ndcg_at_5": snap.get("ndcg_at_5"),
                    "mrr": snap.get("mrr"),
                    "avg_ttft_ms": snap.get("avg_ttft_ms"),
                    "avg_tokens_per_sec": snap.get("avg_tokens_per_sec"),
                    "archive_path": str(p.relative_to(reports_dir.parent)),
                }
            )
        except Exception:
            continue
    return snapshots[-max_n:]


# ======================================================================
#  渲染辅助
# ======================================================================
def _format_pct(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}%}"
    except Exception:
        return "—"


def _format_float(v: Optional[float], unit: str = "") -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}{unit}"
    except Exception:
        return "—"


def _safe_dataframe(details: List[Dict[str, Any]]):
    """为 st.dataframe 构造一个轻量数据视图。"""
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return None
    rows: List[Dict[str, Any]] = []
    for d in details or []:
        rows.append(
            {
                "#": d.get("index"),
                "问题": d.get("question", ""),
                "命中": "✅" if d.get("retrieval_hit") else "❌",
                "关键词覆盖率": d.get("keyword_coverage"),
                "NDCG@5": d.get("ndcg_at_5"),
                "MRR": d.get("mrr"),
                "延迟 (ms)": d.get("latency_ms"),
                "TTFT (ms)": d.get("ttft_ms"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["#", "问题", "命中", "关键词覆盖率", "NDCG@5", "MRR", "延迟 (ms)", "TTFT (ms)"])
    return pd.DataFrame(rows)


# ======================================================================
#  图表构造
# ======================================================================
def _fig_latency_distribution(details: List[Dict[str, Any]]):
    """逐题延迟分布：柱状图 + 箱线图并列。"""
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore

    indices: List[int] = []
    latencies: List[float] = []
    for d in details or []:
        if "latency_ms" in d:
            indices.append(int(d.get("index", len(indices) + 1)))
            latencies.append(float(d["latency_ms"]))

    if not latencies:
        return None

    fig = make_subplots(rows=1, cols=2, subplot_titles=("逐题延迟（柱状图）", "整体延迟分布（箱线图）"))
    fig.add_trace(
        go.Bar(x=indices, y=latencies, name="延迟", marker_color="#2563eb", hovertemplate="#%{x}<br>%{y:.0f} ms<extra></extra>"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Box(y=latencies, name="延迟", marker_color="#10b981", boxmean=True),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="问题序号", row=1, col=1)
    fig.update_yaxes(title_text="latency (ms)", row=1, col=1)
    fig.update_yaxes(title_text="latency (ms)", row=1, col=2)
    fig.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=40, b=30))
    return fig


def _fig_hit_scatter(details: List[Dict[str, Any]]):
    """命中率 vs 关键词覆盖率散点（颜色按是否命中区分）。"""
    import plotly.graph_objects as go  # type: ignore

    hits_x: List[float] = []
    hits_y: List[float] = []
    miss_x: List[float] = []
    miss_y: List[float] = []
    for d in details or []:
        # 用 1（命中）/0（未命中）作为 x 轴离散值，便于观察
        x = 1 if d.get("retrieval_hit") else 0
        kw = d.get("keyword_coverage")
        y = float(kw) if isinstance(kw, (int, float)) else None
        if y is None:
            continue
        if x == 1:
            hits_x.append(x + (len(hits_x) - sum(1 for v in hits_x if v == 1)) * 0.04)  # 抖动
            hits_y.append(y)
        else:
            miss_x.append(x + (len(miss_x) - sum(1 for v in miss_x if v == 0)) * 0.04)
            miss_y.append(y)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hits_x,
            y=hits_y,
            mode="markers",
            name="命中",
            marker=dict(color="#10b981", size=12, symbol="circle"),
            hovertemplate="命中<br>覆盖率=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=miss_x,
            y=miss_y,
            mode="markers",
            name="未命中",
            marker=dict(color="#ef4444", size=12, symbol="x"),
            hovertemplate="未命中<br>覆盖率=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["未命中", "命中"],
        range=[-0.5, 1.5],
        title="检索命中",
    )
    fig.update_yaxes(title="关键词覆盖率", range=[0, 1.05])
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=30))
    return fig


def _fig_history_trend(history: List[Dict[str, Any]]):
    """历史报告对比：命中率 / 关键词覆盖率 / 延迟 双 y 轴。"""
    import plotly.graph_objects as go  # type: ignore

    if not history:
        return None

    xs = [h.get("ts", "") for h in history]

    hit_rate = [float(h.get("retrieval_hit_rate") or 0.0) for h in history]
    kw_cov = [float(h.get("avg_keyword_coverage") or 0.0) for h in history]
    latency = [float(h.get("avg_latency_ms") or 0.0) for h in history]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=hit_rate,
            mode="lines+markers",
            name="命中率",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=8),
            yaxis="y1",
            hovertemplate="%{x}<br>命中率=%{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=kw_cov,
            mode="lines+markers",
            name="关键词覆盖率",
            line=dict(color="#10b981", width=2),
            marker=dict(size=8),
            yaxis="y1",
            hovertemplate="%{x}<br>覆盖率=%{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=latency,
            mode="lines+markers",
            name="平均延迟 (ms)",
            line=dict(color="#f97316", width=2, dash="dot"),
            marker=dict(size=8, symbol="diamond"),
            yaxis="y2",
            hovertemplate="%{x}<br>延迟=%{y:.0f}ms<extra></extra>",
        )
    )

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=30, b=30),
        yaxis=dict(title="比率 (0~1)", range=[0, 1.05]),
        yaxis2=dict(title="延迟 (ms)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ======================================================================
#  HTML 导出
# ======================================================================
def _build_export_html(latest: Optional[Dict[str, Any]], history: List[Dict[str, Any]]) -> str:
    """生成一份"指标快照"HTML，便于下载留档。"""
    summary_rows = ""
    if latest:
        summary_rows = "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in latest.items()
            if k != "details"
        )
    history_rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(h.get(k, '')))}</td>" for k in ("ts", "samples", "retrieval_hit_rate", "avg_keyword_coverage", "avg_latency_ms", "ndcg_at_5", "mrr", "avg_ttft_ms", "avg_tokens_per_sec"))
        + "</tr>"
        for h in history
    )
    return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>指标可视化快照</title>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#111827;}}
 h1{{color:#2563eb;}}
 table{{border-collapse:collapse;width:100%;margin:12px 0;}}
 th,td{{border:1px solid #e5e7eb;padding:6px 10px;text-align:left;font-size:13px;}}
 th{{background:#f6f8fa;}}
 .meta{{color:#6b7280;font-size:12px;}}
</style></head>
<body>
<h1>📊 RAG 评估指标快照</h1>
<p class="meta">导出时间：{html.escape(str(__import__('datetime').datetime.now()))}</p>

<h2>最新一次评估（summary）</h2>
<table>{summary_rows or '<tr><td>暂无</td></tr>'}</table>

<h2>历史评估趋势</h2>
<table>
<thead><tr>
<th>时间戳</th><th>样本数</th><th>命中率</th><th>关键词覆盖率</th><th>延迟(ms)</th>
<th>NDCG@5</th><th>MRR</th><th>TTFT(ms)</th><th>tokens/s</th>
</tr></thead>
<tbody>{history_rows or '<tr><td colspan="9">暂无历史</td></tr>'}</tbody>
</table>
</body></html>
"""


# ======================================================================
#  对外入口
# ======================================================================
def render_page(cfg: dict) -> None:
    """供 ``ui/app.py`` 侧栏调用的页面函数。"""
    import streamlit as st  # type: ignore

    st.header("📊 指标可视化")
    st.caption("基于 data/eval/report.json 与历史存档，提供检索质量与响应延迟的多维分析。")

    eval_dir = resolve_path((cfg.get("paths", {}) or {}).get("eval", "data/eval"))
    reports_dir = eval_dir / "reports"
    latest_path = eval_dir / "report.json"

    latest = _load_latest_report(latest_path)
    history = _load_history_snapshots(reports_dir, max_n=50)

    if latest is None and not history:
        st.warning(
            "尚未找到任何评估报告。请先运行：\n\n"
            "```\npython scripts/evaluate.py --synthesize-demo\n```"
            "或在 UI 「📈 评估」页中点 “▶️ 运行评估”。"
        )
        if st.button("📥 跳到评估页说明"):
            st.info("见左侧导航的「📈 评估」。")
        return

    details: List[Dict[str, Any]] = (latest or {}).get("details", []) or []

    # ---------- 顶层 4 个指标卡 ----------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本数", latest.get("samples", len(details)) if latest else 0)
    c2.metric("平均命中率", _format_pct(latest.get("retrieval_hit_rate")) if latest else "—")
    c3.metric("平均关键词覆盖率", _format_pct(latest.get("avg_keyword_coverage")) if latest else "—")
    c4.metric(
        "平均延迟 (ms)",
        f"{float(latest.get('avg_latency_ms')):.0f}" if latest and latest.get("avg_latency_ms") is not None else "—",
    )

    # ---------- 第二排：NDCG/MRR/TTFT/tokens/s 副指标卡 ----------
    if latest:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("NDCG@5", _format_float(latest.get("ndcg_at_5")),
                  help="排序质量分（0~1）：正确来源是否排在结果前列，越接近 1 越好")
        s2.metric("MRR", _format_float(latest.get("mrr")),
                  help="首个正确来源的平均排名倒数（0~1）：正确来源平均出现在第几位")
        s3.metric(
            "平均 TTFT (ms)",
            f"{float(latest.get('avg_ttft_ms')):.0f}" if latest.get("avg_ttft_ms") is not None else "—",
            help="Time To First Token：从提问到出现第一个字的等待时间，越短感觉响应越快",
        )
        s4.metric(
            "平均 tokens/s",
            f"{float(latest.get('avg_tokens_per_sec')):.2f}" if latest.get("avg_tokens_per_sec") is not None else "—",
            help="回答的生成速度：数值越大，文字出现得越流畅",
        )

    st.divider()

    # ---------- 三个图表 ----------
    if latest:
        st.subheader("📈 逐题延迟分布")
        fig_lat = _fig_latency_distribution(details)
        if fig_lat is not None:
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            st.info("没有逐题延迟数据。")

        st.subheader("🎯 命中率 vs 关键词覆盖率")
        fig_scatter = _fig_hit_scatter(details)
        if fig_scatter is not None:
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("没有命中/覆盖率数据。")
    else:
        st.info("暂无最新报告，仅展示历史趋势。")

    st.subheader("🕒 历史报告对比")
    fig_hist = _fig_history_trend(history)
    if fig_hist is not None:
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("没有历史报告（请运行评估后查看）。")

    st.divider()

    # ---------- 详情表 + 搜索 ----------
    st.subheader("📋 评估详情")
    if details:
        query = st.text_input("🔎 按问题搜索（支持中文）", "")
        filtered = details
        if query:
            q_lower = query.lower()
            filtered = [d for d in details if q_lower in str(d.get("question", "")).lower()]
        df = _safe_dataframe(filtered)
        if df is not None:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("未安装 pandas，退化为 JSON 展示。")
            st.json(filtered[:50])
    else:
        st.info("最新报告中没有逐题详情。")

    # ---------- 导出 ----------
    st.divider()
    st.subheader("💾 导出")
    html_text = _build_export_html(latest, history)
    st.download_button(
        label="📥 下载当前快照为 HTML",
        data=html_text.encode("utf-8"),
        file_name=f"rag_metrics_snapshot_{__import__('datetime').datetime.now().strftime('%Y%m%d-%H%M%S')}.html",
        mime="text/html",
        use_container_width=True,
    )
    with st.expander("👀 在线预览（HTML 源码嵌入）", expanded=False):
        st.components.v1.html(html_text, height=420, scrolling=True)
