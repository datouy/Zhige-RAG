"""多租户 UI 模块 - 用户引导与知识库管理。"""
from __future__ import annotations
import streamlit as st
from typing import Optional

# 页面配置已在 app.py 的 main() 函数中统一处理
# st.set_page_config(page_title="知阁 · 本地知识库 - 多知识库", page_icon="📚", layout="wide")


def init_session_state():
    """初始化 session state。"""
    defaults = {
        "onboarding_step": 1,
        "current_kb": None,
        "user_kbs": [],
        "show_new_kb_dialog": False,
        "is_logged_in": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ======================== 知识库选择器 ========================

def render_kb_selector():
    """渲染知识库选择器（侧边栏顶部）。"""
    if not st.session_state.get("is_logged_in"):
        return

    st.sidebar.title("📚 知识库")

    # 刷新按钮
    if st.sidebar.button("🔄 刷新列表"):
        st.rerun()

    kb_list = st.session_state.get("user_kbs", [])

    if not kb_list:
        st.sidebar.info("暂无知识库，请先创建一个")
        if st.sidebar.button("➕ 创建第一个知识库"):
            st.session_state.show_new_kb_dialog = True
            st.rerun()
        return

    selected = st.sidebar.selectbox(
        "当前知识库",
        kb_list,
        index=kb_list.index(st.session_state.current_kb) if st.session_state.current_kb in kb_list else 0,
        key="kb_selector",
    )
    st.session_state.current_kb = selected

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("➕ 新建", use_container_width=True):
            st.session_state.show_new_kb_dialog = True
            st.rerun()
    with col2:
        if st.button("🗑️ 删除", use_container_width=True):
            st.warning("确认删除当前知识库？", icon="⚠️")


# ======================== 新建知识库对话框 ========================

def render_new_kb_dialog():
    """渲染新建知识库对话框（expander 形式）。"""
    if not st.session_state.get("show_new_kb_dialog"):
        return

    with st.expander("📝 创建新知识库", expanded=True):
        kb_name = st.text_input(
            "知识库名称",
            placeholder="例如：产品文档、技术方案、客服FAQ",
            key="new_kb_name",
        )
        kb_desc = st.text_area(
            "描述（可选）",
            placeholder="简单描述这个知识库的用途...",
            key="new_kb_desc",
            max_chars=200,
        )
        kb_type = st.selectbox(
            "知识库类型",
            ["通用文档", "技术文档", "客服问答", "产品规格"],
            key="new_kb_type",
        )

        col_ok, col_cancel = st.columns([1, 1])
        with col_ok:
            if st.button("✅ 创建", type="primary", use_container_width=True):
                if kb_name:
                    _create_knowledge_base(kb_name, kb_desc, kb_type)
                    st.session_state.show_new_kb_dialog = False
                    st.session_state.current_kb = kb_name
                    st.rerun()
                else:
                    st.error("请输入知识库名称")
        with col_cancel:
            if st.button("❌ 取消", use_container_width=True):
                st.session_state.show_new_kb_dialog = False
                st.rerun()


def _create_knowledge_base(name: str, desc: str, kb_type: str) -> None:
    """调用后端 API 创建知识库。"""
    st.session_state.user_kbs.append(name)
    st.success(f"知识库「{name}」创建成功！")


# ======================== 引导流程 ========================

def render_onboarding_wizard():
    """渲染新用户引导流程（3 步向导）。"""
    if st.session_state.get("is_logged_in") and st.session_state.get("user_kbs"):
        return  # 已完成引导

    st.title("🎉 欢迎使用知阁 · 本地知识库 系统")
    st.markdown("---")

    step = st.session_state.get("onboarding_step", 1)

    if step == 1:
        _render_onboarding_step1()
    elif step == 2:
        _render_onboarding_step2()
    elif step == 3:
        _render_onboarding_step3()


def _render_onboarding_step1():
    """Step 1: 创建第一个知识库。"""
    st.subheader("Step 1/3：创建您的第一个知识库")
    st.markdown("知识库用于组织和管理您的文档，一个账户可以创建多个知识库。")

    kb_name = st.text_input(
        "知识库名称",
        placeholder="例如：产品文档库",
        key="onboard_kb_name",
    )
    kb_desc = st.text_area("描述（可选）", key="onboard_kb_desc")
    kb_type = st.selectbox(
        "知识库类型",
        ["通用文档", "技术文档", "客服问答", "产品规格"],
    )

    col_space, col_btn = st.columns([3, 1])
    with col_btn:
        if st.button("下一步 →", type="primary", use_container_width=True):
            if kb_name:
                st.session_state.current_kb = kb_name
                st.session_state.user_kbs = [kb_name]
                st.session_state.onboarding_step = 2
                st.rerun()
            else:
                st.error("请输入知识库名称")
    with col_space:
        st.markdown("")

    # 进度条
    st.progress(0.33, text="Step 1/3：创建知识库")


def _render_onboarding_step2():
    """Step 2: 上传文档。"""
    st.subheader("Step 2/3：上传文档到知识库")
    st.markdown(f"当前知识库：**{st.session_state.current_kb}**")

    st.info("支持 PDF、TXT、DOCX、Markdown 等格式", icon="ℹ️")

    uploaded = st.file_uploader(
        "选择文件或将文件拖拽到此处",
        type=["pdf", "txt", "docx", "md", "csv"],
        accept_multiple_files=True,
        help="支持批量上传，单个文件不超过 50MB",
    )

    if uploaded:
        st.success(f"已选择 {len(uploaded)} 个文件")
        for f in uploaded:
            st.write(f"  - {f.name} ({f.size / 1024:.1f} KB)")

    col_back, col_skip, col_btn = st.columns([1, 1, 1])
    with col_back:
        if st.button("← 上一步"):
            st.session_state.onboarding_step = 1
            st.rerun()
    with col_skip:
        if st.button("跳过（后续可上传）"):
            st.session_state.onboarding_step = 3
            st.rerun()
    with col_btn:
        if st.button("开始导入 →", type="primary", use_container_width=True):
            st.session_state.onboarding_step = 3
            st.rerun()

    st.progress(0.66, text="Step 2/3：上传文档")


def _render_onboarding_step3():
    """Step 3: 开始使用。"""
    st.subheader("Step 3/3：开始问答")
    st.success(f"知识库「{st.session_state.current_kb}」已就绪！", icon="✅")

    st.markdown("#### 试试问一个问题：")
    query = st.text_input(
        "在此输入问题...",
        placeholder="例如：本产品的退换货政策是什么？",
        key="onboard_query",
        label_visibility="collapsed",
    )

    if query:
        with st.spinner("正在检索和生成答案..."):
            # TODO: 调用 RAG pipeline
            st.info("RAG 检索中...（完成后端集成后可显示答案）")

    st.markdown("---")
    col_finish, col_back = st.columns([2, 1])
    with col_back:
        if st.button("← 返回上一步"):
            st.session_state.onboarding_step = 2
            st.rerun()
    with col_finish:
        if st.button("🏁 完成引导", type="primary", use_container_width=True):
            st.session_state.is_logged_in = True
            st.rerun()

    st.progress(1.0, text="Step 3/3：开始使用")


# ======================== 使用量显示 ========================

def render_usage_meter():
    """渲染当前使用量仪表（侧边栏底部）。"""
    if not st.session_state.get("is_logged_in"):
        return

    st.sidebar.markdown("---")
    st.sidebar.caption("📊 使用量")

    # TODO: 从后端获取真实数据
    chunks_used = 0
    chunks_limit = 1000
    queries_today = 0
    queries_limit = 50

    st.sidebar.metric(
        "文档块数",
        f"{chunks_used}",
        delta=f"限额 {chunks_limit}",
    )
    st.sidebar.metric(
        "今日查询",
        f"{queries_today}",
        delta=f"限额 {queries_limit}",
    )

    if queries_today >= queries_limit:
        st.sidebar.warning("今日查询额度已用完，请明天再试或升级套餐")
