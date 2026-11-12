"""死配置与"配置项存在但无人读取"的守护测试。

背景：``config/config.yaml`` 里曾经有一批**从未被任何代码读取**的键。
它们比"配置写错"更隐蔽 —— 用户照着文档调它们，系统毫无反应，
于是得出"配置不生效"的错误结论（实际是"这项压根没接线"）。

本轮修掉的三个：

1. ``rag.mapreduce.inter_call_delay_sec``
   map 阶段改成并行 ThreadPool 后，串行 sleep 就不再需要了。
   该键却还留在 yaml 里，注释还在指导用户"保持这个间隔"来防限流 ——
   调它当然没用。而真正管用的 ``max_workers`` 反而没在 yaml 里。

2. ``knowledge_graph.retriever.default_hops``
   实际生效的跳数来自 ``knowledge_graph.graph_rag.graph_hops``，
   这个 ``default_hops`` 无人读取。

3. ``knowledge_graph.retriever.enable_fuzzy_match``
   匹配方式被硬编码成 ``fuzzy=True``，配置项形同虚设。
   现已真正接线（``GraphRetriever(enable_fuzzy_match=...)``）。

这里用"静态扫描 + 行为断言"把三件事钉住。
"""

from __future__ import annotations

import pathlib
import re

import pytest

from src.utils import load_config

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _code_sources() -> str:
    """全代码库（不含 tests/ 与文档）的源码文本。"""
    parts = []
    for pat in ("src/**/*.py", "api/**/*.py", "ui/**/*.py", "scripts/**/*.py"):
        for p in ROOT.glob(pat):
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


# ==================== 已删除的死配置 ====================
class TestRemovedDeadKeys:
    @pytest.mark.parametrize(
        "path",
        [
            ("rag", "mapreduce", "inter_call_delay_sec"),
            ("knowledge_graph", "retriever", "default_hops"),
        ],
        ids=["rag.mapreduce.inter_call_delay_sec", "kg.retriever.default_hops"],
    )
    def test_key_removed_from_config(self, path):
        cfg = load_config("config/config.yaml")
        cur = cfg
        for part in path[:-1]:
            assert isinstance(cur, dict) and part in cur, (
                f"{'.'.join(path[:-1])} 这一段配置都不存在，无法断言其子键"
            )
            cur = cur[part]
        leaf = path[-1]
        assert leaf not in cur, (
            f"{'.'.join(path)} 是死配置，应从 config.yaml 删除"
        )

    def test_inter_call_delay_not_referenced_in_code(self):
        """并行 map 之后不该再有"串行 sleep 防限流"的残留。

        只看**真正的代码**（跳过注释行）—— 解释它为何被移除的注释里
        出现这个键名是合理的。
        """
        src = _code_sources()
        code_lines = [
            ln for ln in src.splitlines()
            if not ln.strip().startswith("#")
        ]
        body = "\n".join(code_lines)
        assert "inter_call_delay_sec" not in body, (
            "inter_call_delay_sec 已从配置删除，代码里也不该再出现"
        )


# ==================== 该生效的必须真的生效 ====================
class TestWiredUpKeys:
    def test_max_workers_exists_in_config(self):
        """``max_workers`` 才是 map 阶段并发度的真值来源，必须在 yaml 里。"""
        cfg = load_config("config/config.yaml")
        mr = (cfg.get("rag") or {}).get("mapreduce") or {}
        assert "max_workers" in mr, (
            "rag.mapreduce.max_workers 缺失 —— 并发度只能改代码才能调"
        )
        assert 1 <= int(mr["max_workers"]) <= 8

    def test_max_workers_is_read_by_code(self):
        src = _code_sources()
        assert 'cfg.get("max_workers"' in src, (
            "代码里应当读取 rag.mapreduce.max_workers"
        )

    def test_enable_fuzzy_match_is_read_by_code(self):
        src = _code_sources()
        assert "enable_fuzzy_match" in src, (
            "代码里应当读取 knowledge_graph.retriever.enable_fuzzy_match"
        )

    def test_fuzzy_match_flag_is_honoured(self):
        """关掉模糊匹配后，retriever 不再要求模糊查找。"""
        from src.kg.retriever import GraphRetriever
        from src.kg.store import SQLiteGraphStore
        from src.kg.schema import Entity

        store = SQLiteGraphStore(db_path=":memory:")
        store.upsert_entities([Entity(name="阿里巴巴", type="Org", description="")])

        calls = []

        class _SpyStore(SQLiteGraphStore):
            def find_entities_by_name(self, name, fuzzy=True, limit=10):
                calls.append(fuzzy)
                return super().find_entities_by_name(name, fuzzy=fuzzy, limit=limit)

        spy = _SpyStore(db_path=":memory:")
        spy.upsert_entities([Entity(name="阿里巴巴", type="Org", description="")])

        GraphRetriever(spy, enable_fuzzy_match=False).search("阿里", hops=0)
        assert calls == [False], "enable_fuzzy_match=False 时应当以 fuzzy=False 查询"

        calls.clear()
        GraphRetriever(spy, enable_fuzzy_match=True).search("阿里", hops=0)
        assert calls == [True], "enable_fuzzy_match=True 时应当以 fuzzy=True 查询"


# ==================== 通用：yaml 叶子键不应是"没人读"的 ====================
class TestNoUnreadLeafKeys:
    """通用扫描：yaml 叶子键的末段名应当能在代码里找到。

    这是**启发式**检查（末段名可能碰巧同名），但足以抓住"整段配置被
    遗弃"这类问题 —— 比如整块 ``mapreduce`` 的默认值与 yaml 脱节。
    """

    # 这些键由 validator 声明、由外部系统消费，或值本身就是配置语义，
    # 不要求在业务代码里出现。
    ALLOWED_UNREAD = {
        # 项目元信息
        "name", "version", "description",
    }

    def test_leaf_key_names_are_referenced(self):
        src = _code_sources()
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", src))
        validator = (ROOT / "src" / "config" / "validator.py").read_text(encoding="utf-8")

        def walk(d, pre=""):
            if isinstance(d, dict):
                for k, v in d.items():
                    yield from walk(v, f"{pre}.{k}" if pre else k)
            else:
                yield pre, d

        unread = []
        for path, _ in walk(load_config("config/config.yaml")):
            leaf = path.split(".")[-1]
            if leaf in self.ALLOWED_UNREAD:
                continue
            if leaf in tokens or leaf in validator:
                continue
            unread.append(path)

        assert not unread, (
            "以下配置项在代码里从未被读取（改了不会有任何效果）：\n  "
            + "\n  ".join(unread)
        )
