"""Tests for src/kg/ module (extractor, store, retriever, graph_rag, builder)."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kg import (  # noqa: E402
    Entity,
    GraphRAG,
    GraphRetriever,
    KGExtractor,
    KGStore,
    Neo4jStore,
    Relation,
    SQLiteGraphStore,
    Triple,
    create_kg_store,
)
from src.kg.schema import Triple as SchemaTriple  # noqa: E402


# ------------------------- helpers -------------------------
class FakeLLM:
    """Simple fake LLM that returns a fixed chat output."""

    def __init__(self, output: str = "") -> None:
        self.output = output
        self.calls: List[List[Dict[str, str]]] = []

    def chat(self, messages, generation=None, stream=False):  # noqa: D401
        self.calls.append(list(messages))
        if stream:
            return iter([self.output])
        return self.output


def make_entity(name: str = "Alice", type: str = "Person", description: str = "科学家") -> Entity:
    return Entity(name=name, type=type, description=description)


def make_relation(source: str = "Alice", target: str = "MIT", type: str = "WORKS_FOR") -> Relation:
    return Relation(source=source, target=target, type=type, description="就职于")


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    db = tmp_path / "kg.db"
    return str(db)


@pytest.fixture
def store(tmp_db: str) -> SQLiteGraphStore:
    s = SQLiteGraphStore(db_path=tmp_db)
    s.clear()
    return s


# ========================== schema ==========================
class TestSchema:
    def test_entity_defaults(self):
        e = Entity(name="Alice")
        assert e.name == "Alice"
        assert e.type == "Other"
        assert e.aliases == []
        assert e.attributes == {}

    def test_relation_defaults(self):
        r = Relation(source="A", target="B", type="REL")
        assert r.weight == 1.0
        assert r.attributes == {}

    def test_triple_defaults(self):
        t = Triple(subject="A", predicate="P", object="B")
        assert t.confidence == 1.0
        assert t.chunk_id is None
        assert t.source_doc == ""


# ========================== extractor ==========================
class TestExtractor:
    def test_extract_entities_from_text(self, tmp_db):
        llm = FakeLLM(
            output=json.dumps(
                {
                    "entities": [
                        {"name": "爱因斯坦", "type": "Person", "description": "物理学家"},
                    ],
                    "relations": [],
                },
                ensure_ascii=False,
            )
        )
        ex = KGExtractor(llm=llm)
        ents, rels = ex.extract("爱因斯坦是物理学家。")
        assert len(ents) == 1
        assert ents[0].name == "爱因斯坦"
        assert ents[0].type == "Person"
        assert rels == []

    def test_extract_relations_from_text(self):
        llm = FakeLLM(
            output=json.dumps(
                {
                    "entities": [
                        {"name": "爱因斯坦", "type": "Person"},
                        {"name": "普林斯顿大学", "type": "Organization"},
                    ],
                    "relations": [
                        {"source": "爱因斯坦", "target": "普林斯顿大学", "type": "WORKS_FOR", "description": "工作"},
                    ],
                },
                ensure_ascii=False,
            )
        )
        ex = KGExtractor(llm=llm)
        ents, rels = ex.extract("爱因斯坦在普林斯顿大学工作。")
        assert len(ents) == 2
        assert len(rels) == 1
        assert rels[0].type == "WORKS_FOR"
        assert rels[0].source == "爱因斯坦"
        assert rels[0].target == "普林斯顿大学"

    def test_extract_handles_malformed_json(self):
        # LLM 输出了 Markdown 代码块包裹的 JSON
        llm = FakeLLM(
            output="""以下是结果：
```json
{
  "entities": [{"name": "Foo", "type": "Other"}],
  "relations": []
}
```
"""
        )
        ex = KGExtractor(llm=llm)
        ents, rels = ex.extract("Foo bar")
        assert len(ents) == 1
        assert ents[0].name == "Foo"
        assert rels == []

    def test_extract_handles_pure_garbage(self):
        llm = FakeLLM(output="not json at all, please ignore")
        ex = KGExtractor(llm=llm)
        ents, rels = ex.extract("some text")
        assert ents == []
        assert rels == []

    def test_extract_from_chunks(self):
        llm = FakeLLM(
            output=json.dumps(
                {
                    "entities": [{"name": "Foo", "type": "Other"}],
                    "relations": [{"source": "Foo", "target": "Bar", "type": "REL"}],
                },
                ensure_ascii=False,
            )
        )
        ex = KGExtractor(llm=llm)
        chunks = [
            {"text": "Foo bar.", "source_doc": "doc1.md", "chunk_id": "c1"},
            {"text": "Foo bar2.", "source_doc": "doc2.md", "chunk_id": "c2"},
        ]
        ents, rels, triples = ex.extract_from_chunks(chunks)
        assert len(ents) == 2
        assert len(rels) == 2
        assert len(triples) == 2
        assert triples[0].source_doc == "doc1.md"
        assert triples[0].chunk_id == "c1"

    def test_extract_empty_text(self):
        llm = FakeLLM(output="{}")
        ex = KGExtractor(llm=llm)
        ents, rels = ex.extract("")
        assert ents == []
        assert rels == []


# ========================== store ==========================
class TestSQLiteStore:
    def test_sqlite_store_upsert_entities(self, store: SQLiteGraphStore):
        n = store.upsert_entities([make_entity("Alice"), make_entity("Bob", type="Person")])
        assert n == 2
        assert store.count()["entities"] == 2

    def test_sqlite_store_upsert_relations(self, store: SQLiteGraphStore):
        store.upsert_entities([make_entity("Alice"), make_entity("MIT", type="Organization")])
        n = store.upsert_relations([make_relation("Alice", "MIT")])
        assert n == 1
        assert store.count()["relations"] == 1

    def test_sqlite_store_find_entity(self, store: SQLiteGraphStore):
        store.upsert_entities([make_entity("Alice"), make_entity("Bob")])
        e = store.find_entities_by_name("Alice", fuzzy=False)
        assert len(e) == 1
        assert e[0].name == "Alice"

    def test_sqlite_store_get_relations(self, store: SQLiteGraphStore):
        store.upsert_entities([make_entity("Alice"), make_entity("MIT", type="Organization"), make_entity("China", type="Location")])
        store.upsert_relations(
            [
                make_relation("Alice", "MIT"),
                Relation(source="Alice", target="China", type="LIVES_IN"),
            ]
        )
        rels = store.get_relations("Alice")
        assert len(rels) == 2
        out_only = store.get_relations("Alice", direction="out")
        assert all(r.source == "Alice" for r in out_only)

    def test_sqlite_store_count(self, store: SQLiteGraphStore):
        store.upsert_entities([make_entity("A"), make_entity("B")])
        store.upsert_relations([make_relation("A", "B")])
        cnt = store.count()
        assert cnt["entities"] == 2
        assert cnt["relations"] == 1

    def test_sqlite_store_clear(self, store: SQLiteGraphStore):
        store.upsert_entities([make_entity("A")])
        store.clear()
        cnt = store.count()
        assert cnt["entities"] == 0
        assert cnt["relations"] == 0

    def test_create_kg_store_sqlite_default(self):
        s = create_kg_store({"backend": "sqlite", "sqlite_path": ":memory:"})
        assert isinstance(s, SQLiteGraphStore)

    def test_create_kg_store_neo4j_fallback(self, monkeypatch):
        # 模拟 neo4j 模块未安装
        monkeypatch.setitem(sys.modules, "neo4j", None)
        s = create_kg_store({"backend": "neo4j", "sqlite_path": ":memory:", "neo4j": {}})
        assert isinstance(s, SQLiteGraphStore)


# ========================== retriever ==========================
class TestGraphRetriever:
    def setup_method(self):
        self.store = SQLiteGraphStore(db_path=":memory:")
        self.store.upsert_entities(
            [
                make_entity("Alice"),
                make_entity("MIT", type="Organization"),
                make_entity("Bob"),
                make_entity("Stanford", type="Organization"),
                make_entity("Charlie"),
            ]
        )
        self.store.upsert_relations(
            [
                make_relation("Alice", "MIT"),
                Relation(source="Bob", target="Stanford", type="WORKS_FOR"),
                Relation(source="Charlie", target="MIT", type="STUDIES_AT"),
                Relation(source="Alice", target="Bob", type="KNOWS"),
            ]
        )
        self.retriever = GraphRetriever(self.store)

    def test_graph_retriever_search(self):
        result = self.retriever.search("Alice", top_k_entities=5, hops=1)
        assert "entities" in result
        assert "relations" in result
        assert "subgraph" in result
        assert any(e.name == "Alice" for e in result["entities"])

    def test_graph_retriever_multihop(self):
        result = self.retriever.search("Alice", top_k_entities=5, hops=2)
        # Alice -> MIT, Alice -> Bob -> Stanford
        names = {e.name for e in result["entities"]}
        assert "MIT" in names
        assert "Bob" in names
        assert "Stanford" in names

    def test_graph_retriever_relation_type_filter(self):
        result = self.retriever.search("Alice", top_k_entities=5, hops=2, relation_types=["KNOWS"])
        names = {e.name for e in result["entities"]}
        # WORKS_FOR/STUDIES_AT 关系被过滤，只剩 KNOWS 链
        assert "Bob" in names
        # 通过 WORKS_FOR 链无法触达 Stanford
        assert "Stanford" not in names

    def test_graph_retriever_fuzzy_match(self):
        result = self.retriever.search("Ali", top_k_entities=5, hops=1)
        assert any(e.name == "Alice" for e in result["entities"])

    def test_graph_retriever_cypher_simple(self):
        # Cypher 子集 MATCH (a)-[r:WORKS_FOR]->(b) RETURN a, r, b
        result = self.retriever.query_cypher(
            "MATCH (a)-[r:WORKS_FOR]->(b) RETURN a, r, b"
        )
        # WORKS_FOR 关系存在 1 条
        assert isinstance(result, list)


# ========================== graph_rag ==========================
class TestGraphRAG:
    def test_graph_rag_basic(self):
        # mock LLM
        llm = FakeLLM(output="基于知识库的回答：爱因斯坦在普林斯顿大学工作。")
        # mock vector store
        vs = MagicMock()
        vs.query.return_value = [
            MagicMock(
                id="chunk1",
                text="爱因斯坦在普林斯顿大学工作。",
                score=0.9,
                metadata={"source": "doc1.md"},
            )
        ]
        kg_store = SQLiteGraphStore(db_path=":memory:")
        # 预存一些实体，验证能否扩展
        kg_store.upsert_entities([make_entity("爱因斯坦", type="Person"), make_entity("普林斯顿大学", type="Organization")])
        kg_store.upsert_relations([Relation(source="爱因斯坦", target="普林斯顿大学", type="WORKS_FOR")])
        retriever = GraphRetriever(kg_store)
        extractor = MagicMock()
        extractor.extract_from_chunks.return_value = ([], [], [])
        gr = GraphRAG(vector_store=vs, kg_store=kg_store, extractor=extractor, retriever=retriever, llm=llm)
        result = gr.query("爱因斯坦在哪里工作？", top_k=2, graph_hops=2)
        assert "answer" in result
        assert "sources" in result
        assert "graph_context" in result
        assert result["answer"].startswith("基于知识库的回答")
        assert len(result["sources"]) == 1


# ========================== integration ==========================
class TestIntegration:
    def test_kg_pipeline_integration(self):
        """端到端：抽取 → 入库 → 检索"""
        llm = FakeLLM(
            output=json.dumps(
                {
                    "entities": [
                        {"name": "阿里巴巴", "type": "Organization", "description": "中国互联网公司"},
                        {"name": "马云", "type": "Person", "description": "企业家"},
                        {"name": "杭州", "type": "Location", "description": "城市"},
                    ],
                    "relations": [
                        {"source": "马云", "target": "阿里巴巴", "type": "FOUNDED"},
                        {"source": "阿里巴巴", "target": "杭州", "type": "LOCATED_IN"},
                    ],
                },
                ensure_ascii=False,
            )
        )
        ex = KGExtractor(llm=llm)
        ents, rels, triples = ex.extract_from_chunks(
            [
                {"text": "马云创立了阿里巴巴，总部在杭州。", "source_doc": "doc.md", "chunk_id": "c1"},
            ]
        )
        assert len(ents) == 3
        assert len(rels) == 2
        assert len(triples) == 2

        store = SQLiteGraphStore(db_path=":memory:")
        n_e = store.upsert_entities(ents)
        n_r = store.upsert_relations(rels)
        assert n_e == 3
        assert n_r == 2

        retriever = GraphRetriever(store)
        result = retriever.search("马云", top_k_entities=5, hops=2)
        names = {e.name for e in result["entities"]}
        assert "阿里巴巴" in names
        assert "杭州" in names


class TestConfig:
    def test_kg_config_load(self):
        from src.utils import load_config
        cfg = load_config("config/config.yaml")
        kg = cfg.get("knowledge_graph")
        assert kg is not None
        assert kg.get("backend") in ("sqlite", "neo4j")
        assert "sqlite_path" in kg
        assert "extractor" in kg
        assert "retriever" in kg
        assert "graph_rag" in kg


class TestBuilder:
    def test_build_kg_script_dry_run(self, tmp_path):
        """测试 build_kg.py 在不调用 LLM 时也能跑通（print-only）。"""
        from scripts.build_kg import main as build_main

        # 创建临时目录与文件
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "doc1.txt").write_text("阿里巴巴由马云创立，总部在杭州。", encoding="utf-8")

        # 使用 monkeypatch 替换 sys.argv 与 print
        captured = []
        import builtins
        real_print = builtins.print

        def fake_print(*args, **kwargs):
            captured.append(" ".join(str(a) for a in args))

        # 调用 print-only 模式
        saved_argv = sys.argv
        try:
            sys.argv = [
                "build_kg.py",
                "--input",
                str(raw_dir),
                "--print-only",
                "--config",
                "config/config.yaml",
            ]
            builtins.print = fake_print
            rc = build_main()
        finally:
            sys.argv = saved_argv
            builtins.print = real_print

        assert rc == 0
        assert any("documents" in line for line in captured)