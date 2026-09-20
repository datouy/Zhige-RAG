"""Graph storage abstraction + Neo4j/SQLite implementations."""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.kg.schema import Entity, Relation


class KGStore(ABC):
    """Abstract graph store."""

    @abstractmethod
    def upsert_entities(self, entities: List[Entity]) -> int:
        """Upsert entities, return number of records affected."""

    @abstractmethod
    def upsert_relations(self, relations: List[Relation]) -> int:
        """Upsert relations, return number of records affected."""

    @abstractmethod
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by exact name."""

    @abstractmethod
    def find_entities_by_name(self, name: str, fuzzy: bool = True, limit: int = 10) -> List[Entity]:
        """Find entities by name substring or exact match."""

    @abstractmethod
    def get_relations(self, entity_name: str, direction: str = "both") -> List[Relation]:
        """Get relations for an entity."""

    @abstractmethod
    def query_cypher(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute cypher-style query if supported."""

    @abstractmethod
    def count(self) -> Dict[str, int]:
        """Return entity/relation counts."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all data."""


class SQLiteGraphStore(KGStore):
    """SQLite graph store fallback."""

    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path))
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI 的同步端点跑在线程池里，多个线程会并发访问同一个 store。
        # sqlite3 连接默认不允许跨线程，check_same_thread=False 只是绕过检查；
        # 必须配一把可重入锁串行化所有访问（query_cypher 内部会调用
        # get_entity，因此用 RLock 而不是 Lock）。
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        with self._lock:
            self._configure_conn()
            self._init_db()

    def _configure_conn(self) -> None:
        """提升并发下的可用性：WAL 降低读写互斥，busy_timeout 避免立即抛
        database is locked。:memory: 库不支持 journal_mode，忽略失败。"""
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.commit()

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                type TEXT,
                description TEXT,
                aliases TEXT,
                attributes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                target TEXT,
                type TEXT,
                description TEXT,
                weight REAL DEFAULT 1.0,
                attributes TEXT,
                source_doc TEXT,
                confidence REAL DEFAULT 1.0,
                chunk_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, target, type),
                FOREIGN KEY (source) REFERENCES entities(name),
                FOREIGN KEY (target) REFERENCES entities(name)
            );
            CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source);
            CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target);
            """
        )
        self._conn.commit()

    def upsert_entities(self, entities: List[Entity]) -> int:
        if not entities:
            return 0
        with self._lock:
            try:
                seen: Dict[str, Entity] = {}
                for e in entities:
                    if e.name in seen:
                        old = seen[e.name]
                        merged_aliases = list({a for a in old.aliases + e.aliases})
                        seen[e.name] = Entity(
                            name=old.name,
                            type=old.type or e.type,
                            description=old.description or e.description,
                            aliases=merged_aliases,
                            attributes={**(old.attributes or {}), **(e.attributes or {})},
                        )
                    else:
                        seen[e.name] = e
                cur = self._conn.cursor()
                now = datetime.now().isoformat()
                count = 0
                for e in seen.values():
                    cur.execute(
                        """
                        INSERT INTO entities (name, type, description, aliases, attributes, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(name) DO UPDATE SET
                            type=COALESCE(NULLIF(excluded.type, ''), entities.type),
                            description=COALESCE(NULLIF(excluded.description, ''), entities.description),
                            aliases=excluded.aliases,
                            attributes=excluded.attributes
                        """,
                        (
                            e.name,
                            e.type,
                            e.description,
                            json.dumps(e.aliases, ensure_ascii=False),
                            json.dumps(e.attributes, ensure_ascii=False),
                            now,
                        ),
                    )
                    count += 1
                self._conn.commit()
                return count
            except Exception:
                self._conn.rollback()
                raise

    def upsert_relations(self, relations: List[Relation]) -> int:
        if not relations:
            return 0
        with self._lock:
            try:
                cur = self._conn.cursor()
                now = datetime.now().isoformat()
                # FK 约束开启后，引用不存在实体的关系会直接抛
                # FOREIGN KEY constraint failed（LLM 抽取的结果里
                # relations 引用 entities 之外的实体名并不罕见）。
                # 用 INSERT OR IGNORE 补建缺失的端点实体：已存在的实体
                # 保持原数据不受影响，缺失的以占位实体补齐。
                for r in relations:
                    for endpoint in (r.source, r.target):
                        if endpoint:
                            cur.execute(
                                """
                                INSERT OR IGNORE INTO entities (name, type, description, aliases, attributes, created_at)
                                VALUES (?, 'Other', '', '[]', '{}', ?)
                                """,
                                (endpoint, now),
                            )
                count = 0
                for r in relations:
                    if not r.source or not r.target:
                        continue
                    cur.execute(
                        """
                        INSERT INTO relations
                            (source, target, type, description, weight, attributes, source_doc, confidence, chunk_id, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(source, target, type) DO UPDATE SET
                            description=excluded.description,
                            weight=excluded.weight,
                            attributes=excluded.attributes,
                            source_doc=excluded.source_doc,
                            confidence=excluded.confidence,
                            chunk_id=excluded.chunk_id
                        """,
                        (
                            r.source,
                            r.target,
                            r.type,
                            r.description,
                            float(r.weight),
                            json.dumps(r.attributes, ensure_ascii=False),
                            r.attributes.get("source_doc", ""),
                            float(r.attributes.get("confidence", r.weight)),
                            r.attributes.get("chunk_id"),
                            now,
                        ),
                    )
                    count += 1
                self._conn.commit()
                return count
            except Exception:
                self._conn.rollback()
                raise

    def get_entity(self, name: str) -> Optional[Entity]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT name, type, description, aliases, attributes FROM entities WHERE name = ?", (name,))
            row = cur.fetchone()
            if not row:
                return None
            return Entity(
                name=row[0],
                type=row[1] or "Other",
                description=row[2] or "",
                aliases=json.loads(row[3] or "[]"),
                attributes=json.loads(row[4] or "{}"),
            )

    def find_entities_by_name(self, name: str, fuzzy: bool = True, limit: int = 10) -> List[Entity]:
        with self._lock:
            cur = self._conn.cursor()
            q = f"%{name}%"
            rows = []
            if fuzzy:
                cur.execute(
                    "SELECT name, type, description, aliases, attributes FROM entities WHERE name LIKE ? OR description LIKE ? ORDER BY LENGTH(name) ASC LIMIT ?",
                    (q, q, limit),
                )
                rows = cur.fetchall()
            if not rows:
                cur.execute(
                    "SELECT name, type, description, aliases, attributes FROM entities WHERE name = ? LIMIT ?",
                    (name, limit),
                )
                rows = cur.fetchall()
            return [
                Entity(
                    name=r[0],
                    type=r[1] or "Other",
                    description=r[2] or "",
                    aliases=json.loads(r[3] or "[]"),
                    attributes=json.loads(r[4] or "{}"),
                )
                for r in rows
            ]

    def get_relations(self, entity_name: str, direction: str = "both") -> List[Relation]:
        with self._lock:
            cur = self._conn.cursor()
            rows = []
            if direction in ("both", "out"):
                cur.execute(
                    "SELECT source, target, type, description, weight, attributes, source_doc, confidence, chunk_id FROM relations WHERE source = ?",
                    (entity_name,),
                )
                rows.extend(cur.fetchall())
            if direction in ("both", "in"):
                cur.execute(
                    "SELECT source, target, type, description, weight, attributes, source_doc, confidence, chunk_id FROM relations WHERE target = ?",
                    (entity_name,),
                )
                rows.extend(cur.fetchall())
            out = []
            for r in rows:
                attrs = json.loads(r[5] or "{}")
                attrs.update({"source_doc": r[6] or "", "confidence": float(r[7] or 1.0), "chunk_id": r[8]})
                out.append(
                    Relation(
                        source=r[0],
                        target=r[1],
                        type=r[2] or "",
                        description=r[3] or "",
                        weight=float(r[4] or 1.0),
                        attributes=attrs,
                    )
                )
            return out

    def query_cypher(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        """Minimal Cypher-like subset."""
        with self._lock:
            return self._query_cypher_locked(cypher, params)

    def _query_cypher_locked(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        text = (cypher or "").strip()
        params = params or {}
        if not (text.upper().startswith("MATCH") and "RETURN" in text.upper()):
            return []
        try:
            return_part = text.split("RETURN", 1)[1]
            ret_cols = [c.strip() for c in return_part.split(",")]
            match_part = text.split("RETURN", 1)[0]
            rel_type = None
            source_label = None
            target_label = None
            where_name = None
            where_type = None
            if "-[" in match_part:
                _left, right = match_part.split("-[", 1)
                right_main = right.split("]->", 1)[0]
                rel_part = right_main.strip("[] ")
                if ":" in rel_part:
                    rel_type = rel_part.split(":", 1)[1].strip()
            left = match_part
            if "(" in left:
                left_main = left[left.rfind("(") + 1 : left.rfind(")")]
                if ":" in left_main:
                    source_label = left_main.split(":", 1)[1].strip()
            if ")->(" in match_part:
                right_node = match_part.split(")->(", 1)[1].split(")", 1)[0]
                if ":" in right_node:
                    target_label = right_node.split(":", 1)[1].strip()
            if "WHERE" in text.upper():
                where_clause = text.split("WHERE", 1)[1].split("RETURN", 1)[0]
                if "name" in where_clause and "=" in where_clause:
                    where_name = where_clause.split("=", 1)[1].strip().strip("'\"").strip()
                if "type" in where_clause and "=" in where_clause:
                    where_type = where_clause.split("=", 1)[1].strip().strip("'\"").strip()
            cur = self._conn.cursor()
            if rel_type:
                cur.execute(
                    "SELECT source, target, type, description, weight, attributes, source_doc, confidence, chunk_id FROM relations WHERE type = ?",
                    (rel_type,),
                )
            else:
                cur.execute(
                    "SELECT source, target, type, description, weight, attributes, source_doc, confidence, chunk_id FROM relations"
                )
            target_rows = []
            for row in cur.fetchall():
                if where_name and (row[0] != where_name and row[1] != where_name):
                    continue
                if target_label:
                    cur.execute("SELECT type FROM entities WHERE name = ?", (row[1],))
                    tr = cur.fetchone()
                    if not tr or (tr[0] or "") != target_label:
                        continue
                if source_label:
                    cur.execute("SELECT type FROM entities WHERE name = ?", (row[0],))
                    sr = cur.fetchone()
                    if not sr or (sr[0] or "") != source_label:
                        continue
                if where_type:
                    cur.execute("SELECT type FROM entities WHERE name = ?", (row[0],))
                    sr = cur.fetchone()
                    if not sr or (sr[0] or "") != where_type:
                        continue
                target_rows.append(row)
            results: List[Dict[str, Any]] = []
            for r in target_rows:
                item: Dict[str, Any] = {}
                for col in ret_cols:
                    key = col.lower()
                    if key in ("a", "source"):
                        ent = self.get_entity(r[0])
                        item["a"] = ent.__dict__ if ent else None
                    elif key in ("b", "target"):
                        ent2 = self.get_entity(r[1])
                        item["b"] = ent2.__dict__ if ent2 else None
                    elif key in ("r", "rel"):
                        item["r"] = {
                            "source": r[0],
                            "target": r[1],
                            "type": r[2],
                            "description": r[3],
                            "weight": float(r[4] or 1.0),
                        }
                    else:
                        item[key] = None
                results.append(item)
            return results
        except Exception:
            return []

    def count(self) -> Dict[str, int]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM entities")
            entities = cur.fetchone()[0] or 0
            cur.execute("SELECT COUNT(*) FROM relations")
            relations = cur.fetchone()[0] or 0
            return {"entities": int(entities), "relations": int(relations)}

    def clear(self) -> None:
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute("DELETE FROM relations")
                cur.execute("DELETE FROM entities")
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


class Neo4jStore(KGStore):
    """Optional Neo4j backend."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:
            raise ImportError("neo4j driver is not installed. Install neo4j package.") from exc
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database or "neo4j"
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def _session(self):
        return self.driver.session(database=self.database)

    def upsert_entities(self, entities: List[Entity]) -> int:
        if not entities:
            return 0
        with self._session() as session:
            tx = session.begin_transaction()
            count = 0
            for e in entities:
                tx.run(
                    "MERGE (n:Entity {name: $name}) SET n.type = $type, n.description = $description, n.aliases = $aliases, n.attributes = $attributes, n.created_at = datetime()",
                    {"name": e.name, "type": e.type, "description": e.description, "aliases": [a for a in e.aliases], "attributes": dict(e.attributes)},
                )
                count += 1
            tx.commit()
            return count

    def upsert_relations(self, relations: List[Relation]) -> int:
        if not relations:
            return 0
        with self._session() as session:
            tx = session.begin_transaction()
            count = 0
            for r in relations:
                tx.run(
                    "MATCH (a:Entity {name: $source}) MATCH (b:Entity {name: $target}) MERGE (a)-[rel:RELATION {type: $type}]->(b) SET rel.description = $description, rel.weight = $weight, rel.attributes = $attributes, rel.source_doc = $source_doc, rel.confidence = $confidence, rel.created_at = datetime()",
                    {"source": r.source, "target": r.target, "type": r.type, "description": r.description, "weight": float(r.weight), "attributes": dict(r.attributes), "source_doc": r.attributes.get("source_doc", ""), "confidence": float(r.attributes.get("confidence", r.weight))},
                )
                count += 1
            tx.commit()
            return count

    def get_entity(self, name: str) -> Optional[Entity]:
        with self._session() as session:
            record = session.run(
                "MATCH (n:Entity {name: $name}) RETURN n.name AS name, n.type AS type, n.description AS description, n.aliases AS aliases, n.attributes AS attributes",
                {"name": name},
            ).single()
            if not record:
                return None
            return Entity(
                name=record["name"],
                type=record["type"] or "Other",
                description=record["description"] or "",
                aliases=[a for a in (record["aliases"] or [])],
                attributes=dict(record["attributes"] or {}),
            )

    def find_entities_by_name(self, name: str, fuzzy: bool = True, limit: int = 10) -> List[Entity]:
        with self._session() as session:
            if fuzzy:
                records = session.run(
                    "MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower($name) RETURN n.name AS name, n.type AS type, n.description AS description, n.aliases AS aliases, n.attributes AS attributes LIMIT $limit",
                    {"name": name, "limit": limit},
                ).data()
            else:
                records = session.run(
                    "MATCH (n:Entity {name: $name}) RETURN n.name AS name, n.type AS type, n.description AS description, n.aliases AS aliases, n.attributes AS attributes LIMIT $limit",
                    {"name": name, "limit": limit},
                ).data()
            return [
                Entity(
                    name=r["name"],
                    type=r["type"] or "Other",
                    description=r["description"] or "",
                    aliases=[a for a in (r["aliases"] or [])],
                    attributes=dict(r["attributes"] or {}),
                )
                for r in records
            ]

    def get_relations(self, entity_name: str, direction: str = "both") -> List[Relation]:
        with self._session() as session:
            out_records: List[dict] = []
            in_records: List[dict] = []
            if direction in ("both", "out"):
                out_records = session.run(
                    "MATCH (a:Entity {name: $name})-[r:RELATION]->(b:Entity) RETURN a.name AS source, b.name AS target, r.type AS type, r.description AS description, r.weight AS weight, r.attributes AS attributes, r.source_doc AS source_doc, r.confidence AS confidence",
                    {"name": entity_name},
                ).data()
            if direction in ("both", "in"):
                in_records = session.run(
                    "MATCH (a:Entity {name: $name})<-[r:RELATION]-(b:Entity) RETURN b.name AS source, a.name AS target, r.type AS type, r.description AS description, r.weight AS weight, r.attributes AS attributes, r.source_doc AS source_doc, r.confidence AS confidence",
                    {"name": entity_name},
                ).data()

            def _to_rel(rec):
                return Relation(
                    source=rec["source"],
                    target=rec["target"],
                    type=rec["type"] or "",
                    description=rec["description"] or "",
                    weight=float(rec["weight"] or 1.0),
                    attributes={**(rec["attributes"] or {}), "source_doc": rec.get("source_doc") or "", "confidence": float(rec.get("confidence") or 1.0)},
                )

            return [_to_rel(rec) for rec in out_records] + [_to_rel(rec) for rec in in_records]

    def query_cypher(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        with self._session() as session:
            try:
                result = session.run(cypher, dict(params or {}))
                return [dict(r) for r in result.data()]
            except Exception:
                return []

    def count(self) -> Dict[str, int]:
        with self._session() as session:
            entities = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            relations = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS c").single()["c"]
            return {"entities": int(entities), "relations": int(relations)}

    def clear(self) -> None:
        with self._session() as session:
            session.run("MATCH (n:Entity) DETACH DELETE n")


def create_kg_store(config: Dict) -> KGStore:
    """Create KGStore based on config, fallback to SQLite if neo4j is unavailable."""
    backend = (config.get("backend") or "sqlite").lower()
    if backend == "neo4j":
        try:
            from src.utils import get_logger
            logger = get_logger("kg.store")
            neo4j_cfg = config.get("neo4j", {})
            store = Neo4jStore(
                uri=str(neo4j_cfg.get("uri", "bolt://localhost:7687")),
                user=str(neo4j_cfg.get("user", "neo4j")),
                password=str(neo4j_cfg.get("password", "")),
                database=str(neo4j_cfg.get("database", "neo4j")),
            )
            return store
        except Exception as exc:
            from src.utils import get_logger
            get_logger("kg.store").warning("Neo4j 不可用，回退到 SQLite: %s", exc)
    sqlite_path = config.get("sqlite_path", "data/kg.db")
    return SQLiteGraphStore(db_path=sqlite_path)
