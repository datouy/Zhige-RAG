"""Graph retriever: entity matching + multi-hop expansion."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.kg.schema import Entity, Relation, Triple
from src.kg.store import KGStore


class GraphRetriever:
    """Retrieve subgraph via entity matching and multi-hop expansion."""

    def __init__(self, store: KGStore, embedding_model=None) -> None:
        self.store = store
        self.embedding = embedding_model

    def search(
        self,
        query: str,
        top_k_entities: int = 5,
        hops: int = 2,
        relation_types: Optional[List[str]] = None,
        max_entities: int = 50,
        max_relations: int = 200,
    ) -> Dict[str, Any]:
        """检索子图。

        Args:
            query: 查询文本。
            top_k_entities: 种子实体数。
            hops: 扩展跳数。
            relation_types: 关系类型过滤。
            max_entities / max_relations: 结果总量上限，防止稠密图多跳扩展
            把内存和延迟打爆。
        """
        seed_entities, seed_relations = self._resolve_query(query, top_k_entities=top_k_entities)
        # 用 dict 去重：键分别为实体名 / (source, target, type)
        seen_entities: Dict[str, Entity] = {e.name: e for e in seed_entities}
        seen_relations: Dict[tuple, Relation] = {
            (r.source, r.target, r.type): r for r in seed_relations
        }
        queue = [e.name for e in seed_entities]
        for _ in range(max(0, hops)):
            if len(seen_entities) >= max_entities or len(seen_relations) >= max_relations:
                break
            next_queue: List[str] = []
            for entity_name in queue:
                if len(seen_entities) >= max_entities or len(seen_relations) >= max_relations:
                    break
                n_ents, n_rels = self._expand_neighbors(
                    entity_name,
                    relation_types=relation_types,
                    max_relations=max_relations - len(seen_relations),
                )
                for r in n_rels:
                    key = (r.source, r.target, r.type)
                    if key not in seen_relations:
                        seen_relations[key] = r
                for e in n_ents:
                    if e.name not in seen_entities:
                        seen_entities[e.name] = e
                        next_queue.append(e.name)
                        if len(seen_entities) >= max_entities:
                            break
            queue = next_queue
        all_entities = list(seen_entities.values())
        all_relations = list(seen_relations.values())
        nodes = [
            {
                "name": e.name,
                "type": e.type,
                "description": e.description,
            }
            for e in all_entities
        ]
        edges = [
            {
                "source": r.source,
                "target": r.target,
                "type": r.type,
                "description": r.description,
            }
            for r in all_relations
        ]
        triples = [
            Triple(
                subject=r.source,
                predicate=r.type,
                object=r.target,
                confidence=r.weight,
            )
            for r in all_relations
        ]
        return {
            "entities": all_entities,
            "relations": all_relations,
            "triples": triples,
            "subgraph": {"nodes": nodes, "edges": edges},
        }

    def query_cypher(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        return self.store.query_cypher(cypher, params)

    def _resolve_query(self, query: str, top_k_entities: int = 5) -> Tuple[List[Entity], List[Relation]]:
        candidates = self.store.find_entities_by_name(query, fuzzy=True, limit=top_k_entities)
        entities: List[Entity] = []
        relations: List[Relation] = []
        for e in candidates:
            entities.append(e)
            relations.extend(self.store.get_relations(e.name, direction="both"))
        return entities, relations

    def _expand_neighbors(
        self,
        entity_name: str,
        relation_types: Optional[List[str]] = None,
        max_relations: int = 50,
    ) -> Tuple[List[Entity], List[Relation]]:
        relations = self.store.get_relations(entity_name, direction="both")
        filtered: List[Relation] = []
        for r in relations:
            if relation_types and r.type not in relation_types:
                continue
            filtered.append(r)
            if len(filtered) >= max(0, max_relations):
                break
        neighbor_names = {r.target for r in filtered if r.source == entity_name} | {r.source for r in filtered if r.target == entity_name}
        # 每个邻居只查一次实体（之前对同一名字调了两次 get_entity）
        entities = []
        for name in neighbor_names:
            ent = self.store.get_entity(name)
            if ent is not None:
                entities.append(ent)
        return entities, filtered
