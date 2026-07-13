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
    ) -> Dict[str, Any]:
        entities, relations = self._resolve_query(query, top_k_entities=top_k_entities)
        expanded_entities: List[Entity] = []
        expanded_relations: List[Relation] = []
        seen_entities = {e.name: e for e in entities}
        seen_relations: List[Relation] = []
        queue = [e.name for e in entities]
        for _ in range(max(0, hops)):
            next_queue: List[str] = []
            for entity_name in queue:
                n_ents, n_rels = self._expand_neighbors(entity_name, hops=1, relation_types=relation_types)
                for e in n_ents:
                    if e.name not in seen_entities:
                        seen_entities[e.name] = e
                        expanded_entities.append(e)
                        next_queue.append(e.name)
                for r in n_rels:
                    if r not in seen_relations:
                        seen_relations.append(r)
                        expanded_relations.append(r)
            queue = next_queue
        all_entities = list(seen_entities.values()) + expanded_entities
        all_relations = relations + expanded_relations
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
        self, entity_name: str, hops: int = 1, relation_types: Optional[List[str]] = None
    ) -> Tuple[List[Entity], List[Relation]]:
        relations = self.store.get_relations(entity_name, direction="both")
        filtered: List[Relation] = []
        for r in relations:
            if relation_types and r.type not in relation_types:
                continue
            filtered.append(r)
        neighbor_names = {r.target for r in filtered if r.source == entity_name} | {r.source for r in filtered if r.target == entity_name}
        entities = [self.store.get_entity(name) for name in neighbor_names if self.store.get_entity(name) is not None]
        return entities, filtered
