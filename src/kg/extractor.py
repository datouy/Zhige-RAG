"""LLM-based knowledge graph extractor."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

from src.llm import LocalLLM
from src.kg.schema import Entity, Relation, Triple
from src.utils import get_logger

logger = get_logger("kg.extractor")


class KGExtractor:
    """Extract entities and relations from text using an LLM."""

    EXTRACT_PROMPT = """从以下文本中抽取实体和关系。

按 JSON 格式输出：
{
  "entities": [
    {"name": "实体名", "type": "Person/Organization/Location/Concept/Event", "description": "简短描述"}
  ],
  "relations": [
    {"source": "实体A", "target": "实体B", "type": "RELATION_TYPE", "description": "关系描述"}
  ]
}

只输出 JSON，不要其他内容。

文本：
{text}
"""

    def __init__(
        self,
        llm: LocalLLM,
        max_entities_per_chunk: int = 20,
        max_relations_per_chunk: int = 30,
    ) -> None:
        self.llm = llm
        self.max_entities = max_entities_per_chunk
        self.max_relations = max_relations_per_chunk

    def extract(self, text: str, source_doc: str = "") -> Tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from a single text."""
        if not text.strip():
            return [], []
        prompt = f"""{{
  "entities": [
    {{"name": "实体名", "type": "Person/Organization/Location/Concept/Event", "description": "简短描述"}}
  ],
  "relations": [
    {{"source": "实体A", "target": "实体B", "type": "RELATION_TYPE", "description": "关系描述"}}
  ]
}}

只输出 JSON，不要其他内容。

文本：
{text}
"""
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.llm.chat, [{"role": "user", "content": prompt}], stream=False)
                raw = future.result(timeout=10)
        except Exception as exc:
            logger.warning("LLM 抽取超时或失败：%s", exc)
            return [], []
        return self._parse_llm_output(raw)

    def extract_from_chunks(
        self, chunks: List[Dict]
    ) -> Tuple[List[Entity], List[Relation], List[Triple]]:
        """Extract from chunk list, return entities + relations + triples."""
        entities: List[Entity] = []
        relations: List[Relation] = []
        triples: List[Triple] = []
        for chunk in chunks:
            text = chunk.get("text") or chunk.get("content") or ""
            source_doc = chunk.get("source_doc") or chunk.get("metadata", {}).get("source", "")
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            ents, rels = self.extract(text=text, source_doc=source_doc)
            entities.extend(ents)
            relations.extend(rels)
            for r in rels:
                triples.append(
                    Triple(
                        subject=r.source,
                        predicate=r.type,
                        object=r.target,
                        source_doc=source_doc,
                        confidence=r.weight,
                        chunk_id=chunk_id,
                    )
                )
        return entities, relations, triples

    def _parse_llm_output(self, raw: str) -> Tuple[List[Entity], List[Relation]]:
        """Parse LLM output, tolerating common malformed JSON."""
        if not raw:
            return [], []
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            return [], []
        candidate = cleaned[start : end + 1]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            candidate = re.sub(r"(\w)\s*:\s*", lambda m: f'"{m.group(1)}":', candidate)
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                return [], []
        entities = []
        relations = []
        for item in data.get("entities", []) or []:
            try:
                entities.append(
                    Entity(
                        name=str(item.get("name", "")),
                        type=str(item.get("type", "Other")),
                        description=str(item.get("description", "") or ""),
                    )
                )
            except Exception:
                continue
        for item in data.get("relations", []) or []:
            try:
                relations.append(
                    Relation(
                        source=str(item.get("source", "")),
                        target=str(item.get("target", "")),
                        type=str(item.get("type", "")),
                        description=str(item.get("description", "") or ""),
                    )
                )
            except Exception:
                continue
        return entities, relations
