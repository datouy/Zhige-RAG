"""LLM-based knowledge graph extractor."""

from __future__ import annotations

import json
import re
import threading
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
        extract_timeout: float = 60.0,
    ) -> None:
        self.llm = llm
        self.max_entities = max_entities_per_chunk
        self.max_relations = max_relations_per_chunk
        # 单次抽取超时（秒）。本地 1.5B 模型 10 秒经常跑不完，默认放宽到 60。
        self.extract_timeout = extract_timeout

    def extract(self, text: str, source_doc: str = "") -> Tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from a single text."""
        if not text or not text.strip():
            return [], []
        # 控制输入长度，避免超长文本打爆 prompt
        text = text.strip()
        if len(text) > 6000:
            text = text[:6000]
        # EXTRACT_PROMPT 内含 JSON 大括号，不能走 str.format，用替换注入
        prompt = self.EXTRACT_PROMPT.replace("{text}", text)
        # 用 daemon 线程跑 LLM 调用并 join(timeout)：超时后主流程立即返回。
        # 之前的 ThreadPoolExecutor 写法在 with 块退出时 shutdown(wait=True)
        # 仍会阻塞到生成结束，超时形同虚设。
        holder: Dict[str, object] = {}

        def _run() -> None:
            try:
                holder["raw"] = self.llm.chat(
                    [{"role": "user", "content": prompt}], stream=False
                )
            except Exception as exc:  # noqa: BLE001
                holder["err"] = exc

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(timeout=max(1.0, float(self.extract_timeout)))
        if worker.is_alive():
            logger.warning("LLM 抽取超时（%ss），放弃本次结果", self.extract_timeout)
            return [], []
        if "err" in holder:
            logger.warning("LLM 抽取失败：%s", holder["err"])
            return [], []
        return self._parse_llm_output(holder.get("raw"))

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
