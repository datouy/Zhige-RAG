"""Knowledge Graph schema dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Entity:
    """知识图谱实体。"""

    name: str
    type: str = "Other"  # Person / Organization / Location / Concept / Event / Other
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """知识图谱关系。"""

    source: str  # 源实体名
    target: str  # 目标实体名
    type: str = "RELATED_TO"  # 关系类型（如 WORKS_FOR, LOCATED_IN, CAUSES）
    description: str = ""
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Triple:
    """三元组（事实）。"""

    subject: str
    predicate: str = "RELATED_TO"
    object: str = ""
    source_doc: str = ""  # 来源文档
    confidence: float = 1.0
    chunk_id: Optional[str] = None
