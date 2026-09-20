"""GraphRAG integration: combine vector retrieval and graph expansion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.kg.retriever import GraphRetriever
from src.rag_pipeline import RAGPipeline
from src.vector_store import ChromaStore


class GraphRAG:
    """Graph-augmented RAG."""

    def __init__(
        self,
        vector_store: ChromaStore,
        kg_store,
        extractor,
        retriever: GraphRetriever,
        llm,
    ) -> None:
        self.vector_store = vector_store
        self.kg_store = kg_store
        self.extractor = extractor
        self.retriever = retriever
        self.llm = llm

    def query(self, question: str, top_k: int = 4, graph_hops: int = 2) -> Dict[str, Any]:
        """GraphRAG 问答：向量检索 + 图谱扩展，查询路径只读。

        注意：不要在查询时做 KG 抽取——那需要对每个命中 chunk 各调一次 LLM
        （查询延迟被抽取主导），还会把查询期的低质量抽取写进图谱污染数据。
        实体/关系的抽取与入库应发生在文档入库或 ``kg/build`` 阶段。
        """
        hits = self.vector_store.query(query_text=question, top_k=top_k)
        texts = [h.text for h in hits]
        graph_context = self.retriever.search(question, top_k_entities=top_k, hops=graph_hops)
        context_parts = [f"[文档片段 {i+1}]\n{t}" for i, t in enumerate(texts)]
        if graph_context.get("triples"):
            context_parts.append("[知识图谱]\n" + "\n".join(self._format_triples(graph_context["triples"])))
        context_str = "\n\n".join(context_parts)
        messages = [
            {"role": "system", "content": "你是一个知识库问答助手。请基于已知上下文作答。"},
            {"role": "user", "content": f"问题：{question}\n\n上下文：\n{context_str}"},
        ]
        try:
            answer = self.llm.chat(messages, stream=False)
        except Exception as exc:  # noqa: BLE001
            from src.utils import get_logger

            get_logger("kg.graph_rag").warning("GraphRAG 生成失败：%s", exc)
            answer = "（生成失败）"
        return {
            "answer": answer,
            "sources": [{"id": h.id, "score": h.score, "metadata": h.metadata, "text": h.text[:300]} for h in hits],
            "graph_context": graph_context,
        }

    def _format_triples(self, triples: List[Triple], limit: int = 40) -> List[str]:
        out: List[str] = []
        for t in triples[:limit]:
            out.append(f"{t.subject} -[{t.predicate}]-> {t.object}")
        return out
