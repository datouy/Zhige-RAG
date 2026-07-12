"""RAG 主流程：检索 → 重排 → 生成。

入口类：``RAGPipeline``

典型用法：

    >>> pipeline = RAGPipeline.from_config("config/config.yaml")
    >>> result = pipeline.answer("什么是 RAG？")
    >>> print(result.answer)
    >>> for s in result.sources:
    ...     print(s["cite"])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from .embeddings import EmbeddingModel
from .llm import GenerationConfig, LocalLLM
from .prompt_template import PromptTemplate
from .reranker import BgeReranker
from .utils import Timer, get_logger, load_config, merge_dict, resolve_path
from .vector_store import ChromaStore, Hit

logger = get_logger("rag_pipeline")


@dataclass
class RAGResult:
    """RAG 回答结果。"""

    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    raw_hits: List[Hit] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "timings": self.timings,
        }


class RAGPipeline:
    """RAG 端到端流程。

    Args:
        config: 已合并的配置 dict。
        embedding: EmbeddingModel 实例。
        vector_store: ChromaStore 实例。
        llm: LocalLLM 实例。
        reranker: 可选 BgeReranker 实例。
        prompts: PromptTemplate 实例。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        embedding: EmbeddingModel,
        vector_store: ChromaStore,
        llm: LocalLLM,
        reranker: Optional[BgeReranker] = None,
        prompts: Optional[PromptTemplate] = None,
    ) -> None:
        self.config = config
        self.embedding = embedding
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker
        self.prompts = prompts or PromptTemplate.from_yaml(
            config.get("rag", {}).get("system_prompt_file", "config/prompts.yaml")
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config_path: str = "config/config.yaml",
        overrides: Optional[Dict[str, Any]] = None,
        lazy_llm: bool = False,
    ) -> "RAGPipeline":
        """从配置文件构造完整流程。

        Args:
            config_path: 配置文件路径。
            overrides: 运行时覆盖（dict）。
            lazy_llm: 是否延迟加载 LLM（用于 UI 启动提速）。
        """
        cfg = load_config(config_path)
        if overrides:
            cfg = merge_dict(cfg, overrides)
        # 应用环境变量覆盖
        from .utils import apply_env_overrides

        cfg = apply_env_overrides(cfg)

        # 1. Embedding
        embedding = EmbeddingModel(
            model_name=cfg["embedding"]["model_name"],
            device=cfg["embedding"].get("device", "auto"),
            batch_size=cfg["embedding"].get("batch_size", 32),
            max_seq_length=cfg["embedding"].get("max_seq_length", 512),
            normalize=cfg["embedding"].get("normalize_embeddings", True),
            cache_dir=cfg["embedding"].get("cache_dir"),
            local_files_only=cfg["embedding"].get("local_files_only", False),
        )

        # 2. Vector Store
        vs = ChromaStore(
            persist_directory=cfg["vector_store"]["persist_directory"],
            collection_name=cfg["vector_store"].get("collection_name", "chinese_rag_kb"),
            embedding_model=embedding,
            distance_fn=cfg["vector_store"].get("distance_fn", "cosine"),
        )

        # 3. LLM
        llm = None
        if not lazy_llm:
            llm = LocalLLM(
                model_name=cfg["llm"]["model_name"],
                device=cfg["llm"].get("device", "auto"),
                device_map=cfg["llm"].get("device_map", "auto"),
                torch_dtype=cfg["llm"].get("torch_dtype", "auto"),
                quant=cfg["llm"].get("quantization"),
                cache_dir=cfg["llm"].get("cache_dir"),
                generation=cfg["llm"].get("generation"),
                chat_template=cfg["llm"].get("chat_template", "auto"),
                local_files_only=cfg["llm"].get("local_files_only", False),
                trust_remote_code=cfg["llm"].get("trust_remote_code", False),
            )

        # 4. Reranker（可选）
        reranker = None
        if cfg.get("reranker", {}).get("enabled"):
            reranker = BgeReranker(
                model_name=cfg["reranker"]["model_name"],
                device=cfg["reranker"].get("device", "auto"),
                cache_dir=cfg["reranker"].get("cache_dir", cfg["embedding"].get("cache_dir")),
            )

        # 5. Prompts
        prompts = PromptTemplate.from_yaml(cfg.get("rag", {}).get("system_prompt_file", "config/prompts.yaml"))

        return cls(cfg, embedding, vs, llm, reranker, prompts)

    # ------------------------------------------------------------------
    def ensure_llm(self) -> None:
        """按需加载 LLM。"""
        if self.llm is not None:
            return
        llm_cfg = self.config.get("llm", {})
        self.llm = LocalLLM(
            model_name=llm_cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            device=llm_cfg.get("device", "auto"),
            device_map=llm_cfg.get("device_map", "auto"),
            torch_dtype=llm_cfg.get("torch_dtype", "auto"),
            quant=llm_cfg.get("quantization"),
            cache_dir=llm_cfg.get("cache_dir"),
            generation=llm_cfg.get("generation"),
            chat_template=llm_cfg.get("chat_template", "auto"),
            local_files_only=llm_cfg.get("local_files_only", False),
            trust_remote_code=llm_cfg.get("trust_remote_code", False),
        )

    # ------------------------------------------------------------------
    def _retrieve(self, query: str, top_k: int, where: Optional[Dict[str, Any]]) -> List[Hit]:
        with Timer("retrieve"):
            hits = self.vector_store.query(
                query_text=query,
                top_k=top_k,
                where=where,
                score_threshold=self.config.get("retrieval", {}).get("score_threshold", 0.0),
            )
        return hits

    def _rerank(self, query: str, hits: List[Hit], top_n: int) -> List[Hit]:
        if self.reranker is None or not hits:
            return hits[:top_n]
        with Timer("rerank"):
            return self.reranker.rerank(query, hits, top_n=top_n)

    def _build_context(self, hits: List[Hit], max_chars: int) -> tuple[List[Dict[str, Any]], str]:
        """构造 prompt 中使用的 context。

        Returns:
            (context_chunks 用于 cite, 拼好的 context_str)
        """
        chunks: List[Dict[str, Any]] = []
        budget = max_chars
        for i, h in enumerate(hits, start=1):
            content = h.text
            title = h.metadata.get("title") or h.metadata.get("source") or f"片段{i}"
            if budget <= 0:
                break
            if len(content) > budget:
                content = content[: max(0, budget - 1)] + "…"
            chunks.append({"index": i, "title": title, "content": content, "metadata": h.metadata})
            budget -= len(content)
        return chunks, "\n\n".join(c["content"] for c in chunks)

    def _format_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cites: List[Dict[str, Any]] = []
        for c in chunks:
            meta = c.get("metadata", {}) or {}
            cite_text = self.prompts.format_citation(
                index=c["index"],
                title=c["title"],
                source=str(meta.get("source") or meta.get("filepath") or "未知"),
                page=meta.get("page", "—"),
            )
            cites.append(
                {
                    "index": c["index"],
                    "cite": cite_text,
                    "title": c["title"],
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "score": meta.get("_rerank_score"),
                    "snippet": c["content"][:200],
                }
            )
        return cites

    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> RAGResult | Generator[str, None, RAGResult]:
        """执行一次 RAG 问答。

        Args:
            question: 用户问题。
            top_k: 覆盖默认召回数量。
            where: metadata 过滤条件。
            stream: 是否流式返回。

        Returns:
            非流式：RAGResult；流式：字符串生成器（最终通过 .throw 返回 RAGResult）。
        """
        self.ensure_llm()
        retrieval_cfg = self.config.get("retrieval", {})
        rag_cfg = self.config.get("rag", {})

        k = top_k or retrieval_cfg.get("top_k", 5)
        # 重排候选数 = max(k, top_n*3)
        if self.reranker is not None:
            initial_k = max(k, retrieval_cfg.get("rerank_candidates", retrieval_cfg.get("top_k", 5) * 3))
        else:
            initial_k = k

        hits = self._retrieve(question, top_k=initial_k, where=where)
        if self.reranker is not None:
            hits = self._rerank(question, hits, top_n=k)

        if not hits:
            fallback = rag_cfg.get(
                "fallback_answer",
                "抱歉，根据当前知识库我没有找到相关信息。",
            )
            if stream:
                def _empty_gen(_q=question, _fb=fallback):
                    yield _fb
                    return RAGResult(answer=_fb, sources=[], raw_hits=[], timings={})
                return _empty_gen()
            return RAGResult(answer=fallback, sources=[], raw_hits=[])

        max_ctx = rag_cfg.get("max_context_tokens", 2048)
        context_chunks, _ctx_str = self._build_context(hits, max_chars=max_ctx * 2)  # 中文字符与 token 约 2:1
        messages = self.prompts.build_messages(question, context_chunks)
        gen_cfg = self.llm.gen_cfg

        if not stream:
            with Timer("llm") as t:
                text = self.llm.chat(messages, gen_cfg, stream=False)
            cites = self._format_citations(context_chunks)
            # 非流式路径无法精确拆分 TTFT：以"全部耗时"近似，TTFT 记为 None
            result = RAGResult(answer=text.strip(), sources=cites, raw_hits=hits)
            result.timings["llm_total_ms"] = round(t.elapsed_ms, 2)
            result.timings["ttft_ms"] = None
            result.timings["tokens_generated"] = None
            result.timings["tokens_per_sec"] = None
            return result

        # 流式：先抛 chunks，最后用 throw 返回完整 RAGResult
        def _stream():
            nonlocal ttft_ms
            buf = ""
            try:
                for piece in self.llm.chat(messages, gen_cfg, stream=True):
                    if ttft_ms is None:
                        # 首个 token 到达时刻
                        ttft_ms = (time.perf_counter() - llm_start) * 1000.0
                    buf += piece
                    yield piece
            finally:
                # 收尾时构造完整 result（用户可通过 gen.send 拿到但 UI 一般不需要）
                pass

        llm_start = time.perf_counter()
        ttft_ms: Optional[float] = None
        return _stream()  # UI 中只需迭代字符串片段即可

    # ------------------------------------------------------------------
    def answer_with_timing(
        self,
        question: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> RAGResult:
        """强制走流式路径并记录首 token 延迟 (TTFT) 与生成 token 数。

        适合评估脚本使用：始终返回包含 ``timings`` 字段的 ``RAGResult``，
        不破坏 ``answer()`` 既有调用方。

        timings 字段（单位 ms）:
        - ``llm_total_ms``: LLM 生成总耗时
        - ``ttft_ms``: 首 token 延迟；未流式成功时为 ``None``
        - ``tokens_generated``: 生成 token 数（来自 tokenizer 编码）；
          当无法精确计数时退回为字符数估算
        - ``tokens_per_sec``: tokens / (llm_total - ttft) 的速率
        """
        self.ensure_llm()
        retrieval_cfg = self.config.get("retrieval", {})
        rag_cfg = self.config.get("rag", {})

        k = top_k or retrieval_cfg.get("top_k", 5)
        if self.reranker is not None:
            initial_k = max(k, retrieval_cfg.get("rerank_candidates", k * 3))
        else:
            initial_k = k

        hits = self._retrieve(question, top_k=initial_k, where=where)
        if self.reranker is not None:
            hits = self._rerank(question, hits, top_n=k)

        if not hits:
            fallback = rag_cfg.get(
                "fallback_answer",
                "抱歉，根据当前知识库我没有找到相关信息。",
            )
            return RAGResult(
                answer=fallback,
                sources=[],
                raw_hits=[],
                timings={
                    "llm_total_ms": 0.0,
                    "ttft_ms": None,
                    "tokens_generated": 0,
                    "tokens_per_sec": None,
                },
            )

        max_ctx = rag_cfg.get("max_context_tokens", 2048)
        context_chunks, _ctx_str = self._build_context(hits, max_chars=max_ctx * 2)
        messages = self.prompts.build_messages(question, context_chunks)
        gen_cfg = self.llm.gen_cfg
        cites = self._format_citations(context_chunks)

        llm_start = time.perf_counter()
        ttft_ms: Optional[float] = None
        buf_parts: List[str] = []
        try:
            for piece in self.llm.chat(messages, gen_cfg, stream=True):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - llm_start) * 1000.0
                buf_parts.append(piece)
        except Exception as exc:
            logger.warning("流式生成失败，回退到非流式：%s", exc)
            with Timer("llm") as t:
                text = self.llm.chat(messages, gen_cfg, stream=False)
            llm_total = t.elapsed_ms
            buf_parts = [text]
            ttft_ms = None
        else:
            llm_total = (time.perf_counter() - llm_start) * 1000.0

        full_text = "".join(buf_parts).strip()
        tokens_generated, tokens_estimated = self._estimate_tokens(full_text)
        gen_window_ms = (llm_total - ttft_ms) if (ttft_ms is not None and llm_total > ttft_ms) else None
        if tokens_generated and gen_window_ms and gen_window_ms > 0:
            tokens_per_sec = tokens_generated / (gen_window_ms / 1000.0)
        else:
            tokens_per_sec = None

        return RAGResult(
            answer=full_text,
            sources=cites,
            raw_hits=hits,
            timings={
                "llm_total_ms": round(llm_total, 2),
                "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                "tokens_generated": tokens_generated,
                "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec else None,
                "tokens_estimated": tokens_estimated,
            },
        )

    def _estimate_tokens(self, text: str) -> tuple[int, bool]:
        """估算 token 数。优先用 tokenizer.encode 精确计数，失败时按"1 token ≈ 1.5 中文字符"回退。

        Returns:
            (token_count, is_estimated)
        """
        if not text:
            return 0, False
        try:
            tok = getattr(self.llm, "tokenizer", None)
            if tok is not None:
                ids = tok.encode(text, add_special_tokens=False)
                return len(ids), False
        except Exception:
            pass
        # 回退：粗略估算（中文字符 ≈ 1.5 token；英文按空格分词）
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        other = len(text) - cjk
        est = int(cjk / 1.5 + max(1, other // 4))
        return est, True

    # ------------------------------------------------------------------
    def stream_answer(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """流式问答的高级封装：返回 ``{"event":..., "data":...}`` 事件流。

        事件类型：
        - ``hit``: 命中来源（最早发出）
        - ``token``: 生成 token
        - ``done``: 结束（data 含 cites）
        - ``error``: 异常
        """
        self.ensure_llm()
        retrieval_cfg = self.config.get("retrieval", {})
        k = top_k or retrieval_cfg.get("top_k", 5)
        hits = self._retrieve(question, top_k=k, where=None)
        if self.reranker is not None:
            hits = self._rerank(question, hits, top_n=k)

        # 来源事件
        max_ctx = self.config.get("rag", {}).get("max_context_tokens", 2048)
        context_chunks, _ = self._build_context(hits, max_chars=max_ctx * 2)
        cites = self._format_citations(context_chunks)

        yield {"event": "hits", "data": cites}

        if not hits:
            fallback = self.config.get("rag", {}).get(
                "fallback_answer",
                "抱歉，根据当前知识库我没有找到相关信息。",
            )
            yield {"event": "token", "data": fallback}
            yield {"event": "done", "data": cites}
            return

        messages = self.prompts.build_messages(question, context_chunks)
        try:
            for piece in self.llm.chat(messages, self.llm.gen_cfg, stream=True):
                yield {"event": "token", "data": piece}
        except Exception as exc:
            yield {"event": "error", "data": str(exc)}
            return
        yield {"event": "done", "data": cites}