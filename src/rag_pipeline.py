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

import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from .embeddings import EmbeddingModel
from .llm import GenerationConfig, LocalLLM
from .memory import SESSION_MEMORY, ContextAssembler, rewrite_query_for_retrieval
from .prompt_template import PromptTemplate
from .query_router import classify_query, is_aggregate_query
from .reranker import BgeReranker
from .utils import Timer, get_logger, load_config, merge_dict, resolve_path
from .vector_store import ChromaStore, Hit, hybrid_kwargs

logger = get_logger("rag_pipeline")

# 进程级即时记忆单例从 memory 模块导入（SESSION_MEMORY），
# 供 ui / 脚本统一使用同一会话窗口。

# ----------------------------------------------------------------------
#  Constants
# ----------------------------------------------------------------------
# 检索为空 / 未命中时返回的兜底文本（P3.4 集中化）。
# 之前散落在 ``answer()`` / ``stream_answer()`` / ``async_answer()`` 三个
# 入口里；统一抽常量便于 i18n / 修改文案时只改一处。
FALLBACK_NO_HIT_ANSWER = "抱歉，根据当前知识库我没有找到相关信息。"

# LLM 调用默认超时（秒）
DEFAULT_LLM_TIMEOUT = 120.0
# 最大重试次数
DEFAULT_MAX_RETRIES = 2

# 检索命中最低分兜底：所有命中都低于该相似度时按"未命中"处理，宁可拒答
# 也不让小模型对着无关上下文编造。BM25 独有命中已归一到 (0,1)（s/(1+s)），
# 强词面命中可达 0.7+，不会被误伤；阈值取保守值，只拦截明显无关的查询。
DEFAULT_MIN_SCORE = 0.25


def normalize_query(query: str) -> str:
    """查询归一化：全角字母/数字/符号 → 半角（NFKC），合并空白。

    用户在中文输入法下常打出"ＲＡＧ""１２３"或夹带全角问号——向量编码
    与 BM25 词法都会因全/半角差异失配，先归一再检索。
    """
    if not query:
        return query
    q = unicodedata.normalize("NFKC", query)
    return re.sub(r"\s+", " ", q).strip()


@dataclass
class RAGResult:
    """RAG 回答结果。

    ``verification`` 为可验证性报告（``src/verifier.verify_answer`` 产物），
    None 表示校验未启用或该路径未做校验。
    """

    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    raw_hits: List[Hit] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    verification: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "timings": self.timings,
            "verification": self.verification,
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
        long_term_store: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.embedding = embedding
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker
        self.prompts = prompts or PromptTemplate.from_yaml(
            config.get("rag", {}).get("system_prompt_file", "config/prompts.yaml")
        )
        # 三层上下文组装器（即时/检索/长期记忆 + 运行时变量）
        self.context_assembler = ContextAssembler(self.prompts, config.get("memory"))
        # 长期记忆存储（默认 SQLite SessionLocal；测试可注入替身）
        if long_term_store is None:
            try:
                from .memory import LongTermMemoryStore

                self.long_term_store: Optional[Any] = LongTermMemoryStore()
            except Exception as exc:  # noqa: BLE001
                logger.warning("长期记忆存储初始化失败，长期记忆将不可用: %s", exc)
                self.long_term_store = None
        else:
            self.long_term_store = long_term_store
        # GraphRAG 懒加载状态（_get_graph_retriever 首次调用时探测）
        self._graph_state_checked = False
        self._graph_retriever: Optional[Any] = None

    # ------------------------------------------------------------------
    def _relevance_hint(self, hits: List[Hit]) -> str:
        """检索相关性偏低时给生成模型一条防编造提示（小白测试 A2 护栏）。"""
        if not hits:
            return ""
        try:
            top = max(h.score for h in hits)
        except ValueError:
            return ""
        if top < 0.45:
            return (
                "〔提示：本次检索到的内容与问题相关性较低。若上下文确实与问题无关，"
                "必须直接回答无法回答，禁止编造。〕"
            )
        return ""

    def _runtime_context(self, hits: Optional[List[Hit]] = None) -> Dict[str, str]:
        """运行时变量：注入 prompt 而非写死（换企业名/助手名只改配置）。

        关键信息不写死在 prompt、也不依赖模型训练——"今天几号"这类问题
        由注入的 current_date 保证，而不是指望模型记住训练截止日期。
        """
        rag_cfg = self.config.get("rag", {})
        app_cfg = self.config.get("app", {})
        return {
            "assistant_name": str(rag_cfg.get("assistant_name") or "小识"),
            "org_name": str(rag_cfg.get("org_name") or app_cfg.get("name") or "企业知识库"),
            "current_date": datetime.now().strftime("%Y年%m月%d日"),
            "relevance_hint": self._relevance_hint(hits or []),
        }

    # ------------------------------------------------------------------
    def _security_where(
        self,
        extra_where: Optional[Dict[str, Any]],
        allowed_groups: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """构造数据层安全过滤条件（doc_status + ACL），随检索下发。

        - ``retrieval.status_filter``（默认开）：只召回 doc_status=active 的分块；
          若开启 fallback 且过滤后为空，回退一次不过滤并告警（兼容旧索引
          无 doc_status 元数据的情况——ACL 过滤绝不回退）。
        - ``retrieval.acl_filter``（默认开）：只召回 acl 命中用户可见范围的分块。
        """
        retrieval_cfg = self.config.get("retrieval", {})
        from .vector_store import build_security_where

        return build_security_where(
            extra_where=extra_where,
            status_filter=bool(retrieval_cfg.get("status_filter", True)),
            acl_groups=allowed_groups if retrieval_cfg.get("acl_filter", True) else None,
        )

    def _retrieve(
        self,
        query: str,
        top_k: int,
        where: Optional[Dict[str, Any]],
        allowed_groups: Optional[List[str]] = None,
    ) -> List[Hit]:
        security_where = self._security_where(where, allowed_groups)
        norm_query = normalize_query(query)
        with Timer("retrieve"):
            hits = self.vector_store.query(
                query_text=self._apply_query_instruction(norm_query),
                top_k=top_k,
                where=security_where,
                score_threshold=self.config.get("retrieval", {}).get("score_threshold", 0.0),
            )
        # 旧索引兼容：doc_status 过滤导致全空时回退一次（仅状态过滤，ACL 不回退）
        if (
            not hits
            and security_where is not None
            and self.config.get("retrieval", {}).get("status_filter", True)
            and self.config.get("retrieval", {}).get("status_filter_fallback", True)
        ):
            with Timer("retrieve_unfiltered_fallback"):
                hits = self.vector_store.query(
                    query_text=self._apply_query_instruction(norm_query),
                    top_k=top_k,
                    where=where,
                    score_threshold=self.config.get("retrieval", {}).get("score_threshold", 0.0),
                )
            if hits:
                logger.warning(
                    "doc_status 过滤后 0 命中，已回退为不过滤检索（%d 条）。"
                    "说明索引中存在无 doc_status 元数据的旧分块，请重新入库以启用状态过滤。",
                    len(hits),
                )
        # 低分兜底：最高分仍低于阈值 → 视为未命中，走拒答文案而不是硬答。
        # 过滤回退路径产生的命中同样受此约束，杜绝"为答而答"。
        min_score = float(
            self.config.get("retrieval", {}).get("min_score", DEFAULT_MIN_SCORE) or 0.0
        )
        if hits and min_score > 0 and max(h.score for h in hits) < min_score:
            logger.info(
                "检索最高分 %.3f 低于 min_score=%.3f，按未命中处理（查询：%s）",
                max(h.score for h in hits),
                min_score,
                norm_query[:50],
            )
            return []
        return hits

    def _verify(self, answer: str, context_chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """可验证性校验（可经 rag.verification.enabled 关闭）。"""
        if not self.config.get("rag", {}).get("verification", {"enabled": True}).get("enabled", True):
            return None
        from .verifier import verify_answer

        fallback = self.config.get("rag", {}).get("fallback_answer", FALLBACK_NO_HIT_ANSWER)
        min_ratio = float(self.config.get("rag", {}).get("verification", {}).get("min_cited_ratio", 0.5))
        min_grounding = float(self.config.get("rag", {}).get("verification", {}).get("min_grounding", 0.6))
        return verify_answer(
            answer,
            context_chunks,
            fallback_answer=fallback,
            min_cited_ratio=min_ratio,
            min_grounding=min_grounding,
        ).to_dict()

    def _recall_long_term(self, user_id: Optional[str]) -> List[Dict[str, str]]:
        """召回用户长期记忆（未启用/无 user_id/存储不可用时返回空）。"""
        if not user_id or not self.context_assembler.enabled or self.long_term_store is None:
            return []
        try:
            return self.long_term_store.recall(user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("长期记忆召回失败（已忽略）: %s", exc)
            return []

    # ------------------------------------------------------------------
    def _render_chitchat(self, template: str) -> str:
        """用运行时变量渲染寒暄模板（助手名/企业名不写死在代码里）。"""
        try:
            return template.format(**self._runtime_context())
        except (KeyError, IndexError):
            return template

    def _get_graph_retriever(self) -> Optional[Any]:
        """懒加载 GraphRetriever；未启用或图谱为空（未构建）时返回 None。

        空库守卫很重要：kg.db 是入库期由 build_kg.py 离线构建的，主链路
        只读不写；图谱没建过时静默跳过增强，绝不阻塞正常问答。
        """
        if self._graph_state_checked:
            return self._graph_retriever
        self._graph_state_checked = True
        kg_cfg = self.config.get("knowledge_graph", {}) or {}
        if not kg_cfg.get("enabled") or not (kg_cfg.get("graph_rag", {}) or {}).get("enabled"):
            return None
        try:
            from .kg import GraphRetriever, create_kg_store

            store = create_kg_store(kg_cfg)
            counts = store.count() or {}
            if int(counts.get("entities", 0) or 0) <= 0:
                logger.info(
                    "知识图谱为空，GraphRAG 增强停用（运行 python scripts/build_kg.py 构建图谱后自动生效）"
                )
                return None
            self._graph_retriever = GraphRetriever(store, embedding_model=self.embedding)
            logger.info("GraphRAG 已启用：%s", counts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GraphRAG 初始化失败，跳过图谱增强: %s", exc)
        return self._graph_retriever

    def _augment_with_graph(
        self, question: str, context_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """实体/关系类问题用知识图谱三元组补充上下文（查询期只读）。

        三元组作为一条附加 context（编号顺延）进入 prompt，让小模型能回答
        "X 和 Y 是什么关系"这类跨片段问题；与文档片段冲突时 prompt 中已
        声明以文档为准。图谱检索失败时静默降级为纯文档上下文。
        """
        retriever = self._get_graph_retriever()
        if retriever is None or not context_chunks:
            return context_chunks
        kg_cfg = self.config.get("knowledge_graph", {}) or {}
        gr_cfg = kg_cfg.get("graph_rag", {}) or {}
        try:
            sub = retriever.search(
                normalize_query(question),
                top_k_entities=int((kg_cfg.get("retriever", {}) or {}).get("top_k_entities", 5)),
                hops=int(gr_cfg.get("graph_hops", 2)),
            )
            triples = sub.get("triples") or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("图谱检索失败（已跳过图谱增强）: %s", exc)
            return context_chunks
        if not triples:
            return context_chunks
        lines = [
            f"{t.subject} -[{t.predicate}]-> {t.object}"
            for t in triples[: max(1, int(gr_cfg.get("max_triples", 10)))]
        ]
        context_chunks.append(
            {
                "index": len(context_chunks) + 1,
                "title": "知识图谱",
                "content": (
                    "以下是知识图谱中与问题相关的实体关系（供参考；与文档片段冲突时以文档片段为准）：\n"
                    + "\n".join(lines)
                ),
                "metadata": {"source": "知识图谱", "page": "—"},
            }
        )
        logger.info("GraphRAG 增强：追加 %d 条三元组上下文", len(lines))
        return context_chunks

    # ------------------------------------------------------------------
    def _structured_count_answer(self, question: str, hits: List[Hit]) -> Optional[RAGResult]:
        """聚合统计类问题（"引用了多少文献"）的结构化直接作答。

        分块检索 + 小参数量模型对"数条数"类问题不可靠：上下文只见列表
        碎片，且正文引用角标 [1][2] 会诱导模型去数角标。入库时已对编号
        条目小节（参考文献等）计算了可信条目数（metadata.entry_count），
        此处直接以结构化元数据作答，绕过 LLM 生成。问题不含统计意图或
        检索结果无条目注记时返回 None，走正常生成路径。
        """
        if not hits or not is_aggregate_query(question or ""):
            return None
        cand: List[tuple] = []
        for h in hits:
            ec = str((h.metadata or {}).get("entry_count") or "").strip()
            if ec.isdigit() and int(ec) > 0:
                cand.append((h, int(ec)))
        if not cand:
            return None
        best, n = max(cand, key=lambda t: t[1])
        md = best.metadata or {}
        prefix = str(md.get("chunk_prefix") or md.get("breadcrumb") or "")
        section = prefix.split("｜")[0].split(" > ")[-1].strip() or "编号条目列表"
        src = str(md.get("source") or "知识库文档")
        span_m = re.search(r"\[(\d+)\]\s*-\s*\[(\d+)\]", prefix)
        span = f"（编号[{span_m.group(1)}]-[{span_m.group(2)}]）" if span_m else ""
        # 注意：不要把文件名拼进答案——文件名不在分块文本里，会让 verifier
        # 的词面接地率检查误判为编造（实测"共30条"正确答案被判 ❌）。
        answer = f"根据知识库检索结果，「{section}」部分为编号条目列表，共 {n} 条{span}。 [1]"
        chunks = [{"index": 1, "title": f"{section}（共{n}条）", "content": best.text, "metadata": md}]
        result = RAGResult(answer=answer, sources=self._format_citations(chunks), raw_hits=[best])
        result.timings = {"llm_total_ms": 0.0, "ttft_ms": None, "tokens_generated": 0, "tokens_per_sec": None}
        result.verification = self._verify(answer, chunks)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _build_llm_from_cfg(llm_cfg: Dict[str, Any]) -> "LocalLLM":
        """根据 ``cfg.llm`` 子 dict 构造 ``LocalLLM`` 实例。

        P1.4: ``from_config`` 与 ``ensure_llm`` 共用同一构造逻辑，避免字段
        漂移（device_map / torch_dtype / 量化参数等）。
        """
        gen_cfg = llm_cfg.get("generation", {})
        return LocalLLM(
            model_name=llm_cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            device=llm_cfg.get("device", "auto"),
            device_map=llm_cfg.get("device_map", "auto"),
            torch_dtype=llm_cfg.get("torch_dtype", "auto"),
            quant=llm_cfg.get("quantization"),
            cache_dir=llm_cfg.get("cache_dir"),
            generation=GenerationConfig(
                max_new_tokens=gen_cfg.get("max_new_tokens", 512),
                temperature=gen_cfg.get("temperature", 0.7),
                top_p=gen_cfg.get("top_p", 0.8),
                repetition_penalty=gen_cfg.get("repetition_penalty", 1.05),
                do_sample=gen_cfg.get("do_sample", True),
            ),
            chat_template=llm_cfg.get("chat_template", "auto"),
            local_files_only=llm_cfg.get("local_files_only", False),
            trust_remote_code=llm_cfg.get("trust_remote_code", False),
            max_retries=llm_cfg.get("max_retries", DEFAULT_MAX_RETRIES),
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
            **hybrid_kwargs(cfg.get("vector_store", {})),
        )

        # 3. LLM（P1.4：共用 _build_llm_from_cfg）
        llm = None
        if not lazy_llm:
            llm = cls._build_llm_from_cfg(cfg.get("llm", {}))

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
        # P1.4：与 from_config 共用构造逻辑，避免字段漂移
        self.llm = self._build_llm_from_cfg(self.config.get("llm", {}))

    def _apply_query_instruction(self, query: str) -> str:
        """按配置给查询拼接 BGE 检索指令。

        bge-*-zh-v1.5 官方建议：短查询 → 长段落检索时，查询侧需加
        ``为这个句子生成表示以用于检索相关文章：`` 指令（文档侧不加）。
        指令配置在 ``embedding.query_instruction``，之前一直没被使用。
        """
        instruction = (self.config.get("embedding", {}) or {}).get("query_instruction") or ""
        if not instruction or not query or query.startswith(instruction):
            return query
        return f"{instruction}{query}"

    def _rerank(self, query: str, hits: List[Hit], top_n: int) -> List[Hit]:
        if self.reranker is None or not hits:
            return hits[:top_n]
        with Timer("rerank"):
            ranked = self.reranker.rerank(query, hits, top_n=top_n)
        # 重排后二次过滤（A2 硬护栏）：重排分是 0~1 的相关度，实测无关问题
        # 全部 < 0.1、相关问题 > 0.9。低于阈值 → 按未命中处理，从源头掐断
        # "带引用的编造答案"。
        min_rel = float(
            (self.config.get("reranker", {}) or {}).get("min_relevance", 0.3) or 0.0
        )
        if min_rel > 0:
            kept = [h for h in ranked if h.score >= min_rel]
            if not kept:
                logger.info(
                    "重排后最高分 %.3f 低于 min_relevance=%.2f，全部过滤（查询：%s）",
                    max((h.score for h in ranked), default=0.0),
                    min_rel,
                    query[:50],
                )
                return []
            return kept
        return ranked

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        allowed_groups: Optional[List[str]] = None,
    ) -> List[Hit]:
        """公开检索入口：意图路由之后的"召回 + 重排"，不生成答案。

        只依赖 embedding（可选 reranker），不加载 LLM——供评估脚本在
        CPU 环境做检索回归（``scripts/evaluate.py --retrieval-only``），
        以及调试检索质量。
        """
        retrieval_cfg = self.config.get("retrieval", {})
        k = top_k or retrieval_cfg.get("top_k", 5)
        if self.reranker is not None:
            initial_k = max(k, retrieval_cfg.get("rerank_candidates", k * 3))
        else:
            initial_k = k
        hits = self._retrieve(question, top_k=initial_k, where=where, allowed_groups=allowed_groups)
        if self.reranker is not None:
            hits = self._rerank(normalize_query(question), hits, top_n=k)
        return hits

    def _build_context(self, hits: List[Hit], max_chars: int) -> tuple[List[Dict[str, Any]], str]:
        """构造 prompt 中使用的 context。

        Returns:
            (context_chunks 用于 cite, 拼好的 context_str)
        """
        chunks: List[Dict[str, Any]] = []
        budget = max_chars
        for i, h in enumerate(hits, start=1):
            content = h.text
            meta = h.metadata or {}
            # 标题优先用章节名（breadcrumb/chunk_prefix 末级）：文档标题在
            # 每块重复出现会诱导小模型把它当成"文献列表"复读；章节名才
            # 携带块的真实身份（如"参考文献"）。
            section = ""
            for key in ("chunk_prefix", "breadcrumb", "heading_path"):
                val = meta.get(key)
                if not val:
                    continue
                if isinstance(val, list):
                    val = val[-1] if val else ""
                section = str(val).split(" > ")[-1].split("｜")[0].strip()
                if section:
                    break
            title = section or meta.get("title") or meta.get("source") or f"片段{i}"
            # 编号条目小节（如参考文献）：标题直接带条目数，聚合统计类
            # 问题（"引用了多少文献"）在上下文中即有可靠答案可引用
            entry_count = str(meta.get("entry_count") or "").strip()
            if entry_count and section:
                title = f"{section}（共{entry_count}条）"
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
        history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None,
        allowed_groups: Optional[List[str]] = None,
    ) -> RAGResult | Generator[str, None, RAGResult]:
        """执行一次 RAG 问答。

        Args:
            question: 用户问题。
            top_k: 覆盖默认召回数量。
            where: metadata 过滤条件（在数据层安全过滤之上追加）。
            stream: 是否流式返回。
            history: 即时记忆——当前会话最近几轮对话（OpenAI 消息格式）。
            user_id: 提供时将召回该用户的长期记忆注入 system 上下文。
            allowed_groups: 用户可见的 acl 分组（``*`` 由调用方并入）。

        Returns:
            非流式：RAGResult；流式：字符串生成器（最终通过 .throw 返回 RAGResult）。
        """
        rag_cfg = self.config.get("rag", {})
        history = history or []

        # 意图路由：寒暄/身份类直接模板回应，不花检索与生成资源（也无需加载 LLM）
        decision = classify_query(question)
        if decision.intent == "chitchat" and not is_aggregate_query(question):
            reply = self._render_chitchat(decision.reply or "")
            if stream:
                def _cc_gen(_r=reply):
                    yield _r
                    return RAGResult(answer=_r, sources=[], raw_hits=[], timings={"llm_total_ms": 0.0})
                return _cc_gen()
            return RAGResult(answer=reply, sources=[], raw_hits=[], timings={"llm_total_ms": 0.0})

        self.ensure_llm()
        retrieval_cfg = self.config.get("retrieval", {})
        k = top_k or retrieval_cfg.get("top_k", 5)
        # 重排候选数 = max(k, top_n*3)
        if self.reranker is not None:
            initial_k = max(k, retrieval_cfg.get("rerank_candidates", retrieval_cfg.get("top_k", 5) * 3))
        else:
            initial_k = k

        # 多轮指代消解：仅影响检索查询，不改变生成阶段呈现的原始问题
        search_query = (
            rewrite_query_for_retrieval(question, history)
            if self.context_assembler.enabled and history
            else question
        )
        hits = self._retrieve(search_query, top_k=initial_k, where=where, allowed_groups=allowed_groups)
        if self.reranker is not None:
            hits = self._rerank(question, hits, top_n=k)

        if not hits:
            fallback = rag_cfg.get(
                "fallback_answer",
                FALLBACK_NO_HIT_ANSWER,
            )
            if stream:
                def _empty_gen(_q=question, _fb=fallback):
                    yield _fb
                    return RAGResult(answer=_fb, sources=[], raw_hits=[], timings={})
                return _empty_gen()
            result = RAGResult(answer=fallback, sources=[], raw_hits=[])
            result.verification = self._verify(fallback, [])
            return result

        # 聚合统计类问题（条目数等）：结构化元数据直接作答，不依赖小模型数数
        structured = self._structured_count_answer(question, hits)
        if structured is not None:
            if stream:
                def _st_gen(_r=structured):
                    yield _r.answer
                    return _r
                return _st_gen()
            return structured

        max_ctx = rag_cfg.get("max_context_tokens", 2048)
        context_chunks, _ctx_str = self._build_context(hits, max_chars=max_ctx * 2)  # 中文字符与 token 约 2:1
        context_chunks = self._augment_with_graph(question, context_chunks)
        long_term = self._recall_long_term(user_id)
        messages = self.context_assembler.build(
            question,
            context_chunks,
            history=history,
            long_term=long_term,
            runtime=self._runtime_context(hits),
        )
        gen_cfg = self.llm.gen_cfg

        # 获取 LLM 配置中的超时和重试参数
        llm_cfg = self.config.get("llm", {})
        timeout = llm_cfg.get("timeout", DEFAULT_LLM_TIMEOUT)
        max_retries = llm_cfg.get("max_retries", DEFAULT_MAX_RETRIES)

        if not stream:
            llm_start = time.perf_counter()
            retries_used = 0
            try:
                text = self.llm.chat(messages, gen_cfg, stream=False, timeout=timeout)
            except Exception as llm_err:
                logger.error("LLM 调用最终失败: %s", llm_err)
                text = rag_cfg.get("fallback_answer", FALLBACK_NO_HIT_ANSWER)
                retries_used = max_retries  # 标记为已重试
            llm_total_ms = (time.perf_counter() - llm_start) * 1000
            cites = self._format_citations(context_chunks)
            result = RAGResult(answer=text.strip() if isinstance(text, str) else str(text), sources=cites, raw_hits=hits)
            result.timings["llm_total_ms"] = round(llm_total_ms, 2)
            result.timings["ttft_ms"] = None
            result.timings["tokens_generated"] = None
            result.timings["tokens_per_sec"] = None
            result.timings["retries"] = retries_used
            result.verification = self._verify(result.answer, context_chunks)
            return result

        # 流式：先抛 chunks，最后用 throw 返回完整 RAGResult
        llm_start = time.perf_counter()
        ttft_ms: Optional[float] = None
        retries_used = 0

        def _stream():
            nonlocal ttft_ms, retries_used
            buf = ""
            try:
                for piece in self.llm.chat(messages, gen_cfg, stream=True):
                    if ttft_ms is None:
                        # 首个 token 到达时刻
                        ttft_ms = (time.perf_counter() - llm_start) * 1000.0
                    buf += piece
                    yield piece
            except Exception as stream_err:
                logger.warning("流式生成中断: %s", stream_err)
                # 保留已生成的部分，只补一句错误提示；避免把已有内容整体丢弃
                partial = buf.strip()
                tail = rag_cfg.get("fallback_answer", FALLBACK_NO_HIT_ANSWER)
                notice = f"\n\n（生成中断：{stream_err}）" if partial else tail
                buf += notice
                for piece in notice:
                    yield piece
            finally:
                # 收尾时构造完整 result
                pass

        return _stream()  # UI 中只需迭代字符串片段即可

    # ------------------------------------------------------------------
    def answer_with_timing(
        self,
        question: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None,
        allowed_groups: Optional[List[str]] = None,
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
        retrieval_cfg = self.config.get("retrieval", {})
        rag_cfg = self.config.get("rag", {})
        history = history or []

        # 意图路由：寒暄/身份类直接模板回应（不加载 LLM，延迟≈0）
        decision = classify_query(question)
        if decision.intent == "chitchat":
            reply = self._render_chitchat(decision.reply or "")
            return RAGResult(
                answer=reply,
                sources=[],
                raw_hits=[],
                timings={"llm_total_ms": 0.0, "ttft_ms": 0.0, "tokens_generated": 0, "tokens_per_sec": None},
            )

        self.ensure_llm()
        k = top_k or retrieval_cfg.get("top_k", 5)
        if self.reranker is not None:
            initial_k = max(k, retrieval_cfg.get("rerank_candidates", k * 3))
        else:
            initial_k = k

        search_query = (
            rewrite_query_for_retrieval(question, history)
            if self.context_assembler.enabled and history
            else question
        )
        hits = self._retrieve(search_query, top_k=initial_k, where=where, allowed_groups=allowed_groups)
        if self.reranker is not None:
            hits = self._rerank(question, hits, top_n=k)

        if not hits:
            fallback = rag_cfg.get(
                "fallback_answer",
                FALLBACK_NO_HIT_ANSWER,
            )
            result = RAGResult(
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
            result.verification = self._verify(fallback, [])
            return result

        # 聚合统计类问题（条目数等）：结构化元数据直接作答，不依赖小模型数数
        structured = self._structured_count_answer(question, hits)
        if structured is not None:
            return structured

        max_ctx = rag_cfg.get("max_context_tokens", 2048)
        context_chunks, _ctx_str = self._build_context(hits, max_chars=max_ctx * 2)
        context_chunks = self._augment_with_graph(question, context_chunks)
        messages = self.context_assembler.build(
            question,
            context_chunks,
            history=history,
            long_term=self._recall_long_term(user_id),
            runtime=self._runtime_context(hits),
        )
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

        result = RAGResult(
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
        result.verification = self._verify(full_text, context_chunks)
        return result

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
    def stream_answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None,
        allowed_groups: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """流式问答的高级封装：返回 ``{"event":..., "data":...}`` 事件流。

        事件类型：
        - ``hits``: 命中来源（最早发出）
        - ``token``: 生成 token
        - ``done``: 结束（data 含 cites 与 verification 可验证性报告）
        - ``error``: 异常
        """
        retrieval_cfg = self.config.get("retrieval", {})
        k = top_k or retrieval_cfg.get("top_k", 5)
        history = history or []

        # 意图路由：寒暄/身份类直接模板回应（不加载 LLM）
        decision = classify_query(question)
        if decision.intent == "chitchat":
            reply = self._render_chitchat(decision.reply or "")
            yield {"event": "hits", "data": []}
            yield {"event": "token", "data": reply}
            yield {"event": "done", "data": [], "verification": None}
            return

        self.ensure_llm()
        search_query = (
            rewrite_query_for_retrieval(question, history)
            if self.context_assembler.enabled and history
            else question
        )
        hits = self._retrieve(search_query, top_k=k, where=None, allowed_groups=allowed_groups)
        if self.reranker is not None:
            hits = self._rerank(question, hits, top_n=k)

        # 聚合统计类问题（条目数等）：结构化元数据直接作答，不依赖小模型数数
        structured = self._structured_count_answer(question, hits)
        if structured is not None:
            yield {"event": "hits", "data": structured.sources}
            yield {"event": "token", "data": structured.answer}
            yield {"event": "done", "data": structured.sources, "verification": structured.verification}
            return

        # 来源事件
        max_ctx = self.config.get("rag", {}).get("max_context_tokens", 2048)
        context_chunks, _ = self._build_context(hits, max_chars=max_ctx * 2)
        context_chunks = self._augment_with_graph(question, context_chunks)
        cites = self._format_citations(context_chunks)

        yield {"event": "hits", "data": cites}

        if not hits:
            fallback = self.config.get("rag", {}).get(
                "fallback_answer",
                FALLBACK_NO_HIT_ANSWER,
            )
            yield {"event": "token", "data": fallback}
            yield {"event": "done", "data": cites, "verification": self._verify(fallback, [])}
            return

        messages = self.context_assembler.build(
            question,
            context_chunks,
            history=history,
            long_term=self._recall_long_term(user_id),
            runtime=self._runtime_context(hits),
        )
        buf = ""
        try:
            for piece in self.llm.chat(messages, self.llm.gen_cfg, stream=True):
                buf += piece
                yield {"event": "token", "data": piece}
        except Exception as exc:
            yield {"event": "error", "data": str(exc)}
            return
        yield {"event": "done", "data": cites, "verification": self._verify(buf.strip(), context_chunks)}

    async def astream_answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None,
        allowed_groups: Optional[List[str]] = None,
    ):
        """异步流式问答：yield ``{"event":..., "data":...}`` 事件。

        事件类型：
        - ``hits``: 命中来源（最早发出）
        - ``token``: 生成 token
        - ``done``: 结束（data 含 cites 与 verification）
        - ``error``: 异常

        实现说明：检索 + LLM 推理是同步阻塞的，因此把 ``stream_answer``
        放到独立线程中生产事件，再通过 ``asyncio.Queue`` 转交事件循环。
        之前直接 ``run_in_executor(None, self.stream_answer, ...)`` 只会立即
        返回生成器对象（生成器函数被调用时并不执行函数体），随后的同步
        ``for`` 迭代——包括整个 LLM 推理——仍然跑在事件循环线程上，会把
        整个 FastAPI 事件循环卡住。
        """
        import asyncio

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        _SENTINEL = object()

        def _produce() -> None:
            try:
                for event in self.stream_answer(
                    question,
                    top_k=top_k,
                    history=history,
                    user_id=user_id,
                    allowed_groups=allowed_groups,
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
            except Exception as exc:  # noqa: BLE001
                asyncio.run_coroutine_threadsafe(
                    queue.put({"event": "error", "data": str(exc)}), loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop).result()

        import threading

        producer = threading.Thread(target=_produce, daemon=True)
        producer.start()

        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            yield item