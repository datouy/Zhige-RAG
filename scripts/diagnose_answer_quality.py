"""端到端回答质量诊断：修复前 vs 修复后。

为什么需要这个脚本
------------------
"总结类问题只能给出内容目录 + 每句挂引用 + 免责结尾"这件事，根因横跨四层：

1. 上下文预算（rag.max_context_tokens）—— 太小则模型只看到开头两段；
2. 全文扩展（rag.fulltext_expand）—— 只有 answer() 有，Web 走的 stream 没有；
3. Prompt（config/prompts.yaml）—— 命令"照抄原文"，禁止提炼；
4. 校验（rag.verification）—— 把"提炼"判成"编造"。

本脚本把 1/2 两层的影响做成可对比的数字，并把 3 层的实际 system prompt
打印出来，用来确认改动真的落到了送进模型的那条消息上。

用法
----
    # 只看上下文与 prompt（不调用 LLM，几秒出结果）
    python scripts/diagnose_answer_quality.py

    # 连真实模型跑完整回答
    python scripts/diagnose_answer_quality.py --ask

    # 换问题
    python scripts/diagnose_answer_quality.py "这一章真正想表达什么" --ask
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.utils import apply_env_overrides, load_config  # noqa: E402

DEFAULT_QUESTION = "总结一下这一章的主要内容，以及它真正想表达什么"


def _resolve_collection(explicit: str) -> str:
    """找到"有数据"的那个 collection。

    ``RAGPipeline.from_config`` 用的是裸 collection 名（``chinese_rag_kb``），
    而 Web 端按用户 / 知识库隔离后是 ``{user_prefix}_{base}``。诊断要打在
    真实数据上，否则只会得到"召回 0 块"的假结论。
    """
    if explicit:
        return explicit
    import os
    import sqlite3

    db_path = os.getenv("DB_PATH") or "data/users.db"
    if not os.path.exists(db_path):
        return ""
    try:
        con = sqlite3.connect(db_path)
        row = con.execute(
            "SELECT collection_name FROM knowledge_bases WHERE is_default = 1 LIMIT 1"
        ).fetchone()
        con.close()
        return str(row[0]) if row else ""
    except Exception:
        return ""



def _probe(
    pipeline: RAGPipeline,
    question: str,
    *,
    max_context_tokens: int,
    fulltext_expand: bool,
    top_k: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """在给定配置下组装一次上下文，返回统计信息与分块列表（不调用 LLM）。"""
    saved = pipeline.config
    cfg = copy.deepcopy(saved)
    cfg.setdefault("rag", {})["max_context_tokens"] = max_context_tokens
    cfg["rag"].setdefault("fulltext_expand", {})["enabled"] = fulltext_expand
    pipeline.config = cfg
    try:
        hits = pipeline.retrieve(question, top_k=top_k)
        recalled = len(hits)
        hits = pipeline._expand_to_fulltext(question, hits)
        expanded = len(hits)
        chunks, _ = pipeline._build_context(hits, max_chars=max_context_tokens * 2)
    finally:
        pipeline.config = saved

    total_chars = sum(len(c["content"]) for c in chunks)
    truncated = sum(1 for c in chunks if c["content"].endswith("…"))
    return {
        "recalled": recalled,
        "after_expand": expanded,
        "blocks_in_prompt": len(chunks),
        "chars_in_prompt": total_chars,
        "truncated_blocks": truncated,
    }, chunks


def _print_row(label: str, stat: Dict[str, Any]) -> None:
    print(
        f"  {label:<26} 召回 {stat['recalled']:>3} 块 → 扩展后 {stat['after_expand']:>3} 块"
        f" → 进 prompt {stat['blocks_in_prompt']:>3} 块 / {stat['chars_in_prompt']:>6} 字"
        f"（被截断 {stat['truncated_blocks']} 块）"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="诊断 RAG 回答质量")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ask", action="store_true", help="额外调用真实模型跑完整回答")
    parser.add_argument("--show-context", action="store_true", help="打印送进 prompt 的上下文全文")
    parser.add_argument(
        "--collection",
        default="",
        help="要诊断的 collection 名；默认取 users.db 里默认知识库的真实 collection",
    )
    args = parser.parse_args()

    cfg = apply_env_overrides(load_config(args.config))
    pipeline = RAGPipeline.from_config(args.config, overrides=cfg)

    collection = _resolve_collection(args.collection)
    if collection:
        from src.factories import TenantAwareFactory

        pipeline.vector_store = TenantAwareFactory.get_chroma_store_by_name(collection, cfg)
        print(f"\n[collection] {collection}（{pipeline.vector_store.count()} 块）")
    else:
        print("\n[collection] 未找到默认知识库，使用配置里的裸 collection 名")

    print(f"\n问题：{args.question}\n")

    # ---- 1. 上下文预算 / 全文扩展的前后对比 -----------------------------------
    print("[1] 上下文装配（决定模型能看到多少）")
    old, old_chunks = _probe(
        pipeline, args.question, max_context_tokens=1800, fulltext_expand=False, top_k=args.top_k
    )
    _print_row("修复前 1800tok/无扩展", old)
    mid, _ = _probe(
        pipeline, args.question, max_context_tokens=1800, fulltext_expand=True, top_k=args.top_k
    )
    _print_row("只加全文扩展 1800tok", mid)
    new, new_chunks = _probe(
        pipeline, args.question, max_context_tokens=12000, fulltext_expand=True, top_k=args.top_k
    )
    _print_row("修复后 12000tok+扩展", new)

    growth = (new["chars_in_prompt"] / old["chars_in_prompt"]) if old["chars_in_prompt"] else 0
    print(f"  ⇒ 进 prompt 的字数：{old['chars_in_prompt']} → {new['chars_in_prompt']}（×{growth:.1f}）")

    # ---- 2. 实际送到模型面前的 system prompt --------------------------------
    print("\n[2] 送进模型的 system prompt（前 420 字）")
    messages = pipeline.context_assembler.build(
        args.question,
        new_chunks,
        history=[],
        long_term=[],
        runtime=pipeline._runtime_context([]),
    )
    system_text = messages[0]["content"]
    print("  " + system_text[:420].replace("\n", "\n  "))
    banned = ["优先直接采用上下文中的原文表述", "逐条照抄", "宁可拒答"]
    still_there = [b for b in banned if b in system_text]
    print(
        "\n  ⇒ 旧版『照抄原文』类约束："
        + ("仍存在 -> " + "；".join(still_there) if still_there else "已移除")
    )

    # ---- 3. 上下文覆盖度 ---------------------------------------------------
    if args.show_context or new["blocks_in_prompt"] > old["blocks_in_prompt"]:
        print("\n[3] 修复后上下文覆盖（每块首 40 字）")
        for chunk in new_chunks[:60]:
            head = chunk["content"][:40].replace("\n", " ")
            print(f"  [{chunk['index']:>2}] {chunk['title'][:32]:<32} {head}…")
        if len(new_chunks) > 60:
            print(f"  … 其余 {len(new_chunks) - 60} 块略")

    # ---- 4. 真实模型回答 ---------------------------------------------------
    if args.ask:
        # 前面的 _probe 反复替换过 pipeline.config，为避免任何残留状态影响
        # 生成路径，这里重建一个干净的 pipeline 再问。
        pipeline = RAGPipeline.from_config(args.config, overrides=cfg)
        if collection:
            from src.factories import TenantAwareFactory

            pipeline.vector_store = TenantAwareFactory.get_chroma_store_by_name(collection, cfg)
        print("\n[4] 真实回答（走 Web UI 同一条 astream/stream 路径）")
        generated: List[str] = []
        verification: Any = None
        for event in pipeline.stream_answer(args.question, top_k=args.top_k):
            kind = event.get("event")
            if kind == "token":
                generated.append(event.get("data") or "")
            elif kind == "done":
                verification = event.get("verification")
            elif kind == "error":
                print(f"  [错误] {event.get('data')}")
                return 1
        answer = "".join(generated).strip()
        print("\n" + answer)
        if verification:
            print(
                f"\n  verification: status={verification.get('status')} "
                f"cited_ratio={verification.get('cited_sentence_ratio')} "
                f"unsupported_facts={len(verification.get('unsupported_facts') or [])} "
                f"ungrounded={len(verification.get('unsupported_sentences') or [])}"
            )
        # 简单体检：是否又退化成"条目清单 + 免责声明"
        list_lines = sum(1 for line in answer.splitlines() if line.strip()[:2] in {"1.", "2.", "3.", "4.", "1、", "2、", "3、", "- "})
        has_disclaimer = any(k in answer for k in ("需说明", "无法进一步确认", "以上仅为", "仅供参考"))
        print(
            f"\n  ⇒ 形态体检：列表行 {list_lines} 行；"
            f"{'仍带免责收尾' if has_disclaimer else '无免责收尾'}"
        )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
