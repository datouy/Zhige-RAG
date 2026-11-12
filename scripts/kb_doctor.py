"""知识库台账体检：把"界面上看到的"与"检索真正搜到的"对齐。

为什么需要它
------------
``document_records`` 是入库文档的权威台账，界面上的"N 文档 / M 分块"和文档
列表都读它。但**引入台账之前**入库的向量不会补登记（老库升级后只剩一堆没有
台账的分块），于是出现这种自相矛盾的状况：

    界面：默认知识库 1 文档 / 156 分块
    实际：同一个 collection 里躺着 9 份文档、717 个分块

检索搜的是 collection，所以那些"看不见"的分块照样会参与召回 —— 用户以为
库里只有自己的论文，回答里却冒出 Spring 面试题、别的实习报告。本脚本把这层
差异摊开，并提供显式修复动作（默认只读，不做任何写入）。

用法::

    python scripts/kb_doctor.py                 # 只体检（默认）
    python scripts/kb_doctor.py --apply         # 把未登记的分块补登记进台账
    python scripts/kb_doctor.py --delete "旧文档.docx"   # 彻底删除某个来源

``--delete`` 会从所有知识库的 collection 里删除该来源的全部分块，并清掉台账
记录（不可撤销，删前请先跑一次不带参数的体检）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _open_store(collection_name: str, cfg: dict):
    """打开 collection。刻意不传 embedding：体检/删除都不需要编码，
    避免为了看一眼计数就把向量模型加载起来。"""
    from src.vector_store import ChromaStore

    return ChromaStore(
        persist_directory=cfg["vector_store"]["persist_directory"],
        collection_name=collection_name,
        embedding_model=None,
        distance_fn=cfg["vector_store"].get("distance_fn", "cosine"),
    )


def _survey(db, cfg) -> list[dict]:
    """逐个知识库比对 collection 实际内容与台账。"""
    from src.db.models import DocumentRecord, KnowledgeBase
    from src.kb_service import collection_name_for

    out: list[dict] = []
    for kb in db.query(KnowledgeBase).order_by(KnowledgeBase.created_at.asc()).all():
        name = kb.collection_name or collection_name_for(
            kb.user_id,
            kb.id,
            cfg["vector_store"].get("collection_name", "chinese_rag_kb"),
            cfg.get("embedding", {}),
            is_default=bool(kb.is_default),
        )
        try:
            store = _open_store(name, cfg)
            sources = store.list_sources()
        except Exception as exc:  # noqa: BLE001
            out.append({"kb": kb, "collection": name, "error": str(exc), "sources": []})
            continue

        records = {
            str(r.source): r
            for r in db.query(DocumentRecord)
            .filter(DocumentRecord.user_id == kb.user_id, DocumentRecord.kb_id == kb.id)
            .all()
        }
        for row in sources:
            row["registered"] = row["source"] in records
        out.append(
            {
                "kb": kb,
                "collection": name,
                "sources": sources,
                "actual_chunks": sum(r["chunks"] for r in sources),
                "registered_chunks": sum(
                    (r.chunk_count or 0) for r in records.values()
                ),
                "missing_records": [s for s in records if s not in {r["source"] for r in sources}],
            }
        )
    return out


def _report(surveys: list[dict]) -> None:
    print("=" * 78)
    print("知识库台账体检")
    print("=" * 78)
    for s in surveys:
        kb = s["kb"]
        print(f"\n知识库：{kb.name}（{kb.id}）")
        print(f"  collection：{s['collection']}")
        if s.get("error"):
            print(f"  [跳过] 打不开 collection：{s['error']}")
            continue
        print(f"  台账：{kb.doc_count} 文档 / {kb.chunk_count} 分块（界面显示的就是这两个数）")
        print(f"  实际：{len(s['sources'])} 个来源 / {s['actual_chunks']} 分块（检索真正搜到的范围）")
        orphans = [r for r in s["sources"] if not r["registered"]]
        if orphans:
            print(f"  [!] 未登记来源 {len(orphans)} 个 —— 界面上看不到，但检索会命中：")
            for r in sorted(orphans, key=lambda x: -x["chunks"]):
                print(f"      {r['chunks']:6d} 块  {r['source']}")
        for src in s["missing_records"]:
            print(f"  [!] 台账有记录但库里没有分块：{src}")
    print("\n" + "=" * 78)


def _apply(db, surveys: list[dict]) -> int:
    """把未登记的分块补登记进 document_records，并刷新统计。"""
    from src.db.models import DocumentRecord
    from src.kb_service import refresh_kb_stats

    added = 0
    for s in surveys:
        kb = s["kb"]
        for row in s["sources"]:
            if row["registered"]:
                continue
            src = row["source"]
            db.add(
                DocumentRecord(
                    user_id=kb.user_id,
                    kb_id=kb.id,
                    source=src,
                    title=Path(src).stem,
                    doc_status="active",
                    acl="*",
                    chunk_count=int(row["chunks"]),
                    file_ext=Path(src).suffix,
                )
            )
            added += 1
        db.commit()
        refresh_kb_stats(db, kb.user_id, kb.id)
        print(f"  已补登记并刷新统计：{kb.name}")
    return added


def _delete(db, cfg, source: str) -> int:
    """从所有 collection 删除某来源的全部分块，并清掉台账记录。"""
    from src.db.models import DocumentRecord, KnowledgeBase
    from src.kb_service import refresh_kb_stats

    total = 0
    for kb in db.query(KnowledgeBase).all():
        name = kb.collection_name
        if not name:
            continue
        try:
            store = _open_store(name, cfg)
            if store.count() == 0:
                continue
            n = store.delete_by_metadata({"source": {"$eq": source}})
        except Exception as exc:  # noqa: BLE001
            print(f"  [跳过] {name}：{exc}")
            continue
        if n:
            print(f"  已从 {name} 删除 {n} 块")
            total += n
    removed = (
        db.query(DocumentRecord)
        .filter(DocumentRecord.source == source)
        .delete(synchronize_session=False)
    )
    db.commit()
    if removed:
        print(f"  已清理台账记录 {removed} 条")
    for kb in db.query(KnowledgeBase).all():
        refresh_kb_stats(db, kb.user_id, kb.id)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库台账体检 / 修复")
    parser.add_argument("--apply", action="store_true", help="补登记未登记的分块（写入）")
    parser.add_argument("--delete", metavar="SOURCE", help="彻底删除某个来源（不可撤销）")
    args = parser.parse_args()

    from src.db.database import SessionLocal
    from src.settings_service import effective_config

    cfg = effective_config()
    db = SessionLocal()
    try:
        if args.delete:
            print(f"删除来源：{args.delete}")
            n = _delete(db, cfg, args.delete)
            print(f"完成：共删除 {n} 块")
            return 0

        surveys = _survey(db, cfg)
        _report(surveys)
        if args.apply:
            print("\n开始补登记 ...")
            added = _apply(db, surveys)
            print(f"完成：补登记 {added} 个来源。再次运行本脚本可确认已对齐。")
        else:
            print("（只读体检。补登记用 --apply；删除某个来源用 --delete \"文件名\"）")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
