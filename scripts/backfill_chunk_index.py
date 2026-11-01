"""给已有索引补上 `chunk_index`（分块在原文中的顺序）。

为什么需要它
------------
检索结果默认按**相似度**排序，不是按原文顺序。把它直接当上下文，
"先说结论、后说原因"的叙述会被打乱，模型只能平行罗列，做不出因果推理。

`_build_context` 只有在**所有分块都带 `chunk_index`** 时才会按原文顺序重排
（没有就保持原样，不猜顺序）。新入库的文档会自动写入该字段；老索引需要
本脚本回填一次 —— 回填只需改 metadata，不重新计算向量，几秒完成。

顺序怎么来
----------
拿 `metadata.filepath` 指向的原始文件做基准：把分块正文（剥掉面包屑前缀）
归一化后，在原文里找它的起始位置；按位置升序编号。定位失败的分块保持
原有相对顺序排在被定位分块之后，并在报告里单独列出。

用法
----
    # 先看能定位多少，不写入
    python scripts/backfill_chunk_index.py --dry-run

    # 指定 collection（默认取 users.db 里默认知识库的 collection）
    python scripts/backfill_chunk_index.py --collection u7ac1b8d7010bb6cd_chinese_rag_kb

    # 真正写入
    python scripts/backfill_chunk_index.py --write
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import apply_env_overrides, load_config  # noqa: E402

_HEADER_CHARS = "】"


def _normalize(text: str) -> str:
    """去掉所有空白，便于跨清洗差异做子串定位。"""
    return "".join(ch for ch in (text or "") if not ch.isspace())


def _strip_breadcrumb(text: str) -> str:
    """剥掉分块前的 `【章节路径】` 面包屑（原文里没有这段）。"""
    body = (text or "").lstrip()
    if body.startswith("【"):
        end = body.find(_HEADER_CHARS)
        if 0 < end <= 160:
            return body[end + 1 :].lstrip()
    return body


def _resolve_collection(explicit: str) -> str:
    if explicit:
        return explicit
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


def _locate(norm_source: str, body: str) -> Optional[int]:
    """在归一化原文里定位分块起点；多级降级，失败返回 None。"""
    norm_body = _normalize(body)
    if not norm_body or not norm_source:
        return None
    for size in (48, 32, 20, 12):
        key = norm_body[:size]
        if len(key) < 8:
            break
        pos = norm_source.find(key)
        if pos >= 0:
            return pos
    return None


def _read_source_text(path: Path) -> Optional[str]:
    """读原文纯文本。docx 用 python-docx 抽段落，其它按文本读。"""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        try:
            from docx import Document  # type: ignore

            doc = Document(str(path))
            parts = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    parts.extend(cell.text for cell in row.cells)
            return "\n".join(parts)
        except Exception as exc:  # noqa: BLE001
            print(f"  docx 解析失败（{path.name}）：{exc}")
            return None
    if suffix in {".pdf"}:
        # PDF 的文本层与分块清洗结果差异过大，子串定位不可靠，直接跳过
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="回填 chunk_index")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--collection", default="")
    parser.add_argument("--write", action="store_true", help="真正写入（默认只报告）")
    parser.add_argument("--dry-run", action="store_true", help="只报告，不写入")
    args = parser.parse_args()

    write = bool(args.write) and not args.dry_run

    cfg = apply_env_overrides(load_config(args.config))
    name = _resolve_collection(args.collection)
    if not name:
        print("找不到要处理的 collection，请用 --collection 指定")
        return 1

    from src.factories import TenantAwareFactory

    store = TenantAwareFactory.get_chroma_store_by_name(name, cfg)
    print(f"collection = {name}（{store.count()} 块）")

    data = store.collection.get(include=["documents", "metadatas"])
    ids: List[str] = list(data.get("ids") or [])
    docs: List[str] = list(data.get("documents") or [])
    metas: List[Dict[str, Any]] = list(data.get("metadatas") or [])

    by_source: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
    for cid, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        source = str(meta.get("source") or "")
        by_source.setdefault(source, []).append((cid, doc or "", meta))

    total_updated = 0
    for source, items in sorted(by_source.items()):
        filepath = ""
        for _cid, _doc, meta in items:
            filepath = str(meta.get("filepath") or "")
            if filepath:
                break
        raw_path = Path(filepath) if filepath else None
        if raw_path is None or not raw_path.exists():
            print(f"\n[{source}] 找不到原文（filepath={filepath!r}），跳过")
            continue
        try:
            text = _read_source_text(raw_path)
        except OSError as exc:
            print(f"\n[{source}] 读原文失败：{exc}，跳过")
            continue
        if not text:
            print(f"\n[{source}] 无法抽取原文文本，跳过（该文档分块保持原顺序）")
            continue
        norm_source = _normalize(text)

        located: List[Tuple[int, str, Dict[str, Any]]] = []
        unlocated: List[Tuple[str, Dict[str, Any]]] = []
        for cid, doc, meta in items:
            pos = _locate(norm_source, _strip_breadcrumb(doc))
            if pos is None:
                unlocated.append((cid, meta))
            else:
                located.append((pos, cid, meta))
        located.sort(key=lambda item: item[0])

        update_ids: List[str] = []
        update_metas: List[Dict[str, Any]] = []
        for index, (_pos, cid, meta) in enumerate(located):
            if meta.get("chunk_index") != index:
                new_meta = dict(meta)
                new_meta["chunk_index"] = index
                update_ids.append(cid)
                update_metas.append(new_meta)
        for offset, (cid, meta) in enumerate(unlocated, start=len(located)):
            if meta.get("chunk_index") != offset:
                new_meta = dict(meta)
                new_meta["chunk_index"] = offset
                update_ids.append(cid)
                update_metas.append(new_meta)

        print(
            f"\n[{source[:36]}] 共 {len(items)} 块：定位成功 {len(located)}，"
            f"失败 {len(unlocated)}，待写 {len(update_ids)}"
        )
        if update_ids and write:
            store.collection.update(ids=update_ids, metadatas=update_metas)
            total_updated += len(update_ids)

    if write:
        print(f"\n完成：写入 {total_updated} 条 chunk_index")
        print("提示：向量与文本都没动，无需重新入库。")
    else:
        print("\n以上为 dry-run，未写入。加 --write 落库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
