#!/usr/bin/env python
"""一键导入示例文档：写入示例员工手册并入库到默认知识库。

用法：
    python scripts/import_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_MD = """# 员工手册（示例）

## 一、请假制度

员工请假需提前至少三个工作日向直属主管提出书面申请。
连续请假超过五个工作日的，需同时抄送人力资源部备案。

## 二、差旅报销

员工出差应提前在 OA 系统提交出差申请。
报销需提供交通票据、住宿发票及工作日报；单笔超过 5000 元需共同审批。

## 三、信息安全

员工不得将内部系统账号借给他人使用，离开工位时应锁定电脑。
"""


def main() -> int:
    from src.utils import apply_env_overrides, load_config
    from src.document_meta import enrich_file
    from src.data_quality import run_data_pipeline
    from api.deps import get_splitter

    target = ROOT / "data" / "raw" / "示例-员工手册.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(SAMPLE_MD, encoding="utf-8")
    print("示例文档：", target)

    cfg = apply_env_overrides(load_config(str(ROOT / "config" / "config.yaml")))
    splitter = get_splitter(cfg)
    docs = enrich_file(target, default_acl=(cfg.get("data_quality", {}) or {}).get("default_acl", "*"))
    chunks, report = run_data_pipeline(docs, cfg, splitter)
    print("解析得到", len(docs), "节、", len(chunks), "块")

    from src.db.database import SessionLocal
    from src.middleware.identity import get_local_user
    from src.kb_service import ensure_default_kb
    from src.factories import TenantAwareFactory

    db = SessionLocal()
    try:
        user = get_local_user(db)
        kb = ensure_default_kb(db, user.id, cfg)
        store = TenantAwareFactory.get_chroma_store_by_name(kb.collection_name, cfg)
        sources = sorted({str(c.metadata.get("source") or "示例-员工手册.md") for c in chunks})
        n = store.replace_chunks_for_sources(chunks, sources)
        print("已入库", n, "块到知识库「" + kb.name + "」(" + kb.collection_name + ")")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
