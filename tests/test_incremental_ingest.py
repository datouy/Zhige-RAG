"""入库 content_hash 增量比较回归测试。"""
from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass
class _Chunk:
    text: str
    index: int = 0
    metadata: dict | None = None


def _chunk(source: str, text: str, kb_id: str) -> _Chunk:
    return _Chunk(text=text, metadata={"source": source, "kb_id": kb_id})


def test_partition_changed_sources_skips_same_hash():
    from api.routes import chat as chat_module
    from src.db.database import SessionLocal, init_db
    from src.db.models import DocumentRecord

    init_db()
    user_id = f"user-{uuid.uuid4().hex}"
    kb_id = f"kb-{uuid.uuid4().hex}"
    source = f"guide-{uuid.uuid4().hex}.md"
    original = _chunk(source, "第一段\n第二段", kb_id)
    db = SessionLocal()
    try:
        db.add(
            DocumentRecord(
                user_id=user_id,
                kb_id=kb_id,
                source=source,
                content_hash=chat_module._source_content_hash([original]),
            )
        )
        db.commit()

        changed, sources, summary = chat_module._partition_changed_sources(
            user_id, kb_id, [_chunk(source, "第一段\n第二段", kb_id)]
        )
        assert changed == []
        assert sources == []
        assert summary == {"scanned": 1, "changed": 0, "unchanged": 1}

        changed, sources, summary = chat_module._partition_changed_sources(
            user_id, kb_id, [_chunk(source, "第一段\n已更新", kb_id)]
        )
        assert [chunk.text for chunk in changed] == ["第一段\n已更新"]
        assert sources == [source]
        assert summary == {"scanned": 1, "changed": 1, "unchanged": 0}
    finally:
        db.query(DocumentRecord).filter(DocumentRecord.user_id == user_id).delete()
        db.commit()
        db.close()


def test_replace_changed_sources_writes_only_changed(monkeypatch):
    from api.routes import chat as chat_module
    from src.db.database import SessionLocal, init_db
    from src.db.models import DocumentRecord

    init_db()
    user_id = f"user-{uuid.uuid4().hex}"
    kb_id = f"kb-{uuid.uuid4().hex}"
    same_source = f"same-{uuid.uuid4().hex}.md"
    changed_source = f"changed-{uuid.uuid4().hex}.md"
    same_chunk = _chunk(same_source, "不变", kb_id)
    db = SessionLocal()
    captured = {}

    class _Store:
        def replace_chunks_for_sources(self, chunks, sources, on_commit=None):
            captured["chunks"] = list(chunks)
            captured["sources"] = list(sources)
            if on_commit:
                on_commit()
            return len(chunks)

    try:
        db.add(
            DocumentRecord(
                user_id=user_id,
                kb_id=kb_id,
                source=same_source,
                content_hash=chat_module._source_content_hash([same_chunk]),
            )
        )
        db.commit()
        monkeypatch.setattr(chat_module, "_commit_ingestion_metadata", lambda *args: None)
        report = {}
        inserted, summary = chat_module._replace_changed_sources(
            store=_Store(),
            user_id=user_id,
            kb_id=kb_id,
            chunks=[same_chunk, _chunk(changed_source, "新版", kb_id)],
            report=report,
        )
        assert inserted == 1
        assert captured["sources"] == [changed_source]
        assert [chunk.metadata["source"] for chunk in captured["chunks"]] == [changed_source]
        assert summary == {"scanned": 2, "changed": 1, "unchanged": 1}
        assert report["incremental"] == summary
    finally:
        db.query(DocumentRecord).filter(DocumentRecord.user_id == user_id).delete()
        db.commit()
        db.close()
