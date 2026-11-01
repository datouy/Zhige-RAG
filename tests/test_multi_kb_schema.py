from dataclasses import dataclass
from uuid import uuid4

from api.routes.chat import _register_documents
from src.db.database import SessionLocal, init_db
from src.db.models import DocumentRecord, KnowledgeBase


@dataclass
class _Chunk:
    text: str
    metadata: dict


def test_same_source_allowed_in_two_knowledge_bases():
    init_db()
    user_id = str(uuid4())
    db = SessionLocal()
    try:
        first = KnowledgeBase(
            user_id=user_id,
            name="first",
            collection_name=f"{user_id}_first",
        )
        second = KnowledgeBase(
            user_id=user_id,
            name="second",
            collection_name=f"{user_id}_second",
        )
        db.add_all([first, second])
        db.flush()
        db.add_all(
            [
                DocumentRecord(
                    user_id=user_id,
                    kb_id=first.id,
                    source="same.md",
                    chunk_count=1,
                ),
                DocumentRecord(
                    user_id=user_id,
                    kb_id=second.id,
                    source="same.md",
                    chunk_count=1,
                ),
            ]
        )
        db.commit()
        assert (
            db.query(DocumentRecord)
            .filter(DocumentRecord.user_id == user_id, DocumentRecord.source == "same.md")
            .count()
            == 2
        )
    finally:
        db.close()


def test_register_documents_groups_multiple_sources():
    init_db()
    user_id = str(uuid4())
    kb_id = str(uuid4())
    chunks = [
        _Chunk("a1", {"source": "a.md", "kb_id": kb_id, "title": "A"}),
        _Chunk("a2", {"source": "a.md", "kb_id": kb_id, "title": "A"}),
        _Chunk("b1", {"source": "b.md", "kb_id": kb_id, "title": "B"}),
    ]
    db = SessionLocal()
    try:
        _register_documents(db, user_id, chunks, {})
        db.commit()
        rows = {
            row.source: row
            for row in db.query(DocumentRecord)
            .filter(DocumentRecord.user_id == user_id)
            .all()
        }
        assert rows["a.md"].chunk_count == 2
        assert rows["b.md"].chunk_count == 1
        assert rows["a.md"].content_hash != rows["b.md"].content_hash
    finally:
        db.close()
