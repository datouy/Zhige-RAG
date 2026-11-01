from sqlalchemy import create_engine, inspect, text

import src.db.database as database_module


def test_old_document_record_unique_key_is_migrated(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE document_records (
                id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(36) NOT NULL,
                kb_id VARCHAR(36),
                source VARCHAR(512) NOT NULL,
                title VARCHAR(512),
                doc_status VARCHAR(20) NOT NULL,
                acl VARCHAR(512) NOT NULL,
                department VARCHAR(100),
                content_hash VARCHAR(64),
                chunk_count INTEGER,
                file_ext VARCHAR(16),
                updated_at_source DATETIME,
                created_at DATETIME,
                updated_at DATETIME,
                CONSTRAINT uq_docrecord_user_source UNIQUE (user_id, source)
            )
            """
        )
        conn.execute(
            text(
                "INSERT INTO document_records "
                "(id, user_id, kb_id, source, doc_status, acl) "
                "VALUES ('1', 'u', 'kb', 'a.md', 'active', '*')"
            )
        )
    monkeypatch.setattr(database_module, "engine", engine)
    database_module._migrate_document_record_key()

    unique_columns = {
        tuple(item["column_names"])
        for item in inspect(engine).get_unique_constraints("document_records")
    }
    assert ("user_id", "kb_id", "source") in unique_columns
    assert ("user_id", "source") not in unique_columns
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT source FROM document_records").scalar() == "a.md"
