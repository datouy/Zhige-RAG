from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from src.vector_store import (
    ChromaStore,
    _is_transient_chroma_error,
    _query_chroma_with_retry,
)


class _Embedding:
    dim = 4

    def encode(self, texts, batch_size=8, **kwargs):
        rows = []
        for text in texts:
            raw = sum(ord(ch) for ch in str(text))
            rows.append(
                [
                    1.0,
                    float(raw % 17 + 1),
                    float(len(str(text)) + 1),
                    float(sum(str(text).encode("utf-8")) % 13 + 1),
                ]
            )
        return np.asarray(rows, dtype=np.float32)


@dataclass
class _Chunk:
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _store(tmp_path, name="write_lock"):
    return ChromaStore(
        persist_directory=str(tmp_path),
        collection_name=name,
        embedding_model=_Embedding(),
        hybrid_enabled=True,
    )


def test_revision_refreshes_count_bm25_and_reset_across_instances(tmp_path):
    writer = _store(tmp_path)
    reader = _store(tmp_path)
    assert reader.count() == 0

    writer.add_chunks(
        [_Chunk("旧版报销上限 500 元 [1]", 0, {"source": "policy.md", "acl": "*"})]
    )
    assert reader.count() == 1
    assert "500" in reader._ensure_bm25().texts[-1]

    writer.replace_chunks_for_sources(
        [_Chunk("新版报销上限 800 元 [1]", 0, {"source": "policy.md", "acl": "*"})],
        ["policy.md"],
    )
    # 数量相同也必须依赖内容版本重建，不能只看 N。
    assert reader.count() == 1
    assert "800" in reader._ensure_bm25().texts[-1]

    writer.reset()
    assert reader.count() == 0


def test_concurrent_writers_do_not_lose_chunks(tmp_path):
    stores = [_store(tmp_path, "concurrent") for _ in range(2)]

    def write(pair):
        store_index, item_index = pair
        stores[store_index].add_chunks(
            [
                _Chunk(
                    f"文档 {store_index}-{item_index}",
                    item_index,
                    {"source": f"writer-{store_index}.md", "acl": "*"},
                )
            ]
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, [(i % 2, i // 2) for i in range(20)]))

    assert stores[0].count() == 20
    assert stores[1].content_revision == stores[0].content_revision


class _FlakyCollection:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def query(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {"ids": [self.calls]}


@pytest.mark.parametrize(
    "message",
    [
        "Error executing plan: Internal error: Error finding id",
        "database is locked",
    ],
)
def test_transient_chroma_error_is_retried(message):
    collection = _FlakyCollection([RuntimeError(message)])

    result = _query_chroma_with_retry(collection, {"query_embeddings": [[]]})

    assert result == {"ids": [2]}
    assert collection.calls == 2


def test_transient_retry_reopens_replacement_collection():
    stale = _FlakyCollection([RuntimeError("Error finding id")])
    fresh = _FlakyCollection([])

    result = _query_chroma_with_retry(
        stale,
        {"query_embeddings": [[]]},
        on_retry=lambda: fresh,
    )

    assert result == {"ids": [1]}
    assert stale.calls == 1
    assert fresh.calls == 1


@pytest.mark.parametrize(
    "message",
    [
        "invalid where filter",
        "embedding dimension mismatch",
    ],
)
def test_deterministic_chroma_error_is_not_retried(message):
    collection = _FlakyCollection([RuntimeError(message), RuntimeError(message)])

    with pytest.raises(RuntimeError, match=message):
        _query_chroma_with_retry(collection, {"query_embeddings": [[]]})

    assert collection.calls == 1


def test_transient_error_classification_is_narrow():
    assert _is_transient_chroma_error(RuntimeError("segment not found"))
    assert not _is_transient_chroma_error(RuntimeError("permission denied"))


def test_status_metadata_probe_distinguishes_legacy_and_modern_index(tmp_path):
    store = _store(tmp_path, "status_metadata")
    store.add_chunks(
        [_Chunk("旧索引", 0, {"source": "legacy.md", "acl": "*"})]
    )
    assert store.has_doc_status_metadata() is False

    store.add_chunks(
        [
            _Chunk(
                "新版草稿",
                0,
                {"source": "draft.md", "acl": "*", "doc_status": "draft"},
            )
        ]
    )
    assert store.has_doc_status_metadata() is True
