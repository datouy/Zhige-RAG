"""文档版本管理模块的单元测试。

测试覆盖：
- test_version_increment: 连续 ingest 同一文档，版本号递增
- test_same_hash_skips_version: 相同 hash 不新建版本
- test_diff_detects_modified_chunks: 正确识别修改的 chunks
- test_diff_detects_added_chunks: 正确识别新增 chunks
- test_diff_detects_removed_chunks: 正确识别删除的 chunks
- test_prune_keeps_latest_n: prune 保留最新的 N 个版本
- test_version_metadata_fields: metadata.json 包含所有必需字段
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.version_manager import (
    compute_diff,
    get_version_detail,
    list_all_documents,
    list_versions,
    prune_versions,
    save_version,
)


# ----------------------------------------------------------------------
# 测试工具
# ----------------------------------------------------------------------
class TestVersionManager(unittest.TestCase):
    """版本管理功能测试套件。"""

    TEST_ROOT: Path | None = None

    @classmethod
    def setUpClass(cls):
        """创建临时版本目录。"""
        cls.TEST_ROOT = Path(tempfile.mkdtemp(prefix="test_versions_"))

    @classmethod
    def tearDownClass(cls):
        """清理临时目录。"""
        if cls.TEST_ROOT and cls.TEST_ROOT.exists():
            shutil.rmtree(cls.TEST_ROOT, ignore_errors=True)

    @contextmanager
    def _mock_versions_root(self):
        """临时替换 VERSIONS_ROOT 为测试目录。"""
        import scripts.version_manager as vm
        old_root = vm.VERSIONS_ROOT
        vm.VERSIONS_ROOT = str(self.TEST_ROOT)
        try:
            yield
        finally:
            vm.VERSIONS_ROOT = old_root

    # ------------------------------------------------------------------
    # 测试用例
    # ------------------------------------------------------------------
    def test_version_increment(self):
        """连续 3 次 ingest 同一文档，版本号 v1→v2→v3。"""
        with self._mock_versions_root():
            doc_name = "test-increment.md"
            chunks_v1 = [{"text": "内容版本1", "metadata": {"source": doc_name}}]
            chunks_v2 = [{"text": "内容版本2", "metadata": {"source": doc_name}}]
            chunks_v3 = [{"text": "内容版本3", "metadata": {"source": doc_name}}]

            v1 = save_version(doc_name, chunks_v1, "hash1", created_by="test")
            self.assertEqual(v1, "v1")

            v2 = save_version(doc_name, chunks_v2, "hash2", created_by="test")
            self.assertEqual(v2, "v2")

            v3 = save_version(doc_name, chunks_v3, "hash3", created_by="test")
            self.assertEqual(v3, "v3")

            # 验证版本列表
            versions = list_versions(doc_name)
            self.assertEqual(len(versions), 3)
            self.assertEqual([v["version"] for v in versions], ["v1", "v2", "v3"])

    def test_same_hash_skips_version(self):
        """doc_hash 相同时不新建版本。"""
        with self._mock_versions_root():
            doc_name = "test-hash.md"
            same_hash = "abc123"
            chunks = [{"text": "相同内容", "metadata": {"source": doc_name}}]

            v1 = save_version(doc_name, chunks, same_hash, created_by="test")
            self.assertEqual(v1, "v1")

            # 再次入库相同 hash
            v1_again = save_version(doc_name, chunks, same_hash, created_by="test")
            self.assertEqual(v1_again, "v1")  # 仍然是 v1，不会新建

            # 只有 1 个版本
            versions = list_versions(doc_name)
            self.assertEqual(len(versions), 1)

            # 不同 hash 会新建
            v2 = save_version(doc_name, chunks, "different_hash", created_by="test")
            self.assertEqual(v2, "v2")
            versions = list_versions(doc_name)
            self.assertEqual(len(versions), 2)

    def test_diff_detects_modified_chunks(self):
        """v1/v2 有 1 个修改 chunk，diff 正确识别。"""
        with self._mock_versions_root():
            doc_name = "test-diff-modify.md"
            chunks_v1 = [
                {"text": "第一段内容不变", "metadata": {"index": 0}},
                {"text": "这是将被修改的旧内容", "metadata": {"index": 1}},
                {"text": "最后一段也不变", "metadata": {"index": 2}},
            ]
            chunks_v2 = [
                {"text": "第一段内容不变", "metadata": {"index": 0}},
                {"text": "这是修改后的新内容", "metadata": {"index": 1}},
                {"text": "最后一段也不变", "metadata": {"index": 2}},
            ]

            save_version(doc_name, chunks_v1, "hash1", created_by="test")
            save_version(doc_name, chunks_v2, "hash2", created_by="test")

            result = compute_diff(doc_name, "v1", "v2")
            self.assertIsNotNone(result)

            summary = result["summary"]
            # 应该有 1 个修改
            self.assertEqual(summary["modified"], 1)
            self.assertEqual(summary["added"], 0)
            self.assertEqual(summary["removed"], 0)
            # 2 个不变
            self.assertEqual(summary["unchanged"], 2)

    def test_diff_detects_added_chunks(self):
        """v2 新增 2 个 chunk，diff 正确识别 added=2。"""
        with self._mock_versions_root():
            doc_name = "test-diff-add.md"
            chunks_v1 = [
                {"text": "原有第一段", "metadata": {"index": 0}},
                {"text": "原有第二段", "metadata": {"index": 1}},
            ]
            chunks_v2 = [
                {"text": "原有第一段", "metadata": {"index": 0}},
                {"text": "新增第一段", "metadata": {"index": 1}},
                {"text": "新增第二段", "metadata": {"index": 2}},
                {"text": "原有第二段", "metadata": {"index": 3}},
            ]

            save_version(doc_name, chunks_v1, "hash1", created_by="test")
            save_version(doc_name, chunks_v2, "hash2", created_by="test")

            result = compute_diff(doc_name, "v1", "v2")
            self.assertIsNotNone(result)

            summary = result["summary"]
            self.assertEqual(summary["added"], 2)
            # 检查 changes 中有 2 个 added
            added_changes = [c for c in result["changes"] if c["type"] == "added"]
            self.assertEqual(len(added_changes), 2)

    def test_diff_detects_removed_chunks(self):
        """v2 删除 1 个 chunk，diff 正确识别 removed=1。"""
        with self._mock_versions_root():
            doc_name = "test-diff-remove.md"
            chunks_v1 = [
                {"text": "第一段", "metadata": {"index": 0}},
                {"text": "将被删除的内容", "metadata": {"index": 1}},
                {"text": "第三段", "metadata": {"index": 2}},
            ]
            chunks_v2 = [
                {"text": "第一段", "metadata": {"index": 0}},
                {"text": "第三段", "metadata": {"index": 1}},
            ]

            save_version(doc_name, chunks_v1, "hash1", created_by="test")
            save_version(doc_name, chunks_v2, "hash2", created_by="test")

            result = compute_diff(doc_name, "v1", "v2")
            self.assertIsNotNone(result)

            summary = result["summary"]
            self.assertEqual(summary["removed"], 1)
            removed_changes = [c for c in result["changes"] if c["type"] == "removed"]
            self.assertEqual(len(removed_changes), 1)

    def test_prune_keeps_latest_n(self):
        """prune --keep 2 后只保留最近 2 个版本。"""
        with self._mock_versions_root():
            doc_name = "test-prune.md"

            for i in range(1, 6):
                chunks = [{"text": f"版本{i}内容", "metadata": {"index": 0}}]
                save_version(doc_name, chunks, f"hash{i}", created_by="test")

            versions_before = list_versions(doc_name)
            self.assertEqual(len(versions_before), 5)

            deleted = prune_versions(doc_name, keep=2)
            self.assertEqual(deleted, 3)

            versions_after = list_versions(doc_name)
            self.assertEqual(len(versions_after), 2)
            # 保留的是 v4 和 v5
            self.assertEqual(versions_after[0]["version"], "v4")
            self.assertEqual(versions_after[1]["version"], "v5")

    def test_version_metadata_fields(self):
        """metadata.json 包含所有必需字段。"""
        with self._mock_versions_root():
            doc_name = "test-meta.md"
            chunks = [
                {"text": "测试内容", "metadata": {"source": doc_name}},
            ]

            version = save_version(doc_name, chunks, "test_hash", created_by="test")
            self.assertEqual(version, "v1")

            data = get_version_detail(doc_name, "v1")
            self.assertIsNotNone(data)

            meta = data["metadata"]
            required_fields = ["version", "timestamp", "doc_hash", "num_chunks", "total_chars", "created_by", "source_file"]
            for field in required_fields:
                self.assertIn(field, meta, f"缺少必需字段: {field}")

            # 类型检查
            self.assertEqual(meta["version"], "v1")
            self.assertEqual(meta["doc_hash"], "test_hash")
            self.assertEqual(meta["num_chunks"], 1)
            self.assertIsInstance(meta["total_chars"], int)
            self.assertEqual(meta["created_by"], "test")
            self.assertEqual(meta["source_file"], doc_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
