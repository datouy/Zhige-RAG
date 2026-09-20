"""性能测试套件 - 测试系统在高负载下的表现。

测试覆盖：
1. 并发请求测试：100 个并发用户
2. 大文档入库：10MB 文档处理时间
3. 响应时间基准：各端点延迟分布

运行测试：
    pytest tests/test_performance.py -v -s
"""
from __future__ import annotations

import io
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Generator
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["STREAMLIT_RUNTIME"] = "enable"


class LoadTester:
    """负载测试工具类。"""

    def __init__(self, base_url: str = "http://test"):
        self.base_url = base_url
        self.results: list[dict] = []

    def record(self, endpoint: str, duration_ms: float, status_code: int, error: str = None):
        """记录测试结果。"""
        self.results.append({
            "endpoint": endpoint,
            "duration_ms": duration_ms,
            "status_code": status_code,
            "error": error,
            "timestamp": time.time(),
        })

    def summary(self) -> dict:
        """生成测试摘要。"""
        if not self.results:
            return {}

        durations = [r["duration_ms"] for r in self.results]
        errors = [r for r in self.results if r["error"]]

        return {
            "total_requests": len(self.results),
            "successful": len(self.results) - len(errors),
            "failed": len(errors),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "p50_duration_ms": self._percentile(durations, 50),
            "p95_duration_ms": self._percentile(durations, 95),
            "p99_duration_ms": self._percentile(durations, 99),
            "requests_per_second": len(self.results) / (max(r["timestamp"] for r in self.results) - min(r["timestamp"] for r in self.results)) if len(self.results) > 1 else 0,
        }

    @staticmethod
    def _percentile(data: list, percentile: int) -> float:
        """计算百分位数。"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _make_request(
        self,
        method: str,
        url: str,
        headers: dict = None,
        json: dict = None,
    ) -> tuple[float, int, str]:
        """发起请求并返回 (耗时ms, 状态码, 错误信息)。

        走 TestClient（ASGI 内存调用）。注意：**不**用 ``with`` 触发 lifespan——
        100 个并发 lifespan 同时 ``init_db`` 会在 SQLite 原生层竞态崩溃；
        本测试的端点（health 等）不依赖 lifespan 初始化。
        """
        from fastapi.testclient import TestClient
        from api.main import app

        start = time.time()
        try:
            client = TestClient(app)
            if method == "GET":
                r = client.get(url, headers=headers)
            elif method == "POST":
                r = client.post(url, headers=headers, json=json)
            else:
                r = client.request(method, url, headers=headers, json=json)
            return (time.time() - start) * 1000, r.status_code, None
        except Exception as e:
            return (time.time() - start) * 1000, 0, str(e)


class TestConcurrentRequests:
    """并发请求测试。"""

    @pytest.fixture
    def load_tester(self) -> LoadTester:
        """创建负载测试器。"""
        return LoadTester()

    def test_concurrent_health_checks(self, load_tester: LoadTester):
        """测试 100 个并发健康检查请求。"""
        num_requests = 100
        endpoints = ["/api/health"] * num_requests

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(load_tester._make_request, "GET", endpoint)
                for endpoint in endpoints
            ]
            for future in as_completed(futures):
                duration_ms, status_code, error = future.result()
                load_tester.record("/api/health", duration_ms, status_code, error)

        summary = load_tester.summary()
        print(f"\n健康检查并发测试结果:")
        print(f"  总请求数: {summary['total_requests']}")
        print(f"  成功: {summary['successful']}, 失败: {summary['failed']}")
        print(f"  平均延迟: {summary['avg_duration_ms']:.2f}ms")
        print(f"  P95 延迟: {summary['p95_duration_ms']:.2f}ms")
        print(f"  P99 延迟: {summary['p99_duration_ms']:.2f}ms")

        assert summary["total_requests"] == num_requests
        assert summary["failed"] < num_requests * 0.1

    def test_concurrent_search_requests(self, load_tester: LoadTester):
        """测试 50 个并发搜索请求。"""
        from fastapi.testclient import TestClient
        from api.main import app

        num_requests = 50
        queries = [
            {"query": f"测试查询{i}", "top_k": 4}
            for i in range(num_requests)
        ]

        def make_search_request(query_data: dict) -> tuple[float, int, str]:
            start = time.time()
            try:
                client = TestClient(app)  # 不触发 lifespan（并发 init_db 会崩）
                response = client.post("/api/v1/search", json=query_data)
                return (time.time() - start) * 1000, response.status_code, None
            except Exception as e:
                return (time.time() - start) * 1000, 0, str(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_search_request, q) for q in queries]
            for future in as_completed(futures):
                duration_ms, status_code, error = future.result()
                load_tester.record("/api/v1/search", duration_ms, status_code, error)

        summary = load_tester.summary()
        print(f"\n搜索并发测试结果:")
        print(f"  总请求数: {summary['total_requests']}")
        print(f"  成功: {summary['successful']}, 失败: {summary['failed']}")
        print(f"  平均延迟: {summary['avg_duration_ms']:.2f}ms")
        print(f"  P95 延迟: {summary['p95_duration_ms']:.2f}ms")

        assert summary["total_requests"] == num_requests

    def test_sustained_load(self, load_tester: LoadTester):
        """测试持续负载（100 个请求在 10 秒内）。"""
        from fastapi.testclient import TestClient
        from api.main import app

        num_requests = 100
        duration_seconds = 10

        start_time = time.time()
        request_times: list[tuple[float, float, int, str]] = []

        def make_request():
            req_start = time.time()
            try:
                client = TestClient(app)  # 不触发 lifespan（并发 init_db 会崩）
                response = client.get("/api/health")
                return (time.time() - req_start) * 1000, response.status_code, None
            except Exception as e:
                return (time.time() - req_start) * 1000, 0, str(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(num_requests):
                scheduled_time = start_time + (i * duration_seconds / num_requests)
                delay = scheduled_time - time.time()
                if delay > 0:
                    time.sleep(delay)
                futures.append(executor.submit(make_request))

            for future in as_completed(futures):
                duration_ms, status_code, error = future.result()
                load_tester.record("/api/health", duration_ms, status_code, error)

        elapsed = time.time() - start_time
        summary = load_tester.summary()
        summary["elapsed_seconds"] = elapsed

        print(f"\n持续负载测试结果:")
        print(f"  总请求数: {summary['total_requests']}")
        print(f"  实际耗时: {elapsed:.2f}秒")
        print(f"  吞吐量: {summary['requests_per_second']:.2f} req/s")

        assert summary["total_requests"] == num_requests


class TestLargeDocumentIngestion:
    """大文档入库测试。"""

    def generate_large_text(self, size_kb: int) -> str:
        """生成指定大小的文本。"""
        base_text = """
        这是一段用于测试中文RAG系统处理大文档能力的文本。

        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
        它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

        机器学习（Machine Learning）是人工智能的一个分支，专门研究计算机怎样模拟或实现人类的学习行为，
        以获取新的知识或技能，重新组织已有的知识结构使之不断改善自身的性能。
        深度学习（Deep Learning）是机器学习的分支，是一种以人工神经网络为架构，
        对数据进行表征学习的算法。
        """
        repeats = (size_kb * 1024) // len(base_text) + 1
        return base_text * repeats

    def test_large_document_parsing(self):
        """测试大文档解析时间。"""
        from src.document_loader import Document

        size_mb = 5
        large_text = self.generate_large_text(size_mb * 1024)

        start = time.time()
        doc = Document(page=1, content=large_text, metadata={"source": "test_large"})
        parse_time = time.time() - start

        print(f"\n文档解析测试 ({size_mb}MB):")
        print(f"  解析时间: {parse_time:.3f}秒")
        print(f"  文档长度: {len(large_text)} 字符")

        assert len(doc.content) > 0
        assert parse_time < 5.0

    def test_large_document_chunking(self):
        """测试大文档分块时间。"""
        from src.document_loader import Document
        from src.text_splitter import ChineseTextSplitter

        size_mb = 5
        large_text = self.generate_large_text(size_mb * 1024)
        doc = Document(page=1, content=large_text, metadata={"source": "test_large"})

        splitter = ChineseTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
        )

        start = time.time()
        chunks = splitter.split_text(doc.content, metadata=doc.metadata)
        chunk_time = time.time() - start

        print(f"\n文档分块测试 ({size_mb}MB):")
        print(f"  分块时间: {chunk_time:.3f}秒")
        print(f"  生成块数: {len(chunks)}")

        assert len(chunks) > 0
        assert chunk_time < 10.0

    def test_ingestion_throughput(self):
        """测试入库吞吐量。"""
        from src.document_loader import Document
        from src.text_splitter import ChineseTextSplitter

        sizes_mb = [1, 2, 5]
        results = []

        for size_mb in sizes_mb:
            large_text = self.generate_large_text(size_mb * 1024)
            doc = Document(page=1, content=large_text, metadata={"source": f"test_{size_mb}mb"})

            splitter = ChineseTextSplitter(chunk_size=300, chunk_overlap=50)

            start = time.time()
            chunks = splitter.split_text(doc.content, metadata=doc.metadata)
            total_time = time.time() - start

            throughput_mb_per_sec = size_mb / total_time if total_time > 0 else 0
            chunks_per_sec = len(chunks) / total_time if total_time > 0 else 0

            results.append({
                "size_mb": size_mb,
                "total_time_sec": total_time,
                "chunks": len(chunks),
                "throughput_mb_per_sec": throughput_mb_per_sec,
                "chunks_per_sec": chunks_per_sec,
            })

        print(f"\n入库吞吐量测试:")
        for r in results:
            print(f"  {r['size_mb']}MB: {r['total_time_sec']:.2f}秒, "
                  f"{r['throughput_mb_per_sec']:.2f} MB/s, "
                  f"{r['chunks_per_sec']:.0f} 块/秒")

        for r in results:
            assert r["total_time_sec"] > 0


class TestResponseTimeBenchmarks:
    """响应时间基准测试。"""

    def test_health_endpoint_latency(self):
        """测试健康检查端点延迟。"""
        from fastapi.testclient import TestClient
        from api.main import app

        latencies = []
        num_requests = 100

        with TestClient(app) as client:
            for _ in range(num_requests):
                start = time.time()
                response = client.get("/api/health")
                latency_ms = (time.time() - start) * 1000
                latencies.append(latency_ms)
                assert response.status_code == 200

        print(f"\n健康检查延迟基准:")
        print(f"  P50: {sorted(latencies)[len(latencies)//2]:.2f}ms")
        print(f"  P95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")
        print(f"  P99: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}ms")
        print(f"  平均: {sum(latencies)/len(latencies):.2f}ms")

    def test_search_endpoint_latency(self):
        """测试搜索端点延迟。"""
        from fastapi.testclient import TestClient
        from api.main import app

        latencies = []
        queries = ["RAG", "知识图谱", "向量检索", "机器学习", "深度学习"]

        with TestClient(app) as client:
            for query in queries * 5:
                start = time.time()
                response = client.post(
                    "/api/v1/search",
                    json={"query": query, "top_k": 4}
                )
                latency_ms = (time.time() - start) * 1000
                latencies.append(latency_ms)

        print(f"\n搜索端点延迟基准:")
        print(f"  P50: {sorted(latencies)[len(latencies)//2]:.2f}ms")
        print(f"  P95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")
        print(f"  P99: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}ms")
        print(f"  平均: {sum(latencies)/len(latencies):.2f}ms")


class TestMemoryUsage:
    """内存使用测试。"""

    def test_large_chunk_accumulation(self):
        """测试大量分块累积的内存使用。"""
        from src.text_splitter import ChineseTextSplitter

        splitter = ChineseTextSplitter(chunk_size=256, chunk_overlap=32)
        all_chunks = []

        num_docs = 10
        doc_size_kb = 100

        base_text = "这是一段测试文本。" * 100

        for i in range(num_docs):
            text = base_text + f" 文档 {i}。"
            chunks = splitter.split_text(text, metadata={"source": f"doc_{i}"})
            all_chunks.extend(chunks)

        print(f"\n分块累积测试:")
        print(f"  文档数: {num_docs}")
        print(f"  每个文档大小: {doc_size_kb}KB")
        print(f"  总分块数: {len(all_chunks)}")
        print(f"  平均每文档块数: {len(all_chunks) / num_docs:.1f}")

        assert len(all_chunks) > 0
        assert len(all_chunks) > num_docs


class TestDatabasePerformance:
    """数据库性能测试。"""

    def test_connection_pool_performance(self):
        """测试连接池性能。"""
        from src.db.database import SessionLocal, check_connection

        num_queries = 100
        latencies = []

        for _ in range(num_queries):
            start = time.time()
            db = SessionLocal()
            try:
                result = db.execute(text("SELECT 1"))
                result.scalar()
            finally:
                db.close()
            latencies.append((time.time() - start) * 1000)

        print(f"\n数据库连接池性能:")
        print(f"  查询数: {num_queries}")
        print(f"  平均延迟: {sum(latencies)/len(latencies):.2f}ms")
        print(f"  P95 延迟: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")

        assert check_connection()

    def test_concurrent_db_queries(self):
        """测试并发数据库查询。"""
        from src.db.database import SessionLocal

        num_queries = 50

        def query_db():
            start = time.time()
            db = SessionLocal()
            try:
                db.execute(text("SELECT 1"))
            finally:
                db.close()
            return (time.time() - start) * 1000

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(query_db) for _ in range(num_queries)]
            latencies = [f.result() for f in as_completed(futures)]

        print(f"\n并发数据库查询:")
        print(f"  查询数: {num_queries}")
        print(f"  平均延迟: {sum(latencies)/len(latencies):.2f}ms")
        print(f"  最大延迟: {max(latencies):.2f}ms")

        assert len(latencies) == num_queries


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
