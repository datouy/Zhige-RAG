"""pytest 公共配置：隔离测试运行时并在导入应用前关闭模型预热。"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# setdefault：需要验证预热本身的测试可以自行覆盖
os.environ.setdefault("RAG_WARMUP", "0")
os.environ.setdefault("RAG_PRELOAD_LLM", "0")

# 必须在导入 api.main / src.db.database 之前设置。此前测试直接连接
# ``data/users.db`` 和 ``data/chroma_db``，历史损坏数据会让测试失败，测试
# 自身写会话也会污染真实知识库。
_TEST_RUNTIME = Path(tempfile.mkdtemp(prefix="zhige-pytest-"))
_TEST_DATA = _TEST_RUNTIME / "data"
_TEST_DATA.mkdir(parents=True, exist_ok=True)
os.environ["RAG_DATA_DIR"] = str(_TEST_DATA)
os.environ["DB_PATH"] = str(_TEST_DATA / "users.db")
os.environ["VECTOR_STORE_PERSIST_DIR"] = str(_TEST_DATA / "chroma_db")
os.environ["RAW_DOCS_DIR"] = str(_TEST_DATA / "raw")


def pytest_sessionfinish(session, exitstatus):
    """测试结束后清理临时运行时；Windows 文件句柄未释放时允许残留。"""
    shutil.rmtree(_TEST_RUNTIME, ignore_errors=True)
