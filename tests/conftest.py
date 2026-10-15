"""pytest 公共配置：将项目根目录加入 sys.path。

另外关掉 API 启动期预热（``api/main.py::_warm_up_models``）。每个
``TestClient(app)`` 都会跑一次 lifespan，若不禁用，测试会去加载真实的
embedding 模型与向量库 —— 既慢，也让单元测试依赖本机模型是否就绪。
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# setdefault：需要验证预热本身的测试可以自行覆盖
os.environ.setdefault("RAG_WARMUP", "0")
os.environ.setdefault("RAG_PRELOAD_LLM", "0")
