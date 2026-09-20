"""通用工具函数：日志、计时、文件加载、路径处理等。"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional


# ----------------------------------------------------------------------
#  路径处理
# ----------------------------------------------------------------------
def get_project_root() -> Path:
    """获取项目根目录（配置文件所在目录的祖父目录）。"""
    return Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path, base: Optional[Path] = None) -> Path:
    """将相对路径解析为绝对路径。

    Args:
        path: 相对或绝对路径字符串。
        base: 基准目录，默认为项目根目录。

    Returns:
        解析后的绝对 Path 对象。
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or get_project_root()) / p


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，若不存在则创建。"""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------
#  设备选择（torch 可选依赖）
# ----------------------------------------------------------------------
def select_torch_device(device: str = "auto", logger: Optional[logging.Logger] = None) -> str:
    """解析设备字符串，返回 torch 系组件可识别的设备。

    注意：``torch.cuda.is_available()`` 在 Windows + NVIDIA 驱动存在但 torch
    为 CPU 版时仍可能返回 True，``module.to('cuda')`` 会在此时抛出
    ``AssertionError: Torch not compiled with CUDA enabled``。因此这里额外
    校验 ``torch.version.cuda``，只在真正 CUDA 编译的 torch 上返回 cuda，
    否则回落到 cpu。

    embeddings / reranker / llm 三处共用本函数，避免行为漂移。
    """
    if device and device != "auto":
        if device.startswith("cuda"):
            try:
                import torch  # type: ignore

                if not (torch.cuda.is_available() and torch.version.cuda):
                    if logger is not None:
                        logger.warning(
                            "请求 %s 但当前 torch 未启用 CUDA（torch.version.cuda=%s），自动回落到 cpu",
                            device,
                            getattr(torch.version, "cuda", None),
                        )
                    return "cpu"
            except Exception:
                return "cpu"
        return device
    try:
        import torch  # type: ignore

        if torch.cuda.is_available() and torch.version.cuda:
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ----------------------------------------------------------------------
#  日志
# ----------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """结构化 JSON 日志格式化器，适合日志聚合系统。"""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id") and record.request_id:
            log_obj["request_id"] = record.request_id
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "user_id"):
            log_obj["user_id"] = record.user_id
        if hasattr(record, "duration_ms"):
            log_obj["duration_ms"] = record.duration_ms
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logger(
    name: str = "ChineseRAGKB",
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    console: bool = True,
    json_format: bool = False,
) -> logging.Logger:
    """初始化日志器。

    Args:
        name: Logger 名称。
        level: 日志级别字符串，如 INFO / DEBUG。
        log_file: 日志文件路径，可选。
        console: 是否输出到控制台。
        json_format: 是否使用 JSON 格式化输出（用于生产环境）。
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    if json_format:
        fmt = JSONFormatter()
    else:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_file:
        log_path = resolve_path(log_file)
        ensure_dir(log_path.parent)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    # P3.1: 自动挂载 request_id filter，让每条日志都带上上下文中的 request_id
    try:
        from src.middleware.logging import RequestIDContextFilter
        for handler in logger.handlers:
            if not any(isinstance(f, RequestIDContextFilter) for f in handler.filters):
                handler.addFilter(RequestIDContextFilter())
    except ImportError:
        pass
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取一个子 Logger，继承根 Logger 的配置。

    首次调用时自动把 :class:`RequestIDContextFilter` 挂到根 logger 的所有
    handler 上，确保每条日志都自动带上当前请求的 ``request_id``（P3.1）。
    """
    if name:
        sub = logging.getLogger(f"ChineseRAGKB.{name}")
    else:
        sub = logging.getLogger("ChineseRAGKB")
    _ensure_request_id_filter()
    return sub


_REQUEST_ID_FILTER_INSTALLED = False


def _ensure_request_id_filter() -> None:
    """确保 :class:`RequestIDContextFilter` 已挂在根 logger 上（仅执行一次）。"""
    global _REQUEST_ID_FILTER_INSTALLED
    if _REQUEST_ID_FILTER_INSTALLED:
        return
    try:
        from src.middleware.logging import RequestIDContextFilter
    except ImportError:
        return
    root = logging.getLogger("ChineseRAGKB")
    for handler in root.handlers:
        # 幂等：检查是否已挂过
        if not any(isinstance(f, RequestIDContextFilter) for f in handler.filters):
            handler.addFilter(RequestIDContextFilter())
    _REQUEST_ID_FILTER_INSTALLED = True


# ----------------------------------------------------------------------
#  计时
# ----------------------------------------------------------------------
class Timer:
    """上下文计时器，用于统计代码块耗时。"""

    def __init__(self, name: str = "block", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger()
        self.start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        self.logger.info("%s 耗时 %.2f ms", self.name, self.elapsed_ms)


def timer(name: str = "func") -> Callable:
    """装饰器：统计函数执行耗时。"""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer(name=f"{name}.{fn.__name__}"):
                return fn(*args, **kwargs)

        return wrapper

    return deco


# ----------------------------------------------------------------------
#  文本处理
# ----------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """基础文本清洗：合并多余空白、统一换行。

    Args:
        text: 原始文本。

    Returns:
        清洗后的文本。
    """
    if not text:
        return ""
    # 去除首尾空白
    text = text.strip()
    # 合并空格与制表符
    text = _WHITESPACE_RE.sub(" ", text)
    # 合并多余空行
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text


def chunk_text_by_length(text: str, max_chars: int) -> List[str]:
    """将长文本按最大字符数切片（中英文字符皆按 1 计数）。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text] if text else []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


# ----------------------------------------------------------------------
#  文件与目录
# ----------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def iter_doc_files(root: str | Path, recursive: bool = True) -> Iterable[Path]:
    """遍历目录下所有受支持文档文件。"""
    root = resolve_path(root)
    if not root.exists():
        return
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield p


def file_md5(path: str | Path, chunk_size: int = 65536) -> str:
    """计算文件 MD5，用于去重。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
#  环境变量加载（简易 .env）
# ----------------------------------------------------------------------
def load_env_file(env_path: str | Path = ".env", override: bool = False) -> None:
    """加载 .env 文件到环境变量（不依赖 python-dotenv）。

    Args:
        env_path: .env 文件路径。
        override: 是否覆盖已有环境变量。
    """
    path = resolve_path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not override and key in os.environ:
            continue
        os.environ[key] = value


# ----------------------------------------------------------------------
#  配置加载
# ----------------------------------------------------------------------
# config.yaml 中形如 ${VAR} 或 ${VAR:-default} 的占位符，在加载时用
# 环境变量插值（YAML 本身不支持；例如 knowledge_graph.neo4j.password）
_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate_env(value: Any) -> Any:
    """递归对配置值做环境变量插值。"""
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    if isinstance(value, str):
        def _sub(m: "re.Match[str]") -> str:
            var, default = m.group(1), m.group(2)
            env = os.environ.get(var)
            if env is not None:
                return env
            return default if default is not None else m.group(0)
        return _ENV_PLACEHOLDER_RE.sub(_sub, value)
    return value


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    """加载 YAML 配置文件。

    若 PyYAML 未安装，则抛出 ImportError 提示用户安装。
    支持 ``${VAR}`` / ``${VAR:-default}`` 形式的环境变量占位符。
    """
    import yaml  # type: ignore

    path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件未找到: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _interpolate_env(cfg)


def merge_dict(base: dict, override: dict) -> dict:
    """递归合并两个 dict（override 覆盖 base）。"""
    result = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_dict(result[k], v)
        else:
            result[k] = v
    return result


# ----------------------------------------------------------------------
#  环境变量覆盖支持
# ---------------------------------------------------------------------
def apply_env_overrides(cfg: dict) -> dict:
    """根据环境变量覆盖部分常用配置。"""
    mapping = {
        "DEVICE": ("embedding", "device"),
        "EMBEDDING_DEVICE": ("embedding", "device"),
        "LLM_DEVICE": ("llm", "device"),
        "USE_4BIT": ("llm", "quantization", "enabled"),
        "STREAMLIT_SERVER_PORT": ("ui", "server_port"),
    }
    for env_key, path in mapping.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        cur = cfg
        for p in path[:-1]:
            cur = cur.setdefault(p, {})
        key = path[-1]
        # 简单类型推断
        v_lower = val.lower()
        if v_lower in ("true", "false"):
            cur[key] = v_lower == "true"
        else:
            cur[key] = val
    return cfg