"""CPU-friendly ONNX embedding backend.

The default local deployment keeps a small BGE embedding model on disk, but
does not need PyTorch, Transformers, CUDA, Ollama, or a generative LLM.  The
class intentionally matches the ``EmbeddingModel.encode`` contract used by the
vector store.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence

import numpy as np

from .utils import Timer, get_logger, resolve_path

logger = get_logger("onnx_embedding")


class OnnxEmbedding:
    """Run a sentence-embedding ONNX model with CPU execution."""

    def __init__(
        self,
        model_path: str = "./models/bge-small-zh-v1.5-onnx",
        model_file: str = "onnx/model_quantized.onnx",
        tokenizer_file: str = "tokenizer.json",
        batch_size: int = 16,
        max_seq_length: int = 512,
        normalize: bool = True,
        num_threads: Optional[int] = None,
        pooling: str = "cls",
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
            from tokenizers import Tokenizer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ONNX Embedding 需要 onnxruntime 与 tokenizers："
                "`pip install onnxruntime tokenizers`"
            ) from exc

        root = resolve_path(model_path)
        onnx_path = resolve_path(root / model_file)
        tokenizer_path = resolve_path(root / tokenizer_file)
        config_path = resolve_path(root / "config.json")
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX Embedding 模型不存在：{onnx_path}。"
                "请重新执行 `python scripts/setup_onnx_embedding.py`。"
            )
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"ONNX Embedding tokenizer 不存在：{tokenizer_path}")

        self.model_name = str(root)
        self.model_path = onnx_path
        self.batch_size = max(1, int(batch_size))
        self.max_seq_length = max(8, int(max_seq_length))
        self.normalize = bool(normalize)
        self.pooling = str(pooling or "cls").strip().lower()

        cpu_count = os.cpu_count() or 4
        # i5-13400 has 16 logical CPUs; 8 avoids oversubscription while leaving
        # room for FastAPI, Chroma, and document parsing.
        self.num_threads = max(
            1,
            min(int(num_threads or min(8, cpu_count)), cpu_count),
        )

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.num_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        logger.info(
            "加载 ONNX Embedding: %s (threads=%d, pooling=%s)",
            onnx_path,
            self.num_threads,
            self.pooling,
        )
        with Timer(f"加载 ONNX Embedding {onnx_path.name}"):
            self.session = ort.InferenceSession(
                str(onnx_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

        self._input_names = {item.name for item in self.session.get_inputs()}
        self._output_name = self.session.get_outputs()[0].name
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=self.max_seq_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        self.dim = 0
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.dim = int(config.get("hidden_size") or 0)
            except (OSError, ValueError, TypeError):
                self.dim = 0

    def encode(
        self,
        texts: Sequence[str] | str,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        convert_to_numpy: bool = True,
    ):
        """Encode texts into normalized float32 vectors."""
        del show_progress
        if isinstance(texts, str):
            texts = [texts]
        items = [str(text) if text else " " for text in texts]
        if not items:
            return np.zeros((0, self.dim), dtype="float32")

        bs = max(1, int(batch_size or self.batch_size))
        vectors: list[np.ndarray] = []
        with Timer(f"embedding(onnx) encode x{len(items)}"):
            for start in range(0, len(items), bs):
                vectors.extend(self._encode_batch(items[start : start + bs]))

        array = np.asarray(vectors, dtype=np.float32)
        if self.normalize and array.size:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            array = array / norms
        if not self.dim and array.size:
            self.dim = int(array.shape[1])
        return array if convert_to_numpy else array.tolist()

    def encode_one(self, text: str):
        return self.encode([text], batch_size=1)[0]

    @property
    def dimension(self) -> int:
        return self.dim

    def _encode_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        encodings = self.tokenizer.encode_batch(list(texts))
        feed: dict[str, Any] = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = np.asarray(
                [encoding.ids for encoding in encodings], dtype=np.int64
            )
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = np.asarray(
                [encoding.attention_mask for encoding in encodings], dtype=np.int64
            )
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.asarray(
                [encoding.type_ids for encoding in encodings], dtype=np.int64
            )

        outputs = self.session.run([self._output_name], feed)[0]
        if outputs.ndim == 2:
            return [np.asarray(row, dtype=np.float32) for row in outputs]
        if outputs.ndim != 3:
            raise ValueError(f"ONNX Embedding 输出维度异常：{outputs.shape}")

        if self.pooling == "mean":
            mask = feed.get("attention_mask")
            if mask is None:
                pooled = outputs.mean(axis=1)
            else:
                weights = mask.astype(np.float32)[..., None]
                pooled = (outputs * weights).sum(axis=1) / np.maximum(
                    weights.sum(axis=1), 1.0
                )
        else:
            pooled = outputs[:, 0, :]
        return [np.asarray(row, dtype=np.float32) for row in pooled]


__all__ = ["OnnxEmbedding"]
