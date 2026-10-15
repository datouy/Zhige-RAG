"""d1 决策原语层（Decision Primitives）。

设计来源
--------
参考 Liquid AI 的 **d1 决策模型** 思路：不再走"逐 token 自回归生成"，
而是用**单次前向**直接输出一个决策结论，从而做到毫秒级、0 输出 token、
彻底根除"格式崩塌 / 幻觉"。

d1 定义了三种面向决策的基础原语（Primitives）：

- ``noul``  —— 是/否判定，直接输出 0~1 的校准概率（如"这条文本是不是垃圾"）；
- ``choice``—— 多选一分类，输入一组选项与判定标准，输出各选项概率分布 + 最高置信项；
- ``score`` —— 有序梯度评分（2~10 级阶梯），输出期望分值、置信度与概率分布。

本模块把"决策"抽象成可替换的 provider，映射到本项目的三处高价值环节：

============  ==================================  ==================
环节          文件                                原语
============  ==================================  ==================
检索重排      ``src/reranker.py``                 ``score``
查询意图路由  ``src/query_router.py``             ``choice``
Agent 工具调度 ``src/agent/react_agent.py``        ``choice``
答案校验      ``src/verifier.py``                 ``noul`` + ``score``
============  ==================================  ==================

后端（``decision.backend``）
----------------------------
- ``d1``   —— **原生**加载 ``LiquidAI/d1-3B``（``transformers>=5.14`` + ``torch``），
  调用 ``model.system_one(text, questions)``：真正的单次前向、0 输出 token。
  需自备权重；无 GPU 时 CPU 也可跑（延迟更高但仍远低于生成式模型）。
- ``llm``  —— 用**项目已有的 LLM**（本地 Qwen / Ollama / OpenAI 兼容）以严格
  JSON 约束模拟三种原语。**不装 torch、不下载新权重也能用**，代价是延迟更高、
  且需要防格式漂移（本模块已内置多重解析兜底）。
- ``auto`` —— 先试 ``d1``，权重不可用则回落到 ``llm``。
- ``none`` —— 关闭，调用方回退到原有实现（正则 / bge 交叉编码器 / 规则校验）。

设计约束
--------
1. **绝不抛出**：所有对外方法在异常时返回 ``None``，让调用方无痛降级
   （决策层是"锦上添花"，不能拖垮主链路 —— 与 reranker 的既有约定一致）。
2. **零新增核心依赖**：``llm`` 后端复用 :func:`src.llm_provider.create_llm`，
   ``d1`` 后端按需 import torch/transformers。
3. **可测**：``decide`` 是唯一需要实现的抽象方法，测试用 fake provider 即可。
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from .utils import Timer, get_logger, select_torch_device

logger = get_logger("decision_provider")


# ======================================================================
#  原语类型常量
# ======================================================================
TYPE_NOUL = "noul"
TYPE_CHOICE = "choice"
TYPE_SCORE = "score"
SUPPORTED_PRIMITIVES = (TYPE_NOUL, TYPE_CHOICE, TYPE_SCORE)

# score 原语的阶梯数下限/上限（对齐 d1 官方 2~10 级）
SCORE_MIN_LEVELS = 2
SCORE_MAX_LEVELS = 10


# ======================================================================
#  结果类型
# ======================================================================
@dataclass
class NoulResult:
    """``noul`` 原语结果：是/否 + 校准概率。"""

    probability: float = 0.0  # P(yes)，0~1
    label: str = "no"  # yes / no
    raw: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"type": TYPE_NOUL, "probability": round(self.probability, 4), "label": self.label}


@dataclass
class ChoiceResult:
    """``choice`` 原语结果：多选一的概率分布与最高置信项。"""

    label: str = ""
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    raw: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": TYPE_CHOICE,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


@dataclass
class ScoreResult:
    """``score`` 原语结果：有序梯度评分。"""

    score: float = 0.0  # 期望分值（1~N，N=criteria 级数）
    normalized: float = 0.0  # 归一化到 0~1
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    raw: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": TYPE_SCORE,
            "score": round(self.score, 4),
            "normalized": round(self.normalized, 4),
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


DecisionValue = Union[NoulResult, ChoiceResult, ScoreResult]


# ======================================================================
#  归一化工具
# ======================================================================
def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _normalize_distribution(probs: Dict[str, Any]) -> Dict[str, float]:
    """把任意数值字典归一化成概率分布（总和为 1，负值截断为 0）。"""
    clean: Dict[str, float] = {}
    total = 0.0
    for k, v in (probs or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = 0.0
        f = 0.0 if f < 0.0 else f
        clean[str(k)] = f
        total += f
    if total <= 0:
        n = len(clean) or 1
        return {k: 1.0 / n for k in clean}
    return {k: v / total for k, v in clean.items()}


def _softmax_like(center_idx: int, n: int, sharpness: float = 1.6) -> Dict[int, float]:
    """围绕 ``center_idx`` 生成一个平滑的概率分布（用于 LLM 模拟 score 时补分布）。

    真正的 d1 会直接给出概率分布；LLM 后端只能给出一个离散档位，这里用一个
    几何衰减核合成"看起来合理"的分布，避免下游拿到一个 one-hot 的退化结果。
    """
    if n <= 1:
        return {0: 1.0}
    weights: Dict[int, float] = {}
    total = 0.0
    for i in range(n):
        w = 1.0 / (1.0 + abs(i - center_idx) * sharpness)
        weights[i] = w
        total += w
    return {i: w / total for i, w in weights.items()}


def _validate_spec(name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """校验并规整一个 question 规格，返回 ``{type, instructions, criteria}``。"""
    if not isinstance(spec, dict):
        raise ValueError(f"决策问题 {name!r} 的规格必须是 dict")
    qtype = str(spec.get("type") or "").strip().lower()
    if qtype not in SUPPORTED_PRIMITIVES:
        raise ValueError(
            f"决策问题 {name!r} 的 type={qtype!r} 非法，可选：{'/'.join(SUPPORTED_PRIMITIVES)}"
        )
    instructions = str(spec.get("instructions") or "").strip()
    criteria = spec.get("criteria")
    if qtype == TYPE_CHOICE:
        if isinstance(criteria, (list, tuple)):
            criteria = {str(c): "" for c in criteria}
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError(f"choice 问题 {name!r} 需要 >=2 个选项的 criteria（dict 或 list）")
        criteria = {str(k): str(v or "") for k, v in criteria.items()}
    elif qtype == TYPE_SCORE:
        if isinstance(criteria, dict):
            criteria = list(criteria.keys())
        if not isinstance(criteria, (list, tuple)):
            raise ValueError(f"score 问题 {name!r} 需要有序 criteria（list）")
        if not (SCORE_MIN_LEVELS <= len(criteria) <= SCORE_MAX_LEVELS):
            raise ValueError(
                f"score 问题 {name!r} 的阶梯数须在 {SCORE_MIN_LEVELS}~{SCORE_MAX_LEVELS} 之间，"
                f"当前 {len(criteria)}"
            )
        criteria = [str(c) for c in criteria]
    return {"type": qtype, "instructions": instructions, "criteria": criteria}


# ======================================================================
#  抽象基类
# ======================================================================
class BaseDecisionProvider(ABC):
    """决策 provider 基类。

    子类只需实现 :meth:`decide`；三个便捷方法（noul/choice/score）由基类
    统一封装，保证两种后端的行为与返回类型完全一致。
    """

    name: str = "base"

    # ------------------------------------------------------------------
    @abstractmethod
    def decide(self, text: str, questions: Dict[str, Dict[str, Any]]) -> Dict[str, DecisionValue]:
        """对 ``text`` 一次性回答多个决策问题（对齐 d1 的 ``system_one``）。

        Args:
            text: 待决策的输入文本（如"用户问题 + 候选片段"）。
            questions: ``{问题名: {"type":..., "instructions":..., "criteria":...}}``。

        Returns:
            ``{问题名: NoulResult | ChoiceResult | ScoreResult}``。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _safe_decide(
        self, text: str, questions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, DecisionValue]:
        """包装 ``decide``：任何异常都吞掉并返回空 dict，绝不向上抛。"""
        if not text or not questions:
            return {}
        try:
            with Timer(f"decision[{self.name}] x{len(questions)}"):
                return self.decide(text, questions) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("决策失败（已降级）: %s", exc)
            return {}

    # ------------------------------------------------------------------
    def noul(self, text: str, instructions: str, key: str = "noul") -> Optional[NoulResult]:
        """是/否判定。失败返回 ``None``。"""
        res = self._safe_decide(text, {key: {"type": TYPE_NOUL, "instructions": instructions}})
        val = res.get(key)
        return val if isinstance(val, NoulResult) else None

    def choice(
        self,
        text: str,
        instructions: str,
        criteria: Union[Dict[str, str], Sequence[str]],
        key: str = "choice",
    ) -> Optional[ChoiceResult]:
        """多选一分类。失败返回 ``None``。"""
        res = self._safe_decide(
            text,
            {key: {"type": TYPE_CHOICE, "instructions": instructions, "criteria": criteria}},
        )
        val = res.get(key)
        return val if isinstance(val, ChoiceResult) else None

    def score(
        self,
        text: str,
        instructions: str,
        criteria: Sequence[str],
        key: str = "score",
    ) -> Optional[ScoreResult]:
        """有序梯度评分。失败返回 ``None``。"""
        res = self._safe_decide(
            text,
            {key: {"type": TYPE_SCORE, "instructions": instructions, "criteria": criteria}},
        )
        val = res.get(key)
        return val if isinstance(val, ScoreResult) else None

    def close(self) -> None:  # pragma: no cover - 默认无资源
        """释放资源（默认空操作）。"""
        return None


# ======================================================================
#  后端一：原生 d1 模型（transformers）
# ======================================================================
class D1LocalProvider(BaseDecisionProvider):
    """原生 ``LiquidAI/d1-3B`` 决策模型（单次前向）。

    依赖 ``transformers>=5.14`` 与 ``torch``（见 ``requirements-d1.txt``）。
    权重未就绪时会抛异常，由 :func:`create_decision_provider` 捕获并降级。

    Args:
        model_name: HF 模型名或本地权重目录，默认 ``LiquidAI/d1-3B``。
        device: auto / cpu / cuda / mps。
        cache_dir: HF 缓存目录。
        local_files_only: 只用本地权重、不联网（默认 True，符合本项目"离线优先"）。
        dtype: torch dtype；留空时按设备自动选（CPU→float32，其它→bfloat16）。
        trust_remote_code: d1 使用自定义建模代码，需为 True。
    """

    name = "d1"

    def __init__(
        self,
        model_name: str = "LiquidAI/d1-3B",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        local_files_only: bool = True,
        dtype: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> None:
        import torch  # noqa: PLC0415 - 可选依赖，按需 import
        from transformers import AutoModel  # type: ignore  # noqa: PLC0415

        self.model_name = model_name
        self.device = select_torch_device(device, logger=logger)

        if dtype:
            torch_dtype = getattr(torch, str(dtype), torch.float32)
        else:
            torch_dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        with Timer(f"加载 d1 决策模型 {model_name}"):
            try:
                # transformers>=5.14 使用 dtype 参数名；旧版本为 torch_dtype
                model = AutoModel.from_pretrained(model_name, dtype=torch_dtype, **kwargs)
            except TypeError:
                model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, **kwargs)
            self.model = model.to(self.device)
            self.model.eval()

        if not hasattr(self.model, "system_one"):
            raise ImportError(
                f"{model_name} 未暴露 system_one（不是 d1 决策模型？），"
                "请改用 decision.backend='llm' 或用 d1 官方权重。"
            )
        logger.info("d1 决策模型就绪：%s（device=%s）", model_name, self.device)

    # ------------------------------------------------------------------
    def decide(self, text: str, questions: Dict[str, Dict[str, Any]]) -> Dict[str, DecisionValue]:
        specs: Dict[str, Dict[str, Any]] = {}
        for name, spec in questions.items():
            specs[name] = _validate_spec(name, spec)
        # d1 原生接口只认 type/instructions/criteria 三个字段
        payload = {
            n: {"type": s["type"], "instructions": s["instructions"], "criteria": s["criteria"]}
            for n, s in specs.items()
        }
        raw = self.model.system_one(text, payload)
        if not isinstance(raw, dict):
            # 个别版本可能包一层 {"results": {...}} 或返回 list
            raw = getattr(raw, "__dict__", None) or {}
            if not raw:
                raise ValueError(f"d1 system_one 返回了非 dict 结构：{type(raw).__name__}")
        out: Dict[str, DecisionValue] = {}
        for name, spec in specs.items():
            if name in raw:
                out[name] = _coerce_result(name, spec, raw[name])
        return out


# ======================================================================
#  后端二：用现有 LLM 模拟三种原语
# ======================================================================
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_LLM_SYSTEM_PROMPT = (
    "你是一个「决策分类器」，不做自由生成，只输出一个 JSON 对象。\n"
    "严格遵循：\n"
    "1) 只输出 JSON，不要任何解释、Markdown 代码块或多余文字；\n"
    "2) 顶层 key 必须与题目给定的名称完全一致；\n"
    "3) noul 题型输出 {\"probability\": 0~1 的小数}；\n"
    "4) choice 题型输出 {\"probabilities\": {\"选项名\": 0~1 的小数, ...}}，"
    "各选项都要出现，且总和约为 1；\n"
    "5) score 题型输出 {\"scores\": {\"档位名\": 0~1 的小数, ...}}，"
    "按档位给出权重，越相关权重越高，总和约为 1。"
)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出中尽力抽取一个 JSON 对象（容忍代码块与前后噪声）。"""
    if not text:
        return None
    s = str(text).strip()
    # 去掉 ```json ... ``` 包裹
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        pass
    m = _JSON_BLOCK_RE.search(s)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:  # noqa: BLE001
            # 常见修复：去掉尾随逗号
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                obj = json.loads(fixed)
                return obj if isinstance(obj, dict) else None
            except Exception:  # noqa: BLE001
                return None
    return None


class LLMDecisionProvider(BaseDecisionProvider):
    """用项目已有的 LLM 模拟 d1 三种原语（无需 torch / 无需新权重）。

    适用场景：手头只有本地 Qwen / Ollama / 云端 API，暂时跑不动 d1-3B。
    通过严格 JSON 约束 + 多重解析兜底，把生成式模型"逼"成决策分类器。

    注意：这是**模拟**而非单次前向，延迟与稳定性都不如原生 d1；但相比让模型
    自由输出再正则解析（如 ReAct 的 ``Action:`` 解析），本方案的格式可靠性
    显著更高，且天然产出可用的概率/置信度。

    Args:
        llm: 任意满足 ``chat(messages, generation=None, stream=False) -> str`` 的实例。
        max_new_tokens: 决策输出很短，限制 token 数可显著降延迟。
    """

    name = "llm"

    def __init__(self, llm: Any, max_new_tokens: int = 256) -> None:
        if llm is None:
            raise ValueError("LLMDecisionProvider 需要非空 llm")
        self.llm = llm
        self.max_new_tokens = int(max_new_tokens)

    # ------------------------------------------------------------------
    def _render_prompt(self, text: str, specs: Dict[str, Dict[str, Any]]) -> str:
        parts: List[str] = ["输入文本：", "<<<", text.strip(), ">>>", "", "需要判定的题目："]
        for name, spec in specs.items():
            parts.append(f"- 题名 {name!r}（类型 {spec['type']}）")
            if spec["instructions"]:
                parts.append(f"  判定标准：{spec['instructions']}")
            if spec["type"] == TYPE_CHOICE:
                parts.append("  可选项：")
                for opt, desc in spec["criteria"].items():
                    parts.append(f"    * {opt}" + (f"：{desc}" if desc else ""))
                parts.append(f'  输出形如：{{"{name}": {{"probabilities": {{"选项名": 0.0~1.0, ...}}}}}}')
            elif spec["type"] == TYPE_SCORE:
                parts.append("  评分档位（由低到高）：" + " < ".join(spec["criteria"]))
                parts.append(
                    f'  输出形如：{{"{name}": {{"scores": {{"档位名": 0.0~1.0, ...}}}}}}'
                )
            else:  # noul
                parts.append(
                    f'  输出形如：{{"{name}": {{"probability": 0.0~1.0}}}}（1 表示"是"）'
                )
        parts.append("")
        parts.append("只输出 JSON：")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def decide(self, text: str, questions: Dict[str, Dict[str, Any]]) -> Dict[str, DecisionValue]:
        specs = {n: _validate_spec(n, s) for n, s in questions.items()}
        prompt = self._render_prompt(text, specs)
        messages = [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            gen = getattr(self.llm, "gen_cfg", None)
            out = self.llm.chat(messages, gen, stream=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("决策 LLM 调用失败: %s", exc)
            return {}
        obj = _extract_json_object(out if isinstance(out, str) else str(out))
        if not obj:
            logger.warning("决策 LLM 未返回可解析 JSON，已跳过（原文前 120 字：%s）", str(out)[:120])
            return {}
        results: Dict[str, DecisionValue] = {}
        for name, spec in specs.items():
            payload = obj.get(name)
            if payload is None:
                continue
            results[name] = _coerce_result(name, spec, payload)
        return results


# ======================================================================
#  统一结果归一化
# ======================================================================
def _coerce_result(name: str, spec: Dict[str, Any], payload: Any) -> DecisionValue:
    """把后端返回的任意结构归一化成对应的 Result 类型。"""
    qtype = spec["type"]
    criteria = spec["criteria"]

    # 后端可能把结果包在 {"result": ...} / {"value": ...} 里
    if isinstance(payload, dict) and len(payload) == 1:
        inner_key = next(iter(payload))
        if inner_key in ("result", "value", "output", "decision") and not isinstance(
            payload[inner_key], dict
        ):
            payload = payload[inner_key]

    if qtype == TYPE_NOUL:
        return _coerce_noul(payload)
    if qtype == TYPE_CHOICE:
        return _coerce_choice(payload, criteria)
    return _coerce_score(payload, criteria)


def _coerce_noul(payload: Any) -> NoulResult:
    prob = 0.0
    if isinstance(payload, dict):
        for k in ("probability", "prob", "p", "yes", "score"):
            if k in payload:
                prob = _clamp01(payload[k])
                break
        else:
            # {"label": "yes"} 形式
            lab = str(payload.get("label") or payload.get("answer") or "").strip().lower()
            prob = 1.0 if lab in ("yes", "true", "是", "1") else 0.0
    elif isinstance(payload, (int, float)):
        prob = _clamp01(payload)
    elif isinstance(payload, str):
        prob = 1.0 if payload.strip().lower() in ("yes", "true", "是", "1") else 0.0
    return NoulResult(probability=prob, label="yes" if prob >= 0.5 else "no", raw=payload)


def _coerce_choice(payload: Any, criteria: Dict[str, str]) -> ChoiceResult:
    options = list(criteria.keys())
    probs: Dict[str, Any] = {}
    if isinstance(payload, dict):
        raw_probs = payload.get("probabilities") or payload.get("probs") or payload.get("scores")
        if isinstance(raw_probs, dict):
            probs = raw_probs
        elif "label" in payload or "choice" in payload or "answer" in payload:
            chosen = str(
                payload.get("label") or payload.get("choice") or payload.get("answer")
            ).strip()
            conf = _clamp01(payload.get("confidence", 0.9), 0.9)
            # 命中选项给 conf，其余均分剩余
            rest = (1.0 - conf) / max(1, len(options) - 1) if len(options) > 1 else 0.0
            probs = {opt: (conf if opt == chosen else rest) for opt in options}
    elif isinstance(payload, str):
        chosen = payload.strip()
        probs = {opt: (1.0 if opt == chosen else 0.0) for opt in options}

    # 只保留合法选项；缺失的补 0
    probs = {opt: probs.get(opt, 0.0) for opt in options}
    dist = _normalize_distribution(probs)
    if dist:
        best = max(dist, key=lambda k: dist[k])
        return ChoiceResult(
            label=best, confidence=dist[best], probabilities=dist, raw=payload
        )
    return ChoiceResult(label=options[0] if options else "", confidence=0.0, raw=payload)


def _coerce_score(payload: Any, criteria: List[str]) -> ScoreResult:
    n = len(criteria)
    probs: Dict[str, float] = {}
    if isinstance(payload, dict):
        raw_probs = payload.get("probabilities") or payload.get("probs") or payload.get("scores")
        if isinstance(raw_probs, dict) and raw_probs:
            probs = {str(k): float(v) for k, v in raw_probs.items()}
        elif "score" in payload or "level" in payload or "index" in payload:
            raw = payload.get("score", payload.get("level", payload.get("index")))
            idx = _score_to_index(raw, criteria)
            kernel = _softmax_like(idx, n)
            probs = {criteria[i]: w for i, w in kernel.items()}
    elif isinstance(payload, (int, float)):
        idx = _score_to_index(payload, criteria)
        probs = {criteria[i]: w for i, w in _softmax_like(idx, n).items()}
    elif isinstance(payload, str):
        idx = _score_to_index(payload, criteria)
        probs = {criteria[i]: w for i, w in _softmax_like(idx, n).items()}

    # 只保留合法档位
    probs = {c: float(probs.get(c, 0.0)) for c in criteria}
    dist = _normalize_distribution(probs)
    expected = 0.0
    for i, c in enumerate(criteria):
        expected += dist.get(c, 0.0) * (i + 1)
    normalized = (expected - 1.0) / (n - 1) if n > 1 else 0.0
    confidence = max(dist.values()) if dist else 0.0
    return ScoreResult(
        score=expected,
        normalized=_clamp01(normalized),
        confidence=confidence,
        probabilities=dist,
        raw=payload,
    )


def _score_to_index(raw: Any, criteria: List[str]) -> int:
    """把后端给的分数/档位名映射成 criteria 的下标。"""
    n = len(criteria)
    # 档位名直接命中
    if isinstance(raw, str):
        s = raw.strip()
        if s in criteria:
            return criteria.index(s)
        # 形如 "3/5"、"3级"、"第3档"
        m = re.search(r"(\d+)", s)
        if m:
            raw = int(m.group(1))
        else:
            return n - 1
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return n - 1
    # 约定：若数值落在 1..n 视为档位序号；若落在 0..1 视为归一化分数
    if 1.0 <= v <= float(n):
        return max(0, min(n - 1, int(round(v)) - 1))
    if 0.0 <= v <= 1.0:
        return max(0, min(n - 1, int(round(v * (n - 1)))))
    # 其它（如 0~10 分）线性映射
    return max(0, min(n - 1, int(round(v / 10.0 * (n - 1)))))


# ======================================================================
#  工厂
# ======================================================================
_BACKEND_ALIASES = {
    "": "auto",
    "none": "none",
    "off": "none",
    "disabled": "none",
    "false": "none",
    "d1": "d1",
    "liquid": "d1",
    "local_d1": "d1",
    "llm": "llm",
    "local": "llm",
    "ollama": "llm",
    "openai": "llm",
    "auto": "auto",
}


def create_decision_provider(
    cfg: Optional[Dict[str, Any]] = None,
    llm: Optional[Any] = None,
    llm_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[BaseDecisionProvider]:
    """按 ``decision`` 配置创建决策 provider。

    配置示例（``config/config.yaml``）::

        decision:
          enabled: true
          backend: "auto"        # d1 | llm | auto | none
          model_name: "LiquidAI/d1-3B"
          device: "auto"
          local_files_only: true

    Args:
        cfg: ``config['decision']`` 子 dict。
        llm: 可选 LLM 实例（``llm`` 后端复用；为空时按 ``llm_cfg`` 现建）。
        llm_cfg: ``config['llm']`` 子 dict —— 用于在未注入 llm 实例时按项目
            已配置的后端（local / ollama / openai）现建。若为空则回落到
            ``cfg['llm']``。

    Returns:
        provider 实例；``enabled`` 为假 / 后端为 none / 创建失败时返回 ``None``。
    """
    cfg = dict(cfg or {})
    if not cfg.get("enabled"):
        return None
    backend = _BACKEND_ALIASES.get(str(cfg.get("backend") or "auto").strip().lower(), "auto")
    if backend == "none":
        return None

    # 后端一：原生 d1
    if backend in ("d1", "auto"):
        try:
            return D1LocalProvider(
                model_name=str(cfg.get("model_name") or "LiquidAI/d1-3B"),
                device=str(cfg.get("device") or "auto"),
                cache_dir=cfg.get("cache_dir"),
                local_files_only=bool(cfg.get("local_files_only", True)),
                dtype=cfg.get("dtype"),
                trust_remote_code=bool(cfg.get("trust_remote_code", True)),
            )
        except Exception as exc:  # noqa: BLE001
            if backend == "d1":
                logger.warning("d1 后端不可用（%s）；决策层将停用。", exc)
                return None
            logger.info("d1 后端不可用（%s），auto 回落到 llm 后端。", exc)

    # 后端二：用现有 LLM 模拟
    if llm is None:
        try:
            from src.llm_provider import create_llm

            llm = create_llm(llm_cfg if llm_cfg is not None else (cfg.get("llm") or {}))
        except Exception as exc:  # noqa: BLE001
            logger.warning("决策层 llm 后端构造失败：%s", exc)
            return None
    try:
        provider = LLMDecisionProvider(
            llm, max_new_tokens=int(cfg.get("max_new_tokens", 256))
        )
        logger.info("决策层已启用：backend=llm（用现有 LLM 模拟 d1 原语）")
        return provider
    except Exception as exc:  # noqa: BLE001
        logger.warning("决策层 llm 后端不可用：%s", exc)
        return None


__all__ = [
    "TYPE_NOUL",
    "TYPE_CHOICE",
    "TYPE_SCORE",
    "SUPPORTED_PRIMITIVES",
    "NoulResult",
    "ChoiceResult",
    "ScoreResult",
    "DecisionValue",
    "BaseDecisionProvider",
    "D1LocalProvider",
    "LLMDecisionProvider",
    "create_decision_provider",
]
