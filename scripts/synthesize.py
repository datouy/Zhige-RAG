"""数据合成 pipeline：从 markdown 抽取实体/概念，生成 Easy+Medium 评估题。

设计约束：
- 纯文本处理，不初始化向量库。
- 复用 ``src.utils`` / ``src.document_loader``。
- LLM 调用串行，失败时跳过 section 并记录警告。
- 全局去重、按 max-questions 截断、可选 dry-run/append。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.document_loader import Document, _load_markdown as load_markdown
from src.utils import apply_env_overrides, ensure_dir, get_logger, load_config, resolve_path

logger = get_logger("synthesize")

# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractedKnowledge:
    entities: List[str]
    definition: str
    key_facts: List[str]


@dataclass(frozen=True)
class QuestionRecord:
    question: str
    expected_sources: List[str]
    expected_keywords: List[str]
    difficulty: str
    source_section: str
    template_id: str
    generation_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "expected_sources": list(self.expected_sources),
            "expected_keywords": list(self.expected_keywords),
            "difficulty": self.difficulty,
            "source_section": self.source_section,
            "template_id": self.template_id,
            "generation_note": self.generation_note,
        }


# ----------------------------------------------------------------------
# LLM 客户端
# ----------------------------------------------------------------------
class _LLMClient:
    """轻量 LLM 客户端，适配本地 / OpenAI-compatible / Ollama。"""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        llm_cfg = cfg.get("llm") or {}
        backend = (cfg.get("synthesize") or {}).get("default_backend", "local")
        self.backend = backend
        self.model_name = llm_cfg.get("model_name", "")
        self.openai_api_base = llm_cfg.get("openai_api_base", "http://localhost:11434/v1")
        self._client = None
        self._local_llm = None

        if backend == "openai":
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(
                    api_key=llm_cfg.get("openai_api_key") or "sk-placeholder",
                    base_url=self.openai_api_base,
                )
            except Exception as exc:
                logger.warning("OpenAI 客户端初始化失败，回退到本地模式: %s", exc)
                self.backend = "local"

        if backend == "local":
            self._load_local(llm_cfg)

    def _load_local(self, llm_cfg: Dict[str, Any]) -> None:
        try:
            from src.llm import LocalLLM

            self._local_llm = LocalLLM(
                model_name=self.model_name,
                device=llm_cfg.get("device", "auto"),
                device_map=llm_cfg.get("device_map", "auto"),
                torch_dtype=llm_cfg.get("torch_dtype", "auto"),
                quant=llm_cfg.get("quantization"),
                cache_dir=llm_cfg.get("cache_dir"),
                generation=llm_cfg.get("generation"),
                chat_template=llm_cfg.get("chat_template", "auto"),
                local_files_only=llm_cfg.get("local_files_only", False),
                trust_remote_code=llm_cfg.get("trust_remote_code", False),
            )
        except Exception as exc:
            logger.error("本地 LLM 初始化失败: %s", exc)
            raise RuntimeError(f"本地 LLM 初始化失败: {exc}") from exc

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 256) -> str:
        if self.backend == "openai" and self._client:
            try:
                completion = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = completion.choices[0].message.content or ""
                return text.strip()
            except Exception as exc:
                logger.warning("OpenAI 接口调用失败: %s", exc)
                if self._local_llm:
                    return self._local_llm.chat(
                        [{"role": "user", "content": prompt}],
                        generation=None,
                        stream=False,
                    ).strip()
                raise
        if self._local_llm:
            return self._local_llm.chat(
                [{"role": "user", "content": prompt}],
                generation=None,
                stream=False,
            ).strip()
        raise RuntimeError("LLM 客户端未就绪")


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------
_EXTRACT_PROMPT = (
    "你是中文知识抽取助手。给定一段文档内容，输出精炼 JSON。\n"
    "要求：\n"
    "1. entities: 2~4 个核心概念/实体（中文）。\n"
    "2. definition: 一句话定义核心概念。\n"
    "3. key_facts: 3~5 条关键事实，均为中文短语。\n"
    "若内容过短或无信息，返回 {{\"entities\":[],\"definition\":\"\",\"key_facts\":[]}}。\n"
    "只输出 JSON，不要解释。\n\n"
    "内容：\n"
    "{section_text}\n"
)


# ----------------------------------------------------------------------
# 核心处理
# ----------------------------------------------------------------------
def _split_sections(document: Document) -> List[Tuple[str, str]]:
    """按 ``## `` 切分章节。"""
    sections: List[Tuple[str, str]] = []
    cur_title = document.metadata.get("title") or "正文"
    cur_buf: List[str] = []
    text = document.content or ""
    for line in text.splitlines():
        if re.match(r"^##\s+", line):
            if cur_buf:
                sections.append((cur_title, "\n".join(cur_buf).strip()))
            cur_title = re.sub(r"^##\s*", "", line).strip() or "未命名章节"
            cur_buf = []
        else:
            cur_buf.append(line)
    if cur_buf:
        sections.append((cur_title, "\n".join(cur_buf).strip()))
    if not sections:
        sections = [(cur_title, text.strip())]
    return sections


def _safe_parse_extraction(text: str) -> ExtractedKnowledge:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except Exception:
        return ExtractedKnowledge(entities=[], definition="", key_facts=[])
    entities = [str(x).strip() for x in (data.get("entities") or []) if str(x).strip()]
    definition = str(data.get("definition") or "").strip()
    key_facts = [str(x).strip() for x in (data.get("key_facts") or []) if str(x).strip()]
    return ExtractedKnowledge(entities=entities, definition=definition, key_facts=key_facts)


def _extract_knowledge(llm: _LLMClient, section_text: str) -> Optional[ExtractedKnowledge]:
    if not (section_text or "").strip():
        return None
    prompt = _EXTRACT_PROMPT.replace("{section_text}", section_text[:1200])
    try:
        raw = llm.generate(prompt, temperature=0.1, max_tokens=256)
        return _safe_parse_extraction(raw)
    except Exception as exc:
        logger.warning("提取实体失败，跳过该 section: %s", exc)
        return None


_EASY_TEMPLATES = [
    ("tmpl_1", "{entity} 是什么？"),
    ("tmpl_2", "{entity} 有哪些特点？"),
    ("tmpl_3", "请简述 {entity} 的作用。"),
    ("tmpl_4", "{entity} 的核心原理是什么？"),
]

_MEDIUM_TEMPLATES = [
    ("tmpl_5", "{entity1} 和 {entity2} 有什么区别？"),
    ("tmpl_6", "相比 {entity2}，{entity1} 有什么优势？"),
    ("tmpl_7", "在 {section_title} 场景下，{entity} 是如何工作的？"),
    ("tmpl_8", "如果 {entity} 失效，可能的原因有哪些？"),
    ("tmpl_9", "{entity} 的核心要点是什么？"),
    ("tmpl_10", "{entity} 在 {section_title} 中的地位/作用是什么？"),
]


def _build_questions(
    knowledge: ExtractedKnowledge,
    section_title: str,
    sources: List[str],
) -> List[QuestionRecord]:
    questions: List[QuestionRecord] = []
    entities = knowledge.entities[:3] or []
    if not entities:
        # fallback
        note = "section 无实体，使用标题 fallback"
        questions.append(
            QuestionRecord(
                question=f"{section_title} 的核心内容是什么？",
                expected_sources=sources,
                expected_keywords=knowledge.key_facts[:3],
                difficulty="medium",
                source_section=section_title,
                template_id="fallback_section_title",
                generation_note=note,
            )
        )
        return questions

    easy_templates = random.sample(_EASY_TEMPLATES, k=len(_EASY_TEMPLATES))
    medium_templates = random.sample(_MEDIUM_TEMPLATES, k=len(_MEDIUM_TEMPLATES))

    easy_count = random.randint(1, min(3, len(easy_templates), max(1, len(entities))))
    medium_count = random.randint(0, min(2, len(medium_templates)))

    for idx in range(min(easy_count, len(easy_templates))):
        tmpl_id, tmpl = easy_templates[idx]
        entity = entities[idx % len(entities)]
        questions.append(
            QuestionRecord(
                question=tmpl.format(entity=entity, section_title=section_title),
                expected_sources=sources,
                expected_keywords=knowledge.key_facts[:3],
                difficulty="easy",
                source_section=section_title,
                template_id=tmpl_id,
                generation_note=f"基于 entity[{entity}] + section[{section_title}] 生成",
            )
        )

    if medium_count >= 1:
        e1, e2 = entities[0], entities[-1]
        tmpl_id, tmpl = medium_templates[0]
        questions.append(
            QuestionRecord(
                question=tmpl.format(entity1=e1, entity2=e2, section_title=section_title, entity=e1),
                expected_sources=sources,
                expected_keywords=knowledge.key_facts[:3],
                difficulty="medium",
                source_section=section_title,
                template_id=tmpl_id,
                generation_note=f"基于 entity[{e1}, {e2}] + section[{section_title}] 生成",
            )
        )

    if medium_count >= 2:
        entity = entities[0]
        tmpl_id, tmpl = medium_templates[1]
        questions.append(
            QuestionRecord(
                question=tmpl.format(entity1=entity, entity2="传统微调", section_title=section_title, entity=entity),
                expected_sources=sources,
                expected_keywords=knowledge.key_facts[:3],
                difficulty="medium",
                source_section=section_title,
                template_id=tmpl_id,
                generation_note=f"基于 entity[{entity}] + section[{section_title}] 生成",
            )
        )

    return questions


def _write_jsonl(path: Path, records: Sequence[QuestionRecord], append: bool = False) -> None:
    ensure_dir(path.parent)
    mode = "a" if append and path.exists() else "w"
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def _load_existing_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="中文 RAG 评估集数据合成")
    p.add_argument("--input", default="data/raw", help="输入 markdown 文件或目录")
    p.add_argument("--output", default="data/eval/eval_set.jsonl", help="输出 JSONL 路径")
    p.add_argument("--max-questions", type=int, default=50, help="最大题目数量")
    p.add_argument("--llm-backend", default="local", choices=["local", "openai", "ollama"], help="LLM 后端")
    p.add_argument("--model", default=None, help="覆盖 config 中的模型名")
    p.add_argument("--difficulty", default="all", choices=["easy", "medium", "all"], help="难度过滤")
    p.add_argument("--dry-run", action="store_true", help="只打印题目，不写入文件")
    p.add_argument("--append", action="store_true", help="追加到现有 JSONL（默认覆盖）")
    p.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    cfg = load_config(args.config)
    cfg = apply_env_overrides(cfg)

    if args.llm_backend != "local":
        llm_cfg = cfg.get("llm") or {}
        llm_cfg["backend"] = args.llm_backend
        cfg["llm"] = llm_cfg
    if args.model:
        cfg.setdefault("llm", {})["model_name"] = args.model

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    if input_path.is_file():
        md_files = [input_path]
    elif input_path.is_dir():
        md_files = sorted([p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".markdown"}])
    else:
        logger.error("输入路径不存在: %s", input_path)
        return 1

    if not md_files:
        logger.error("未发现 markdown 文件: %s", input_path)
        return 1

    llm = _LLMClient(cfg)

    seen_hashes = set()
    if not args.dry_run and args.append:
        for item in _load_existing_questions(output_path):
            q = item.get("question")
            if not q:
                continue
            seen_hashes.add(hashlib.md5(q.encode("utf-8")).hexdigest())

    collected: List[QuestionRecord] = []
    docs_count = 0
    sections_count = 0
    skipped_sections = 0

    for md_path in md_files:
        documents = load_markdown(md_path)
        if not documents:
            logger.warning("加载为空，已跳过: %s", md_path)
            continue
        docs_count += len(documents)
        for document in documents:
            source_name = str(document.metadata.get("source") or md_path.name)
            for section_title, section_text in _split_sections(document):
                sections_count += 1
                knowledge = _extract_knowledge(llm, section_text)
                if knowledge is None:
                    skipped_sections += 1
                    continue
                candidates = _build_questions(knowledge, section_title, [source_name])
                for rec in candidates:
                    if args.difficulty != "all" and rec.difficulty != args.difficulty:
                        continue
                    q_hash = hashlib.md5(rec.question.encode("utf-8")).hexdigest()
                    if q_hash in seen_hashes:
                        continue
                    seen_hashes.add(q_hash)
                    collected.append(rec)
                    if len(collected) >= args.max_questions:
                        break
                if len(collected) >= args.max_questions:
                    break
                time.sleep(0.5)
            if len(collected) >= args.max_questions:
                break
        if len(collected) >= args.max_questions:
            break

    collected = collected[: args.max_questions]

    if args.dry_run:
        print(f"[dry-run] 将生成 {len(collected)} 条题目：\n")
        for rec in collected:
            print(json.dumps(rec.to_dict(), ensure_ascii=False))
    else:
        _write_jsonl(output_path, list(collected), append=args.append)
        print(f"已写入 {len(collected)} 条题目 -> {output_path}")

    print(f"\n统计：文档={docs_count} 章节={sections_count} 跳过={skipped_sections} 产出={len(collected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
