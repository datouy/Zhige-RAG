"""中文 RAG Prompt 模板加载与格式化。"""

from __future__ import annotations

from typing import Any, Dict, List

from .utils import get_logger, resolve_path

logger = get_logger("prompt_template")


class PromptTemplate:
    """从 YAML 文件加载的 Prompt 模板集合。"""

    def __init__(self, prompts: Dict[str, str]):
        self.prompts = prompts

    @classmethod
    def from_yaml(cls, path: str = "config/prompts.yaml") -> "PromptTemplate":
        import yaml  # type: ignore

        p = resolve_path(path)
        if not p.exists():
            logger.warning("Prompt 配置文件不存在: %s，使用默认", p)
            return cls(DEFAULT_PROMPTS)
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # 合并默认值，确保所有 key 存在
        merged = {**DEFAULT_PROMPTS, **data}
        return cls(merged)

    # ------------------------------------------------------------------
    def system(self) -> str:
        return self.prompts.get("system", DEFAULT_PROMPTS["system"]).strip()

    def build_messages(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """组装对话消息。

        Args:
            question: 用户问题。
            context_chunks: 每个 chunk 至少含 title、content、index。

        Returns:
            OpenAI 风格消息列表。
        """
        if not context_chunks:
            user_content = self.prompts.get("empty_context", DEFAULT_PROMPTS["empty_context"]).strip()
            return [
                {"role": "system", "content": self.system()},
                {"role": "user", "content": f"{user_content}\n\n问题：{question}"},
            ]

        ctx_lines = []
        for c in context_chunks:
            idx = c.get("index", 0)
            title = c.get("title") or c.get("source") or f"片段{idx}"
            content = c.get("content", "")
            template = self.prompts.get("context_chunk", DEFAULT_PROMPTS["context_chunk"])
            ctx_lines.append(template.format(index=idx, title=title, content=content))
        context = "\n\n".join(ctx_lines)

        user_template = self.prompts.get("user_template", DEFAULT_PROMPTS["user_template"])
        user_content = user_template.format(context=context, question=question)
        return [
            {"role": "system", "content": self.system()},
            {"role": "user", "content": user_content},
        ]

    def format_citation(self, index: int, title: str, source: str, page: Any) -> str:
        tpl = self.prompts.get("cite_block", DEFAULT_PROMPTS["cite_block"])
        return tpl.format(index=index, title=title, source=source, page=page)

    def eval_judge_prompt(self, reference: str, prediction: str) -> str:
        tpl = self.prompts.get("eval_judge", DEFAULT_PROMPTS["eval_judge"])
        return tpl.format(reference=reference, prediction=prediction)


# ----------------------------------------------------------------------
#  默认 Prompt（未配置时的兜底）
#  与 config/prompts.yaml 保持同一取向：材料是依据、不是抄写对象；
#  引用标在关键处而非每句；提炼与归因是任务，不是越界。
#  防编造的护栏收敛到"不得引入材料里没有的事实/数字/人名/结论"。
#  这里不使用 {var} 运行时占位（兜底场景拿不到运行时变量），
#  变量注入由 ContextAssembler 完成。
# ----------------------------------------------------------------------
DEFAULT_PROMPTS: Dict[str, str] = {
    "system": (
        "你是一个专业的中文知识库助手，名字叫'小识'。\n"
        "用户把资料交给你阅读后提问，目标是**把材料讲清楚**，不是把材料摘一遍。\n"
        "【准则】1. 材料是唯一依据：不引入材料里没有的事实、数字、人名、结论。"
        "2. 材料是理解对象，不是抄写对象；读懂后用自己的话重组，把原句一条条摘下来编号排列不算完成任务。"
        "3. 数字、日期、金额如实采用，可归类对比，不可改动数值。"
        "4. 分清'材料作者的判断''文中人物说的话''引述的第三方报告'，归属写清楚。\n"
        "【结构】开头一句总判断（禁止'以…为背景''本文主要讲述了'这类目录式开头）；"
        "中间 2~4 层，每层一个论点 + 材料中的事实作证，层间要推进（动因→机制→后果，或 表象→实质→代价），"
        "不要写成并列条目；有前后对比、因果错位、受益方与受损方不一致时指出来；结尾一句收束并回扣开头。\n"
        "【引用】关键数据与关键论断后标 [n]，不必每句都标；概括与因果分析不需要引用。"
        "确实无法回答时才说'资料无法确定'，不要用免责声明收尾。"
    ),
    "user_template": (
        "【资料】\n{context}\n\n"
        "【问题】\n{question}\n\n"
        "【作答要求】\n"
        "- 先判断这些资料整体在讲什么、想论证什么，再按'总判断 → 分层论证 → 收束'组织回答。\n"
        "- 把资料里的具体数字、时间、金额作为论据用起来。\n"
        "- 不要逐句照抄，也不要在结尾附加免责说明。"
    ),
    "cite_block": "[{index}] {title}（来源：{source}，第 {page}）",
    "context_chunk": "[{index}] {title}\n{content}",
    "empty_context": (
        "当前知识库中未检索到与问题相关的内容。请尝试：\n"
        "- 换个问法或使用更具体的关键词；\n"
        "- 在'上传与管理'页面添加相关文档；\n"
        "- 调高检索 Top-K 以扩大召回。"
    ),
    "eval_judge": (
        "你是一个严格的评分员。给定'参考答案'和'模型回答'，判断模型回答是否在事实层面与参考答案一致。\n"
        "- 输出格式：第一行写 'YES' 或 'NO'，第二行写一句不超过 30 字的理由。\n"
        "- 参考答案：\n{reference}\n- 模型回答：\n{prediction}"
    ),
}

