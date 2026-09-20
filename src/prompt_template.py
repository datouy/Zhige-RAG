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
#  注意：与 config/prompts.yaml 的行为约束保持一致——核心指标是
#  "可被验证"：宁可拒答，不可编造。这里不使用 {var} 运行时占位
# （兜底场景拿不到运行时变量），变量注入由 ContextAssembler 完成。
# ----------------------------------------------------------------------
DEFAULT_PROMPTS: Dict[str, str] = {
    "system": (
        "你是一个专业、严谨的中文知识库助手，名字叫'小识'。\n"
        "你的职责是：\n"
        "1. 仅依据'已知上下文'回答用户的问题，不要使用任何先验知识或外部信息。\n"
        "2. 如果上下文不足以回答问题，请明确回答'根据当前知识库，我无法回答这个问题'，"
        "并给出可操作的建议（例如换个问法、上传更多资料）。宁可拒答，不可编造。\n"
        "3. 回答时必须标注信息来源，使用形如 [1]、[2] 的引用编号，并在末尾列出对应来源。\n"
        "4. 保持简洁、结构化（必要时使用编号列表），使用简体中文。\n"
        "5. 禁止编造事实、禁止输出有害、违法、歧视性内容。"
    ),
    "user_template": (
        "【已知上下文】\n{context}\n\n"
        "【用户问题】\n{question}\n\n"
        "【回答要求】\n"
        "- 只使用'已知上下文'中的信息作答。\n"
        "- 在引用具体信息时使用编号角标，例如：'RAG 的核心思想是检索增强生成 [1]。'\n"
        "- 若上下文与问题无关或不足以回答，请直接说明无法回答。\n"
        "- 末尾以'参考资料'列出对应的来源编号与标题（不要重复正文中的内容）。"
    ),
    "cite_block": "[{index}] {title}（来源：{source}，第 {page} 页）",
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