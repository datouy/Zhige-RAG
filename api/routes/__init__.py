"""API 路由包。

每个子模块暴露一个 ``router``（``fastapi.APIRouter``）实例；``api/main.py``
统一 ``include_router``。各端点用 ``Depends(get_current_user)`` 拿当前用户
身份（本地单用户，无需登录）。

模块清单
--------
- ``system``       : 健康 / 就绪 / 指标 / 配置
- ``chat``         : 检索 / 同步问答 / 入库 / WebSocket 流式问答
- ``kg``           : 知识图谱 / GraphRAG（含 cypher 注入防护）
- ``agent``        : ReAct Agent 路由
- ``saved_agents`` : 保存 Agent（知识范围 / 说明 / 工具权限）
- ``mcp``          : 本机 MCP stdio 服务配置与工具发现
- ``eval``         : 评估端点
- ``feedback``     : 反馈层（用户反馈 / 统计 / 差评导出评估集 / 长期记忆管理）
- ``kb``           : 知识库（多库管理）与会话（列表 / 重命名 / 历史消息）
- ``settings``     : 模型设置（界面上切换 local / ollama / openai，即时生效）
- ``team``         : 可选团队认证、成员角色、知识库共享与 MCP 审批
- ``onboarding``   : 首次体验与示例知识库
- ``admin``        : 质量治理、审计、评估和工作区备份恢复
"""
from .admin import router as admin_router
from .agent import router as agent_router
from .chat import router as chat_router
from .eval import router as eval_router
from .feedback import router as feedback_router
from .kb import router as kb_router
from .kg import router as kg_router
from .mcp import router as mcp_router
from .onboarding import router as onboarding_router
from .saved_agents import router as saved_agents_router
from .settings import router as settings_router
from .system import router as system_router
from .team import router as team_router

__all__ = [
    "admin_router",
    "agent_router",
    "chat_router",
    "eval_router",
    "feedback_router",
    "kb_router",
    "kg_router",
    "mcp_router",
    "onboarding_router",
    "saved_agents_router",
    "settings_router",
    "system_router",
    "team_router",
]
