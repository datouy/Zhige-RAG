"""API 路由包。

每个子模块暴露一个 ``router``（``fastapi.APIRouter``）实例；``api/main.py``
统一 ``include_router`` 并注入 ``Depends(get_current_user)`` 依赖。

模块清单
--------
- ``auth``         : 认证路由（注册 / 登录 / 刷新 / 当前用户 / 登出）
- ``subscription`` : 订阅 / 套餐路由
- ``system``       : 健康 / 就绪 / 指标 / 配置（部分需认证）
- ``chat``         : 检索 / 同步问答 / 入库 / WebSocket 流式问答
- ``kg``           : 知识图谱 / GraphRAG（含 cypher 注入防护）
- ``agent``        : ReAct Agent 路由
- ``eval``         : 评估端点
- ``feedback``     : 反馈层（用户反馈 / 统计 / 差评导出评估集 / 长期记忆管理）
- ``kb``           : 知识库（多库管理）与会话（列表 / 重命名 / 历史消息）
- ``settings``     : 模型设置（界面上切换 local / ollama / openai，即时生效）
"""
from .agent import router as agent_router
from .auth import router as auth_router
from .chat import router as chat_router
from .eval import router as eval_router
from .feedback import router as feedback_router
from .kb import router as kb_router
from .kg import router as kg_router
from .settings import router as settings_router
from .subscription import router as subscription_router
from .system import router as system_router

__all__ = [
    "agent_router",
    "auth_router",
    "chat_router",
    "eval_router",
    "feedback_router",
    "kb_router",
    "kg_router",
    "settings_router",
    "subscription_router",
    "system_router",
]