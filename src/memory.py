"""上下文与记忆：即时记忆 / 检索记忆 / 长期记忆。

设计要点（对应架构要求"区分即时、检索、长期三类上下文；关键信息不写死
在 prompt，也不依赖模型训练"）：

- **即时记忆** :class:`SessionMemory`
  当前会话最近 N 轮对话（进程内，TTL 过期），为多轮问答提供指代消解背景。
- **检索记忆**（不在此实现）
  每次问答由向量库实时召回，见 ``RAGPipeline._retrieve``——它是"事实"
  的唯一来源，会话间不持久。
- **长期记忆** :class:`LongTermMemoryStore`
  用户级键值事实（SQLite，``long_term_memories`` 表），由用户显式指令
  （"记住……"）或反馈层沉淀写入；检索时**结构化注入 system**，而非混入向量。
- :class:`ContextAssembler`
  把三类上下文 + 运行时变量（助手名、企业名、当前日期、用户身份）组装成
  prompt 消息——模板来自 ``config/prompts.yaml``，运行时变量**注入**而非
  写死，企业换名/换人不需要改代码或重新训练。

线程安全：``SessionMemory`` 进程内共享（FastAPI 线程池并发访问），全程持锁。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

from .utils import get_logger

logger = get_logger("memory")

# "记住……" 显式指令（长期记忆写入规则，无 LLM 依赖）
_REMEMBER_RE = re.compile(r"记住[：:，,]?\s*(.{2,200})")
# "我的 X 是 Y" / "我们的 X 是 Y" 自我介绍类事实
_PROFILE_RE = re.compile(r"(?:我|我们|公司|团队)的\s*([^\s，,。；;：:]{1,12})\s*(?:是|为)\s*([^\s，,。；;]{1,100})")
# 指代词：命中时触发检索查询改写
_DEICTIC_RE = re.compile(r"(它|他|她|他们|它们|这个|那个|这些|那些|刚才|上面|上述|之前|继续|还有呢|为什么)")


# ----------------------------------------------------------------------
#  即时记忆（会话窗口）
# ----------------------------------------------------------------------
@dataclass
class Turn:
    """一轮对话。"""

    role: str  # user / assistant
    content: str
    ts: float = field(default_factory=time.time)


class SessionMemory:
    """即时记忆：按 (user_id, session_id) 保存最近 N 轮对话，带 TTL。

    只保留窗口内的内容，随会话结束/过期消失——与长期记忆严格区分。
    """

    def __init__(self, max_turns: int = 3, ttl_minutes: float = 30.0, max_sessions: int = 2000) -> None:
        self.max_turns = max(0, int(max_turns))
        self.ttl_seconds = max(1.0, float(ttl_minutes) * 60.0)
        self.max_sessions = max_sessions
        self._sessions: Dict[str, List[Turn]] = {}
        self._lock = RLock()

    @staticmethod
    def _key(user_id: str, session_id: str) -> str:
        return f"{user_id}::{session_id or 'default'}"

    def append(self, user_id: str, session_id: str, role: str, content: str) -> None:
        if self.max_turns <= 0 or not content:
            return
        key = self._key(user_id, session_id)
        now = time.time()
        with self._lock:
            self._evict_expired_locked(now)
            turns = self._sessions.setdefault(key, [])
            turns.append(Turn(role=role, content=str(content)[:2000], ts=now))
            # 窗口裁剪：保留最近 max_turns*2 条（user+assistant 各算一条）
            if len(turns) > self.max_turns * 2:
                del turns[: len(turns) - self.max_turns * 2]

    def get_history(self, user_id: str, session_id: str) -> List[Dict[str, str]]:
        """取当前会话窗口内的历史（OpenAI 消息格式），过期即空。"""
        if self.max_turns <= 0:
            return []
        key = self._key(user_id, session_id)
        now = time.time()
        with self._lock:
            self._evict_expired_locked(now)
            turns = self._sessions.get(key) or []
            return [{"role": t.role, "content": t.content} for t in turns if now - t.ts <= self.ttl_seconds]

    def clear(self, user_id: str, session_id: str = "") -> int:
        """清空会话（session_id 为空清该用户全部会话），返回清除条数。"""
        with self._lock:
            if session_id:
                turns = self._sessions.pop(self._key(user_id, session_id), [])
                return len(turns)
            keys = [k for k in self._sessions if k.startswith(f"{user_id}::")]
            n = sum(len(self._sessions.pop(k)) for k in keys)
            return n

    def _evict_expired_locked(self, now: float) -> None:
        """淘汰过期会话（须持锁调用）。"""
        expired = [k for k, turns in self._sessions.items() if not turns or now - turns[-1].ts > self.ttl_seconds]
        for k in expired:
            del self._sessions[k]
        # 硬上限：防极端场景内存膨胀
        while len(self._sessions) > self.max_sessions:
            oldest = min(self._sessions, key=lambda k: self._sessions[k][-1].ts if self._sessions[k] else 0)
            del self._sessions[oldest]


# ----------------------------------------------------------------------
#  长期记忆（SQLite）
# ----------------------------------------------------------------------
class LongTermMemoryStore:
    """长期记忆存储：``long_term_memories`` 表的增删查 + 显式指令抽取。"""

    def __init__(self, session_factory=None) -> None:
        # 延迟导入避免循环依赖；测试可注入内存 session factory
        if session_factory is None:
            from .db.database import SessionLocal

            session_factory = SessionLocal
        self._session_factory = session_factory

    def remember(self, user_id: str, key: str, value: str, source: str = "user") -> bool:
        """写入/更新一条长期记忆（按 user+key upsert）。"""
        key = (key or "").strip()[:200]
        value = (value or "").strip()[:2000]
        if not user_id or not key or not value:
            return False
        from .db.models import LongTermMemory
        from datetime import datetime

        db = self._session_factory()
        try:
            row = (
                db.query(LongTermMemory)
                .filter(LongTermMemory.user_id == user_id, LongTermMemory.mem_key == key)
                .first()
            )
            if row is None:
                db.add(LongTermMemory(user_id=user_id, mem_key=key, mem_value=value, source=source))
            else:
                row.mem_value = value
                row.source = source
                row.updated_at = datetime.utcnow()
            db.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            logger.error("写入长期记忆失败: %s", exc)
            return False
        finally:
            db.close()

    def recall(self, user_id: str, limit: int = 20) -> List[Dict[str, str]]:
        """取用户全部长期记忆（注入 system 用，量小直接全取后截断）。"""
        if not user_id:
            return []
        from .db.models import LongTermMemory

        db = self._session_factory()
        try:
            rows = (
                db.query(LongTermMemory)
                .filter(LongTermMemory.user_id == user_id)
                .order_by(LongTermMemory.updated_at.desc())
                .limit(max(1, int(limit)))
                .all()
            )
            return [{"key": r.mem_key, "value": r.mem_value} for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.error("读取长期记忆失败: %s", exc)
            return []
        finally:
            db.close()

    def forget(self, user_id: str, key: str) -> bool:
        """删除指定长期记忆。"""
        from .db.models import LongTermMemory

        db = self._session_factory()
        try:
            n = (
                db.query(LongTermMemory)
                .filter(LongTermMemory.user_id == user_id, LongTermMemory.mem_key == key)
                .delete()
            )
            db.commit()
            return n > 0
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            logger.error("删除长期记忆失败: %s", exc)
            return False
        finally:
            db.close()

    # ------------------------------------------------------------------
    @staticmethod
    def extract_from_text(text: str) -> List[Tuple[str, str]]:
        """从用户消息中抽取"可记忆"事实（规则式，返回 (key, value) 列表）。

        支持两类显式指令：
        - ``记住：<内容>``            → key="用户叮嘱", value=内容
        - ``我的X是Y`` / ``公司的X是Y`` → key=X, value=Y
        """
        results: List[Tuple[str, str]] = []
        text = (text or "").strip()
        if not text:
            return results
        for m in _REMEMBER_RE.finditer(text):
            value = m.group(1).strip().rstrip("。.！!？?")
            if value:
                results.append(("用户叮嘱", value))
        for m in _PROFILE_RE.finditer(text):
            key, value = m.group(1).strip(), m.group(2).strip()
            if key and value:
                results.append((key, value))
        return results

    def maybe_remember_from_message(self, user_id: str, message: str) -> List[Tuple[str, str]]:
        """检测用户消息中的记忆指令并落库；返回实际写入的 (key, value)。"""
        facts = self.extract_from_text(message)
        written: List[Tuple[str, str]] = []
        for key, value in facts:
            if self.remember(user_id, key, value, source="user"):
                written.append((key, value))
        if written:
            logger.info("已写入长期记忆 %d 条（user=%s）: %s", len(written), user_id, written)
        return written


# ----------------------------------------------------------------------
#  检索查询改写（多轮指代消解）
# ----------------------------------------------------------------------
def rewrite_query_for_retrieval(query: str, history: List[Dict[str, str]]) -> str:
    """把多轮背景注入检索查询：命中指代词且存在历史时，拼接上一轮用户问题。

    向量检索只看当前 query，"它怎么报销？"脱离上文将无法召回。
    改写只影响**检索**，不改变用户原始问题在生成阶段的呈现。
    """
    query = (query or "").strip()
    if not query or not history:
        return query
    if not _DEICTIC_RE.search(query):
        return query
    last_user = next((h["content"] for h in reversed(history) if h.get("role") == "user"), "")
    if not last_user or last_user == query:
        return query
    rewritten = f"{last_user[:80]}；{query}"
    logger.debug("检索查询改写：%r -> %r", query, rewritten)
    return rewritten


# ----------------------------------------------------------------------
#  上下文组装器
# ----------------------------------------------------------------------
class ContextAssembler:
    """把三类上下文 + 运行时变量组装为 prompt 消息。

    ``runtime`` 支持的变量（在 prompts.yaml 的 system/user 模板中以
    ``{var}`` 占位，缺失时静默保留原样，绝不因未知 key 抛错）：
    - ``assistant_name``：助手名（默认"小识"）
    - ``org_name``      ：企业/组织名
    - ``current_date``  ：当前日期（防"现在几号"类问题依赖模型训练数据）
    - ``user_name``     ：当前用户显示名
    """

    def __init__(self, prompts: Any, memory_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.prompts = prompts
        self.memory_cfg = memory_cfg or {}
        self.enabled = bool(self.memory_cfg.get("enabled", True))
        self.max_memory_items = int(self.memory_cfg.get("long_term_inject_limit", 10))

    def _render(self, template: str, runtime: Dict[str, str]) -> str:
        """安全的模板渲染：逐个替换已知变量，其余 ``{}`` 原样保留。"""
        out = template
        for k, v in (runtime or {}).items():
            out = out.replace("{" + k + "}", str(v))
        return out

    def _long_term_block(self, memories: List[Dict[str, str]]) -> str:
        """长期记忆 → 结构化文本块（写明"仅供个性化参考，回答仍须以上下文为准"）。"""
        if not memories:
            return ""
        lines = [f"- {m['key']}：{m['value']}" for m in memories[: self.max_memory_items]]
        return (
            "【用户长期记忆】（仅用于个性化称呼与语境理解，回答事实仍只能依据'已知上下文'）\n"
            + "\n".join(lines)
        )

    def build(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        long_term: Optional[List[Dict[str, str]]] = None,
        runtime: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """组装最终消息序列：system(运行时变量+长期记忆) → 历史 → 本轮。"""
        runtime = runtime or {}
        system = self._render(self.prompts.system(), runtime)

        lt_block = self._long_term_block(long_term or []) if self.enabled else ""
        if lt_block:
            system = f"{system}\n\n{lt_block}"

        if not context_chunks:
            user_content = self.prompts.prompts.get("empty_context", "").strip()
            user_content = self._render(user_content, runtime)
            final_user = f"{user_content}\n\n问题：{question}"
        else:
            ctx_lines = []
            tpl = self.prompts.prompts.get("context_chunk", "[{index}] {title}\n{content}")
            for c in context_chunks:
                idx = c.get("index", 0)
                title = c.get("title") or c.get("source") or f"片段{idx}"
                ctx_lines.append(tpl.format(index=idx, title=title, content=c.get("content", "")))
            context = "\n\n".join(ctx_lines)
            user_tpl = self.prompts.prompts.get(
                "user_template", "【已知上下文】\n{context}\n\n【用户问题】\n{question}"
            )
            user_content = self._render(user_tpl, runtime).replace("{context}", context).replace("{question}", question)
            final_user = user_content

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        # 即时记忆：历史轮次（最多保留窗口，超长内容截断）
        if self.enabled and history:
            for h in history[-(int(self.memory_cfg.get("session_turns", 3)) * 2):]:
                messages.append({"role": h.get("role", "user"), "content": str(h.get("content", ""))[:1500]})
        messages.append({"role": "user", "content": final_user})
        return messages


# ----------------------------------------------------------------------
#  进程级即时记忆单例
# ----------------------------------------------------------------------
# 全局共享：同一 (user, session) 无论走同步 / WS / 哪个线程，共享同一窗口。
# 供 api/routes/chat.py 与 ui 直接导入使用。
SESSION_MEMORY = SessionMemory(max_turns=3, ttl_minutes=30.0)
