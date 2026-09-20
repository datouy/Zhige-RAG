"""审计日志中间件，记录敏感操作的审计跟踪。

用于满足合规性要求，记录：
- 认证操作（登录、注册、登出）
- 数据访问（文档检索、KG 操作）
- 文件上传/下载
- 配置变更
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.utils import get_logger, resolve_path, ensure_dir

logger = get_logger("audit")


class AuditAction(str, Enum):
    """审计动作枚举。"""
    # 认证相关
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    REGISTER = "register"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    
    # 数据访问相关
    DOCUMENT_LIST = "document_list"
    DOCUMENT_SEARCH = "document_search"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_DOWNLOAD = "document_download"
    CHAT_QUERY = "chat_query"
    
    # 知识图谱相关
    KG_QUERY = "kg_query"
    KG_BUILD = "kg_build"
    KG_EXTRACT = "kg_extract"
    
    # Agent 相关
    AGENT_CHAT = "agent_chat"
    
    # 管理操作
    CONFIG_CHANGE = "config_change"
    USER_DISABLE = "user_disable"
    USER_ENABLE = "user_enable"


class AuditLogger:
    """审计日志记录器。
    
    支持两种存储方式：
    1. 文件存储（JSON Lines 格式）
    2. 数据库存储（通过 UsageLog 表）
    """

    def __init__(
        self,
        audit_file: Optional[str] = None,
        use_db: bool = True,
        use_file: bool = True,
    ):
        """初始化审计日志记录器。
        
        Args:
            audit_file: 审计日志文件路径，默认为 logs/audit.jsonl
            use_db: 是否记录到数据库 UsageLog 表
            use_file: 是否记录到文件
        """
        self.use_db = use_db
        self.use_file = use_file
        
        if use_file:
            self.audit_file = audit_file or "logs/audit.jsonl"
            self._audit_path = resolve_path(self.audit_file)
            ensure_dir(self._audit_path.parent)
        else:
            self._audit_path = None

    def _format_audit_entry(
        self,
        user_id: Optional[str],
        action: AuditAction,
        ip: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        resource: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """格式化审计日志条目。"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action.value if isinstance(action, AuditAction) else action,
            "success": success,
            "ip": ip,
            "resource": resource,
            "details": details or {},
            "error_message": error_message,
        }
        # 移除 None 值
        return {k: v for k, v in entry.items() if v is not None}

    def _write_to_file(self, entry: Dict[str, Any]) -> None:
        """写入审计日志到文件。"""
        if not self._audit_path:
            return
        try:
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("写入审计日志文件失败: %s", e)

    def _write_to_db(
        self,
        db: Session,
        user_id: Optional[str],
        action: AuditAction,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """写入审计日志到数据库。"""
        if not self.use_db:
            return
        try:
            from src.db.models import UsageLog
            
            entry = self._format_audit_entry(
                user_id=user_id,
                action=action,
                success=success,
                details=details,
                error_message=error_message,
            )
            
            log = UsageLog(
                user_id=user_id or "anonymous",
                endpoint=entry["action"],
                method="AUDIT",
                status_code=200 if success else 401,
                error_message=error_message,
                latency_ms=0,
            )
            db.add(log)
            db.commit()
        except Exception as e:
            logger.error("写入审计日志数据库失败: %s", e)

    def log(
        self,
        action: AuditAction,
        user_id: Optional[str] = None,
        ip: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        resource: Optional[str] = None,
        error_message: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> None:
        """记录审计日志。
        
        Args:
            action: 审计动作
            user_id: 用户 ID
            ip: 客户端 IP
            success: 操作是否成功
            details: 额外详细信息
            resource: 操作的资源
            error_message: 错误信息（失败时）
            db: 数据库会话（可选）
        """
        entry = self._format_audit_entry(
            user_id=user_id,
            action=action,
            ip=ip,
            success=success,
            details=details,
            resource=resource,
            error_message=error_message,
        )
        
        # 写入文件
        if self.use_file:
            self._write_to_file(entry)
        
        # 写入数据库
        if self.use_db and db is not None:
            self._write_to_db(db, user_id, action, success, details, error_message)
        
        # 输出到应用日志
        log_level = logger.info if success else logger.warning
        log_level(
            "审计: user=%s action=%s success=%s ip=%s resource=%s",
            user_id,
            action.value if isinstance(action, AuditAction) else action,
            success,
            ip,
            resource,
        )

    def log_auth(
        self,
        user_id: Optional[str],
        action: AuditAction,
        ip: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> None:
        """记录认证操作审计日志。"""
        self.log(
            action=action,
            user_id=user_id,
            ip=ip,
            success=success,
            error_message=error_message,
            db=db,
        )

    def log_data_access(
        self,
        user_id: str,
        resource: str,
        action: AuditAction,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None,
    ) -> None:
        """记录数据访问审计日志。"""
        self.log(
            action=action,
            user_id=user_id,
            success=success,
            resource=resource,
            details=details,
            db=db,
        )

    def log_file_operation(
        self,
        user_id: str,
        action: AuditAction,
        filename: str,
        success: bool = True,
        file_size: Optional[int] = None,
        error_message: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> None:
        """记录文件操作审计日志。"""
        self.log(
            action=action,
            user_id=user_id,
            success=success,
            resource=filename,
            details={"file_size": file_size} if file_size else None,
            error_message=error_message,
            db=db,
        )

    def log_kg_operation(
        self,
        user_id: str,
        action: AuditAction,
        operation: str,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> None:
        """记录知识图谱操作审计日志。"""
        self.log(
            action=action,
            user_id=user_id,
            success=success,
            resource=f"kg:{operation}",
            details=details,
            error_message=error_message,
            db=db,
        )

    def get_recent_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取最近的审计日志。
        
        Args:
            user_id: 按用户过滤（可选）
            action: 按动作过滤（可选）
            limit: 返回条数限制
            
        Returns:
            审计日志条目列表
        """
        if not self._audit_path or not self._audit_path.exists():
            return []
        
        entries = []
        try:
            with open(self._audit_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        # 应用过滤器
                        if user_id and entry.get("user_id") != user_id:
                            continue
                        if action and entry.get("action") != action.value:
                            continue
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error("读取审计日志失败: %s", e)
        
        return entries[-limit:]


# 全局审计日志记录器实例
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """获取全局审计日志记录器实例。"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(
            audit_file=os.getenv("AUDIT_LOG_FILE", "logs/audit.jsonl"),
            use_db=os.getenv("AUDIT_USE_DB", "true").lower() == "true",
            use_file=os.getenv("AUDIT_USE_FILE", "true").lower() == "true",
        )
    return _audit_logger
