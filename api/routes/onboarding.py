"""首次体验：可重复载入的可信问答示例库。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from api.deps import get_runtime_config, get_splitter
from src.data_quality import run_data_pipeline
from src.db.database import get_db
from src.db.models import DocumentRecord, KnowledgeBase, User
from src.document_loader import Document
from src.factories import TenantAwareFactory
from src.kb_service import KBError, create_kb, list_accessible_kbs
from src.middleware.identity import get_current_user
from src.team_auth import require_editor
from src.utils import ensure_dir, get_logger, resolve_path

logger = get_logger("api.routes.onboarding")
router = APIRouter()

DEMO_KB_NAME = "可信问答示例"
DEMO_DESCRIPTION = "用四篇短资料体验最新版制度、证据引用与无依据拒答"

_DEMO_DOCS = (
    {
        "filename": "知阁示例-差旅报销制度-2026.md",
        "title": "差旅与报销制度（2026 版）",
        "content": """# 差旅与报销制度（2026 版）

本制度自 2026 年 1 月 1 日起执行，并替代 2024 版差旅标准。

## 住宿标准

一线城市员工住宿标准为每晚 450 元，其他城市为每晚 350 元。超出标准必须在出发前取得直属主管书面审批。

## 报销时限

员工应在出差结束后 7 日内提交报销单并附电子发票。逾期提交需要部门负责人和财务负责人共同审批。

## 高铁与飞机

单程高铁运行时间不超过 5 小时的，原则上选择高铁二等座。因客户急需或夜间出行可选择飞机经济舱。
""",
        "question": "2026 版制度下，一线城市住宿标准和报销时限分别是什么？",
    },
    {
        "filename": "知阁示例-合同审批规则.md",
        "title": "合同审批规则",
        "content": """# 合同审批规则

## 金额门槛

合同含税金额不超过 50 万元的，由业务负责人和部门负责人审批。

合同含税金额超过 50 万元的，必须增加法务负责人与财务负责人审批；超过 200 万元的，还需总经理审批。

## 必备附件

提交审批时应附对方主体资质、报价单、付款计划与验收标准。涉及个人信息的合同还必须附数据保护条款审查记录。
""",
        "question": "超过 50 万元的合同需要哪些角色审批？",
    },
    {
        "filename": "知阁示例-答案可信机制.md",
        "title": "知阁答案可信机制",
        "content": """# 知阁答案可信机制

每个事实应尽可能附带引用编号，用户可以点击引用回到原始分块查看上下文。

系统会对回答做可验证性检查，展示引用覆盖率、无效引用、待核实事实和回答总耗时。

当知识库没有足够资料时，系统应明确拒答，并把问题记录为知识缺口，而不是编造答案。管理员可以在质量治理面板补充资料并关闭缺口。

知识库资料会标记 active、draft 和 expired 状态。下架、重索引、删除和共享操作都会写入审计日志。
""",
        "question": "知阁如何让回答可追溯？",
    },
    {
        "filename": "知阁示例-产品运营手册.md",
        "title": "产品运营手册",
        "content": """# 产品运营手册

## 支持范围

知阁支持 PDF、Word、PowerPoint、Excel、Markdown、文本、图片和网页资料。上传完成后可以在资料面板查看分块状态。

## 团队协作

团队管理员可以创建 viewer、editor 和 admin 三类成员。知识库所有者可把知识库共享为只读或可编辑。

## 数据安全

完整备份包含数据库、向量索引、原文、评估集和知识图谱文件，但不单独打包环境变量、模型密钥、登录 Cookie 或浏览器数据。备份含账号口令摘要，应按敏感数据保管；恢复包需要管理员二次确认并在服务重启时应用。
""",
        "question": "团队里的 viewer 和 editor 有什么区别？",
    },
)

DEMO_EXAMPLES = (
    {
        "label": "制度与流程",
        "title": "查清具体标准和时限",
        "question": "2026 版制度下，一线城市住宿标准和报销时限分别是什么？",
        "hint": "定位到制度条款并附原文",
    },
    {
        "label": "合同与审批",
        "title": "确认多级审批责任",
        "question": "超过 50 万元的合同需要哪些角色审批？",
        "hint": "汇总审批角色并标注依据",
    },
    {
        "label": "产品与安全",
        "title": "看懂系统如何保证可信",
        "question": "知阁如何让回答可追溯？",
        "hint": "从产品机制资料中提取答案",
    },
)
SAMPLE_QUESTIONS = tuple(item["question"] for item in DEMO_EXAMPLES)
REFUSAL_QUESTION = "公司年会的抽奖一等奖是什么？"
REFUSAL_EXAMPLE = {
    "label": "可信边界",
    "title": "没有资料时明确拒答",
    "question": REFUSAL_QUESTION,
    "hint": "验证系统不会编造答案",
    "tone": "boundary",
}


def _demo_kb(db: Session, user: User) -> KnowledgeBase | None:
    return (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.user_id == user.id,
            KnowledgeBase.name == DEMO_KB_NAME,
        )
        .first()
    )


def _demo_status(db: Session, user: User) -> Dict[str, Any]:
    demo = _demo_kb(db, user)
    demo_docs = 0
    demo_chunks = 0
    if demo is not None:
        rows = (
            db.query(DocumentRecord)
            .filter(
                DocumentRecord.user_id == user.id,
                DocumentRecord.kb_id == demo.id,
                DocumentRecord.index_status == "indexed",
            )
            .all()
        )
        demo_docs = len(rows)
        demo_chunks = sum(int(row.chunk_count or 0) for row in rows)

    accessible = list_accessible_kbs(db, user)
    total_docs = sum(int(kb.doc_count or 0) for kb, _ in accessible)
    return {
        "ready": demo_docs >= len(_DEMO_DOCS),
        "first_run": total_docs == 0,
        "demo_kb_id": demo.id if demo is not None else None,
        "demo_documents": demo_docs,
        "demo_chunks": demo_chunks,
        "steps": [
            {"id": "demo", "title": "载入示例知识库", "done": demo_docs > 0},
            {"id": "answer", "title": "提一个可引用的问题", "done": False},
            {"id": "refusal", "title": "验证无依据拒答", "done": False},
        ],
        "sample_questions": list(SAMPLE_QUESTIONS),
        "examples": [*DEMO_EXAMPLES, REFUSAL_EXAMPLE],
        "refusal_question": REFUSAL_QUESTION,
        "refusal_example": REFUSAL_EXAMPLE,
    }


@router.get("/api/v1/onboarding", tags=["首次体验"])
def onboarding_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    return _demo_status(db, current_user)


@router.post("/api/v1/onboarding/demo", tags=["首次体验"])
async def create_demo(
    current_user: User = Depends(require_editor),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """创建或刷新示例库；所有写入使用固定来源，重复调用不会产生重复文档。"""
    cfg = get_runtime_config()
    kb = _demo_kb(db, current_user)
    if kb is None:
        try:
            kb = create_kb(db, current_user, DEMO_KB_NAME, DEMO_DESCRIPTION, cfg)
        except KBError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_dir = resolve_path(cfg.get("paths", {}).get("raw_docs", "data/raw")) / current_user.id[:8]
    ensure_dir(raw_dir)
    docs = []
    for item in _DEMO_DOCS:
        path = raw_dir / item["filename"]
        path.write_text(item["content"], encoding="utf-8")
        docs.append(
            Document(
                page=1,
                content=item["content"],
                metadata={
                    "source": item["filename"],
                    "filepath": str(path),
                    "title": item["title"],
                    "ext": ".md",
                    "doc_status": "active",
                    "acl": current_user.department or cfg.get("data_quality", {}).get("default_acl", "*"),
                    "department": current_user.department or "",
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "kb_id": kb.id,
                },
            )
        )

    chunks, report = await run_in_threadpool(
        run_data_pipeline,
        docs,
        cfg,
        get_splitter(cfg),
        None,
    )
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=f"示例资料未通过当前数据准入门：{report.get('gate', {})}",
        )
    for chunk in chunks:
        chunk.metadata["kb_id"] = kb.id

    store = TenantAwareFactory.get_chroma_store_by_name(kb.collection_name, cfg)

    def _commit() -> None:
        from api.routes.chat import _commit_ingestion_metadata

        _commit_ingestion_metadata(current_user.id, chunks, report, kb.id)

    inserted = await run_in_threadpool(
        store.replace_chunks_for_sources,
        chunks,
        [str(item["filename"]) for item in _DEMO_DOCS],
        _commit,
    )
    try:
        from api.deps import invalidate_runtime_cache_without_loading

        await run_in_threadpool(
            invalidate_runtime_cache_without_loading,
            {
                "id": kb.id,
                "name": kb.name,
                "collection_name": kb.collection_name,
                "owner_id": kb.user_id,
            },
        )
    except Exception:
        logger.debug("示例库缓存失效失败", exc_info=True)

    status = _demo_status(db, current_user)
    return {
        **status,
        "ready": True,
        "ingested": int(inserted),
        "documents": len(_DEMO_DOCS),
        "chunks": len(chunks),
        "kb": {
            "id": kb.id,
            "name": kb.name,
            "collection_name": kb.collection_name,
        },
    }
