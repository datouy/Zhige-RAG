"""多租户迁移脚本。

将现有的单用户 kg.db 和 Chroma collection 迁移到多租户格式。

用法：
    python scripts/migrate_to_multi_tenant.py [--user-id USER_ID]

注意：
    - 脚本会备份原始文件（添加 .backup 后缀）
    - 默认 user_id 为 "default_user"
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def migrate_single_user_data(default_user_id: str = "default_user") -> dict:
    """将现有的单一 kg.db 迁移为多租户格式。

    Args:
        default_user_id: 迁移的目标用户 ID

    Returns:
        迁移结果字典
    """
    results = {
        "kg_db": {"status": "skipped", "source": None, "target": None},
        "chroma": {"status": "info", "message": "Chroma 多租户由 TenantAwareFactory 自动处理"},
    }

    # 迁移 KG 数据库
    source_kg_db = Path("data/kg.db")
    if source_kg_db.exists():
        # 获取用户 ID 前8位
        user_prefix = default_user_id[:8] if len(default_user_id) >= 8 else default_user_id
        target_kg_db = Path(f"data/kg_{user_prefix}.db")

        # 备份原文件
        backup_kg_db = source_kg_db.with_suffix(".db.backup")
        if not backup_kg_db.exists():
            shutil.copy2(source_kg_db, backup_kg_db)
            results["kg_db"]["backup"] = str(backup_kg_db)

        # 复制到用户专属文件
        target_kg_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_kg_db, target_kg_db)
        results["kg_db"]["status"] = "migrated"
        results["kg_db"]["source"] = str(source_kg_db)
        results["kg_db"]["target"] = str(target_kg_db)
        print(f"✓ KG 数据库已迁移: {source_kg_db} -> {target_kg_db}")
    else:
        print(f"ℹ 未找到 {source_kg_db}，跳过 KG 迁移")
        results["kg_db"]["status"] = "not_found"
        results["kg_db"]["source"] = str(source_kg_db)

    # Chroma collection 迁移说明
    print("\nℹ Chroma 多租户隔离说明：")
    print("  - TenantAwareFactory 会自动为每个用户创建独立的 collection")
    print("  - collection 命名规则: {user_id[:8]}_{base_collection_name}")
    print("  - 现有数据保留在 chinese_rag_kb collection 中")
    print("  - 可以通过以下方式迁移 Chroma 数据：")
    print("    1. 创建新用户专属 collection")
    print("    2. 从原有 collection 复制数据到新 collection")
    print("    3. 删除原有 collection（可选）")

    return results


def create_default_user_data_dir(default_user_id: str = "default_user") -> None:
    """创建默认用户的数据目录。

    Args:
        default_user_id: 用户 ID
    """
    user_prefix = default_user_id[:8] if len(default_user_id) >= 8 else default_user_id

    # 创建用户目录
    dirs_to_create = [
        f"data/raw/{user_prefix}",  # 用户原始文档目录
    ]

    for dir_path in dirs_to_create:
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="多租户迁移脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    # 使用默认用户 ID 迁移
    python scripts/migrate_to_multi_tenant.py

    # 指定用户 ID 迁移
    python scripts/migrate_to_multi_tenant.py --user-id my_user_123
        """,
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=os.getenv("DEFAULT_USER_ID", "default_user"),
        help="迁移的目标用户 ID（默认: default_user）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要执行的操作，不实际执行",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("多租户迁移工具")
    print("=" * 60)
    print(f"目标用户 ID: {args.user_id}")
    print()

    if args.dry_run:
        print("⚠ DRY RUN 模式：仅显示将要执行的操作")
        print()
        print("将要执行的操作：")
        print("  1. 备份 data/kg.db -> data/kg.db.backup")
        print(f"  2. 复制 data/kg.db -> data/kg_{args.user_id[:8]}.db")
        print(f"  3. 创建用户目录: data/raw/{args.user_id[:8]}/")
        print()
        print("Chroma collection 将保持不变，由 TenantAwareFactory 自动处理")
    else:
        # 创建用户目录
        create_default_user_data_dir(args.user_id)

        # 执行迁移
        results = migrate_single_user_data(args.user_id)

        print()
        print("=" * 60)
        print("迁移完成")
        print("=" * 60)
        print(f"KG 数据库: {results['kg_db']['status']}")
        if results['kg_db'].get('target'):
            print(f"  目标文件: {results['kg_db']['target']}")
        if results['kg_db'].get('backup'):
            print(f"  备份文件: {results['kg_db']['backup']}")
        print()
        print("下一步：")
        print("  1. 启动应用: uvicorn api.main:app --reload")
        print("  2. 注册新用户或使用默认用户登录")
        print("  3. 上传文档测试多租户隔离")


if __name__ == "__main__":
    main()
