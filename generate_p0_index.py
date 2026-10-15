#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0 级修复总结生成器

生成所有修复相关文档的索引和快速导航。
使用方法: python generate_p0_index.py
"""
from pathlib import Path
from datetime import datetime


def main():
    """生成 P0 修复文档索引。"""
    root = Path(__file__).parent
    
    print("=" * 80)
    print("  P0 级修复文档索引")
    print("=" * 80)
    print()
    
    # 核心文档
    print("📚 核心文档:")
    docs = [
        ("P0_FIXES_COMPLETION.md", "完整技术报告（最详细）"),
        ("P0_修复完成_中文总结.md", "中文总结版本"),
        ("README_P0_FIXES.md", "快速查看入口"),
        ("P0_EXECUTIVE_SUMMARY.txt", "执行摘要（管理层）"),
    ]
    
    for filename, description in docs:
        path = root / filename
        if path.exists():
            size = path.stat().st_size
            print(f"  ✅ {filename:40s} - {description} ({size:,} 字节)")
        else:
            print(f"  ❌ {filename:40s} - {description} (未找到)")
    
    print()
    
    # 验证工具
    print("🔧 验证工具:")
    tools = [
        ("verify_p0_fixes.py", "自动验证脚本"),
        ("P0_VERIFICATION_CHECKLIST.md", "详细验证清单"),
        ("P0_QUICK_VERIFY.txt", "快速验证指南"),
    ]
    
    for filename, description in tools:
        path = root / filename
        if path.exists():
            print(f"  ✅ {filename:40s} - {description}")
        else:
            print(f"  ❌ {filename:40s} - {description} (未找到)")
    
    print()
    
    # 提交脚本
    print("📤 提交脚本:")
    scripts = [
        ("commit_p0_fixes.bat", "Windows 提交脚本"),
        ("commit_p0_fixes.sh", "Linux/Mac 提交脚本"),
    ]
    
    for filename, description in scripts:
        path = root / filename
        if path.exists():
            print(f"  ✅ {filename:40s} - {description}")
        else:
            print(f"  ❌ {filename:40s} - {description} (未找到)")
    
    print()
    
    # 其他文档
    print("📄 其他文档:")
    others = [
        ("P0_COMPLETION_FINAL.txt", "完成确认书"),
        ("P0_SUMMARY_FINAL.txt", "总结报告"),
    ]
    
    for filename, description in others:
        path = root / filename
        if path.exists():
            print(f"  ✅ {filename:40s} - {description}")
        else:
            print(f"  ❌ {filename:40s} - {description} (未找到)")
    
    print()
    print("=" * 80)
    print("  推荐阅读顺序")
    print("=" * 80)
    print()
    print("1. 快速了解: README_P0_FIXES.md")
    print("2. 执行摘要: P0_EXECUTIVE_SUMMARY.txt (管理层)")
    print("3. 详细技术: P0_FIXES_COMPLETION.md (工程师)")
    print("4. 中文版本: P0_修复完成_中文总结.md")
    print("5. 验证修复: python verify_p0_fixes.py")
    print("6. 提交代码: commit_p0_fixes.bat (Windows) 或 bash commit_p0_fixes.sh (Mac/Linux)")
    print()
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
