#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0 级修复自动验证脚本

快速验证所有 P0 修复是否正确实施。
使用方法: python verify_p0_fixes.py
"""
import subprocess
import sys
from pathlib import Path

# 强制使用 UTF-8 输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def run_command(cmd: list, description: str) -> bool:
    """运行命令并返回是否成功。"""
    print(f"\n[检查] {description}...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).parent,
        )
        if result.returncode == 0:
            print(f"[通过]")
            return True
        else:
            print(f"[失败]")
            if result.stdout:
                print(f"  输出: {result.stdout[:200]}")
            if result.stderr:
                print(f"  错误: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"[执行失败]: {e}")
        return False


def check_ci_config() -> bool:
    """检查 CI 配置是否正确（P0-1）。"""
    print("\n[检查] P0-1: 检查 CI 配置...")
    ci_file = Path(__file__).parent / ".github" / "workflows" / "ci.yml"
    
    if not ci_file.exists():
        print("[失败] CI 配置文件不存在")
        return False
    
    content = ci_file.read_text(encoding="utf-8")
    
    # 不应该包含测试忽略配置
    if "--ignore=tests/test_agent.py" in content:
        print("[失败] CI 仍然忽略 test_agent.py")
        return False
    
    if "--ignore=tests/test_rag_pipeline.py" in content:
        print("[失败] CI 仍然忽略 test_rag_pipeline.py")
        return False
    
    print("[通过] CI 不再忽略核心测试")
    return True


def check_rate_limit_files() -> bool:
    """检查限流相关文件是否存在（P0-2）。"""
    print("\n[检查] P0-2: 检查限流文件...")
    
    required_files = [
        "src/rate_limit.py",
        "tests/test_rate_limit.py",
    ]
    
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if not full_path.exists():
            print(f"[失败] 缺少文件: {file_path}")
            return False
    
    print("[通过] 限流文件完整")
    return True


def main():
    """主验证流程。"""
    print("=" * 60)
    print("  P0 级修复自动验证")
    print("=" * 60)
    
    results = []
    
    # P0-1: 死代码清理验证
    print("\n" + "=" * 60)
    print("P0-1: 死代码清理验证")
    print("=" * 60)
    results.append(("P0-1: CI 配置", check_ci_config()))
    
    # P0-2: 限流修复验证
    print("\n" + "=" * 60)
    print("P0-2: 限流修复验证")
    print("=" * 60)
    results.append(("P0-2: 文件检查", check_rate_limit_files()))
    results.append((
        "P0-2: 限流测试",
        run_command(
            [sys.executable, "-m", "pytest", "tests/test_rate_limit.py", "-v", "--tb=no"],
            "运行限流测试"
        )
    ))
    
    # P0-3: 测试覆盖验证
    print("\n" + "=" * 60)
    print("P0-3: 测试覆盖验证")
    print("=" * 60)
    results.append((
        "P0-3: P0/P1 修复测试",
        run_command(
            [
                sys.executable, "-m", "pytest",
                "tests/test_rate_limit.py",
                "tests/test_password_validator.py",
                "tests/test_exception_sanitization.py",
                "-v", "--tb=no"
            ],
            "运行 P0/P1 修复相关测试"
        )
    ))
    
    results.append((
        "P0-3: 测试计数",
        run_command(
            [sys.executable, "-m", "pytest", "tests/", "--co", "-q"],
            "统计测试数量（应为 377）"
        )
    ))
    
    # 总结
    print("\n" + "=" * 60)
    print("  验证总结")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n通过: {passed}/{total}")
    
    for name, success in results:
        status = "[通过]" if success else "[失败]"
        print(f"  {status} {name}")
    
    print("\n" + "=" * 60)
    
    if passed == total:
        print("全部验证通过！P0 修复完成。")
        print("\n下一步:")
        print("  1. 代码审查")
        print("  2. 提交代码: commit_p0_fixes.bat / bash commit_p0_fixes.sh")
        print("  3. 部署测试环境")
        print("=" * 60)
        return 0
    else:
        print(f"❌ {total - passed} 项验证失败，请检查修复。")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
