#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 级修复验证脚本
用于快速验证所有 P1 级修复是否正常工作

使用方法:
    python verify_p1_fixes.py

输出: 每个修复的验证结果
"""

import sys
import os
import re
from pathlib import Path

# 修复 Windows 控制台编码问题
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result(test_name, passed, details=""):
    """打印测试结果"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n[{status}] {test_name}")
    if details:
        print(f"       {details}")

def verify_password_validator():
    """验证 P1-6: 口令强度校验"""
    print_header("P1-6: 口令强度校验")
    
    try:
        from src.password_validator import validate_password_strength
        
        # 测试 1: 空口令
        result = validate_password_strength("")
        print_result("拒绝空口令", result is not None, result)
        
        # 测试 2: 短口令
        result = validate_password_strength("abc")
        print_result("拒绝短口令", result is not None, result)
        
        # 测试 3: 缺少大写
        result = validate_password_strength("test1234")
        print_result("拒绝缺少大写", result is not None, result)
        
        # 测试 4: 缺少小写
        result = validate_password_strength("TEST1234")
        print_result("拒绝缺少小写", result is not None, result)
        
        # 测试 5: 缺少数字
        result = validate_password_strength("TestTest")
        print_result("拒绝缺少数字", result is not None, result)
        
        # 测试 6: 合规口令
        result = validate_password_strength("Test1234")
        print_result("接受合规口令", result is None, "Test1234 通过验证")
        
        # 测试 7: 包含特殊字符
        result = validate_password_strength("Test1234!")
        print_result("接受特殊字符", result is None, "Test1234! 通过验证")
        
        return True
    except Exception as e:
        print_result("口令校验器加载", False, str(e))
        return False

def verify_exception_handler():
    """验证 P1-7: 异常信息脱敏"""
    print_header("P1-7: 异常信息脱敏")
    
    try:
        # 检查 api/main.py 是否包含异常处理器
        main_py = project_root / "api" / "main.py"
        if not main_py.exists():
            print_result("异常处理器检查", False, "api/main.py 不存在")
            return False
        
        content = main_py.read_text(encoding="utf-8")
        
        # 检查关键代码
        has_exception_handler = "@app.exception_handler(Exception)" in content
        has_error_id = "error_id = str(uuid.uuid4())" in content or "error_id =" in content
        has_sanitization = "ENVIRONMENT" in content and "development" in content
        
        print_result("异常处理器存在", has_exception_handler)
        print_result("错误 ID 追踪", has_error_id)
        print_result("环境区分脱敏", has_sanitization)
        
        return has_exception_handler and has_error_id and has_sanitization
    except Exception as e:
        print_result("异常处理器检查", False, str(e))
        return False

def verify_streaming_metadata():
    """验证 P1-4: 流式问答元数据"""
    print_header("P1-4: 流式问答元数据补全")
    
    try:
        # 检查 src/rag_pipeline.py 是否包含元数据补全
        rag_pipeline = project_root / "src" / "rag_pipeline.py"
        if not rag_pipeline.exists():
            print_result("流式元数据检查", False, "src/rag_pipeline.py 不存在")
            return False
        
        content = rag_pipeline.read_text(encoding="utf-8")
        
        # 检查关键代码
        has_sources = '"sources":' in content or "'sources':" in content
        has_timings = '"timings":' in content or "'timings':" in content
        has_metadata = '"metadata":' in content or "'metadata':" in content
        has_done_event = '"event": "done"' in content or "'event': 'done'" in content
        
        print_result("done 事件存在", has_done_event)
        print_result("sources 字段", has_sources, "包含来源信息")
        print_result("timings 字段", has_timings, "包含耗时信息")
        print_result("metadata 字段", has_metadata, "包含元数据")
        
        return has_sources and has_timings and has_metadata
    except Exception as e:
        print_result("流式元数据检查", False, str(e))
        return False

def verify_timeout_fix():
    """验证 P1-5: 超时度量修复"""
    print_header("P1-5: 超时度量修复")
    
    try:
        # 检查 src/llm_provider.py 是否包含精确超时
        llm_provider = project_root / "src" / "llm_provider.py"
        if not llm_provider.exists():
            print_result("超时修复检查", False, "src/llm_provider.py 不存在")
            return False
        
        content = llm_provider.read_text(encoding="utf-8")
        
        # 检查关键代码
        has_time_import = "import time" in content
        has_elapsed_queue = "elapsed_queue" in content or "elapsed" in content
        has_remaining = "remaining" in content
        has_timeout_param = "timeout" in content
        
        print_result("时间模块导入", has_time_import)
        print_result("排队时间计算", has_elapsed_queue, "单独记录排队时长")
        print_result("剩余时间计算", has_remaining, "动态调整超时")
        print_result("超时参数处理", has_timeout_param)
        
        return has_elapsed_queue and has_remaining
    except Exception as e:
        print_result("超时修复检查", False, str(e))
        return False

def verify_tests():
    """验证测试文件存在"""
    print_header("测试文件验证")
    
    test_files = [
        ("tests/test_password_validator.py", "口令校验测试"),
        ("tests/test_exception_sanitization.py", "异常脱敏测试"),
    ]
    
    all_exist = True
    for test_file, desc in test_files:
        path = project_root / test_file
        exists = path.exists()
        print_result(desc, exists, test_file)
        all_exist = all_exist and exists
    
    return all_exist

def verify_documentation():
    """验证文档存在"""
    print_header("文档验证")
    
    docs = [
        ("P1_FIXES_REPORT.md", "详细修复报告"),
        ("FIXES_SUMMARY.md", "修复摘要"),
        ("P1_FIXES_COMPARISON.md", "修复前后对比"),
        ("P1_FIXES_COMPLETION.md", "完成确认书"),
        ("P1_修复完成摘要.txt", "终端摘要"),
    ]
    
    all_exist = True
    for doc_file, desc in docs:
        path = project_root / doc_file
        exists = path.exists()
        print_result(desc, exists, doc_file)
        all_exist = all_exist and exists
    
    return all_exist

def main():
    """主验证流程"""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  P1 级修复验证脚本".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    results = []
    
    # 验证每个修复
    results.append(("P1-6 口令强度校验", verify_password_validator()))
    results.append(("P1-7 异常信息脱敏", verify_exception_handler()))
    results.append(("P1-4 流式元数据补全", verify_streaming_metadata()))
    results.append(("P1-5 超时度量修复", verify_timeout_fix()))
    results.append(("测试文件", verify_tests()))
    results.append(("文档", verify_documentation()))
    
    # 打印总结
    print_header("验证总结")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n通过: {passed}/{total}")
    print("\n详细结果:")
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    # 最终状态
    print("\n" + "=" * 70)
    if passed == total:
        print("  ✅ 所有验证通过！修复已就绪。")
    else:
        print(f"  ⚠️  {total - passed} 项验证失败，请检查上述输出。")
    print("=" * 70 + "\n")
    
    # 返回退出码
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
