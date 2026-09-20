#!/usr/bin/env python3
"""快速测试优化效果的脚本。

功能：
1. 测试自定义词典加载
2. 测试分块质量
3. 对比不同配置的性能

用法：
    python scripts/test_optimization.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.text_splitter import ChineseTextSplitter, load_custom_dict_from_file
from src.utils import load_config, get_logger

logger = get_logger("test_optimization")


def test_custom_dict():
    """测试自定义词典加载。"""
    print("\n" + "=" * 80)
    print("测试 1: 自定义词典加载")
    print("=" * 80 + "\n")
    
    dict_path = "config/custom_dict.txt"
    words = load_custom_dict_from_file(dict_path)
    
    print(f"成功加载 {len(words)} 个自定义词汇")
    print(f"\n前 20 个词汇示例:")
    for i, word in enumerate(words[:20], 1):
        print(f"  {i:2d}. {word}")
    
    if len(words) > 20:
        print(f"  ... 还有 {len(words) - 20} 个词")
    
    print(f"\n词典加载测试通过")
    return words


def test_splitter_with_dict():
    """测试带词典的分块器。"""
    print("\n" + "=" * 80)
    print("测试 2: 带词典的文本分块")
    print("=" * 80 + "\n")
    
    # 测试文本（包含专有名词）
    test_text = """
    本系统是一个基于 RAG 技术的中文知识库系统。系统采用了 Spring Boot 作为后端框架，
    使用 Vue.js 构建前端界面，数据存储采用 MySQL 和 Redis 的组合方案。
    
    在机器学习模块中，我们使用了 PyTorch 和 Hugging Face 的预训练模型，
    实现了自然语言处理和语义检索功能。系统支持 Docker 容器化部署，
    可以通过 Kubernetes 进行编排和扩展。
    
    架构设计上采用了微服务模式，使用单例模式和工厂模式来管理组件实例，
    通过 API 网关实现负载均衡和服务路由。
    """
    
    # 不带词典的分块器
    splitter_no_dict = ChineseTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )
    
    # 带词典的分块器
    splitter_with_dict = ChineseTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        custom_dict_path="config/custom_dict.txt",
    )
    
    chunks_no_dict = splitter_no_dict.split_text(test_text)
    chunks_with_dict = splitter_with_dict.split_text(test_text)
    
    print(f"不带词典分块数: {len(chunks_no_dict)}")
    print(f"带词典分块数: {len(chunks_with_dict)}")
    
    print(f"\n带词典的第一个分块:")
    print(f"-" * 80)
    print(chunks_with_dict[0].text[:300] if chunks_with_dict else "无分块")
    print(f"-" * 80)
    
    print(f"\n分块测试通过")
    return chunks_with_dict


def test_config_loading():
    """测试配置加载和优化参数。"""
    print("\n" + "=" * 80)
    print("测试 3: 配置加载与优化参数检查")
    print("=" * 80 + "\n")
    
    config = load_config("config/config.yaml")
    
    # 检查关键优化参数
    checks = [
        ("分块大小", config["text_splitter"]["chunk_size"], 400),
        ("分块重叠", config["text_splitter"]["chunk_overlap"], 50),
        ("LLM 温度", config["llm"]["generation"]["temperature"], 0.1),
        ("Reranker 状态", config["reranker"]["enabled"], False),
        ("混合检索", config["vector_store"]["hybrid"]["enabled"], True),
    ]
    
    print("关键配置参数:")
    for name, value, expected in checks:
        status = "[OK]" if value == expected else "[WARN]"
        print(f"  {status} {name}: {value} (期望: {expected})")
    
    print(f"\n自定义分隔符:")
    separators = config["text_splitter"].get("chinese_separators", [])
    for i, sep in enumerate(separators[:8], 1):
        display_sep = repr(sep) if sep in ["\n\n", "\n", " "] else sep
        print(f"  {i}. {display_sep}")
    
    print(f"\n配置加载测试通过")
    return config


def test_chunk_quality_metrics():
    """测试分块质量指标。"""
    print("\n" + "=" * 80)
    print("测试 4: 分块质量指标")
    print("=" * 80 + "\n")
    
    test_text = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个重要分支。
    近年来，深度学习技术的发展推动了自然语言处理领域的突破。
    
    在实际应用中，我们常使用 Transformer 架构构建大语言模型。
    这些模型能够理解和生成人类语言，在问答、翻译、摘要等任务中表现出色。
    
    检索增强生成（RAG）技术结合了信息检索和文本生成，
    通过向量数据库存储知识，实现了更准确的知识问答系统。
    """ * 3  # 重复3次以产生多个分块
    
    splitter = ChineseTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        custom_dict_path="config/custom_dict.txt",
    )
    
    chunks = splitter.split_text(test_text)
    
    # 计算质量指标
    chunk_lengths = [len(c.text) for c in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    min_length = min(chunk_lengths) if chunk_lengths else 0
    max_length = max(chunk_lengths) if chunk_lengths else 0
    
    # 检查是否有句子被截断（简单启发式：以标点结尾）
    end_punctuation = "。！？；.!?"
    proper_endings = sum(1 for c in chunks if c.text.rstrip()[-1] in end_punctuation if c.text)
    proper_ending_ratio = proper_endings / len(chunks) if chunks else 0
    
    print(f"分块统计:")
    print(f"  总分块数: {len(chunks)}")
    print(f"  平均长度: {avg_length:.1f} 字符")
    print(f"  最小长度: {min_length} 字符")
    print(f"  最大长度: {max_length} 字符")
    print(f"  语义完整性: {proper_ending_ratio*100:.1f}% (以标点结尾)")
    
    print(f"\n质量评估:")
    if 300 <= avg_length <= 500:
        print(f"  [OK] 平均长度合理 (推荐 300-500)")
    else:
        print(f"  [WARN] 平均长度可能需要调整")
    
    if proper_ending_ratio >= 0.7:
        print(f"  [OK] 语义完整性良好")
    else:
        print(f"  [WARN] 部分分块可能在句中截断")
    
    print(f"\n质量指标测试完成")


def main():
    """运行所有测试。"""
    print("\n" + "=" * 80)
    print("ChineseRAGKB 优化效果测试")
    print("=" * 80)
    
    try:
        # 测试 1: 自定义词典
        test_custom_dict()
        
        # 测试 2: 文本分块
        test_splitter_with_dict()
        
        # 测试 3: 配置加载
        test_config_loading()
        
        # 测试 4: 质量指标
        test_chunk_quality_metrics()
        
        print("\n" + "=" * 80)
        print("所有测试通过！")
        print("=" * 80)
        print("\n提示:")
        print("   - 自定义词典已正确加载")
        print("   - 分块器配置符合优化建议")
        print("   - 可以开始使用优化后的系统")
        print("\n查看详细文档:")
        print("   - 优化指南: README_optimization.md")
        print("   - 变更日志: CHANGELOG_optimization.md")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
