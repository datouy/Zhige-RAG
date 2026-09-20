"""检查分块质量工具。

用法：
    python scripts/check_chunks.py <文档路径> [--num 5]
    
示例：
    python scripts/check_chunks.py data/raw/测试文档.txt
    python scripts/check_chunks.py data/raw/测试文档.pdf --num 10
"""

import argparse
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config
from src.document_loader import load_single_document
from src.text_splitter import ChineseTextSplitter, load_custom_dict_from_file


def check_chunks(doc_path: str, num_chunks: int = 5):
    """检查文档分块质量。"""
    print(f"\n{'='*80}")
    print(f"📄 检查文档: {doc_path}")
    print(f"{'='*80}\n")
    
    # 加载配置
    config = load_config()
    
    # 加载文档
    doc = load_single_document(doc_path)
    if not doc:
        print("❌ 无法加载文档")
        return
    
    print(f"✅ 文档加载成功")
    print(f"   - 原始长度: {len(doc.content)} 字符")
    print(f"   - 元数据: {doc.metadata}\n")
    
    # 加载自定义词典
    custom_dict_path = "config/custom_dict.txt"
    custom_words = load_custom_dict_from_file(custom_dict_path)
    
    # 初始化分块器
    ts_cfg = config.get("text_splitter", {})
    splitter = ChineseTextSplitter(
        chunk_size=ts_cfg.get("chunk_size", 400),
        chunk_overlap=ts_cfg.get("chunk_overlap", 50),
        separators=ts_cfg.get("chinese_separators"),
        min_chunk_size=ts_cfg.get("min_chunk_size", 32),
        custom_dict_words=custom_words,
    )
    
    # 分块
    chunks = splitter.split_text(doc.content, metadata=doc.metadata)
    
    print(f"📊 分块统计:")
    print(f"   - 总块数: {len(chunks)}")
    print(f"   - 平均长度: {sum(len(c.text) for c in chunks) / len(chunks):.1f} 字符")
    print(f"   - 最小长度: {min(len(c.text) for c in chunks)}")
    print(f"   - 最大长度: {max(len(c.text) for c in chunks)}\n")
    
    # 打印前 N 个分块
    print(f"{'='*80}")
    print(f"📝 前 {min(num_chunks, len(chunks))} 个分块预览:")
    print(f"{'='*80}\n")
    
    for i, chunk in enumerate(chunks[:num_chunks]):
        print(f"{'─'*80}")
        print(f"块 {i+1}:")
        print(f"  长度: {len(chunk.text)} 字符")
        print(f"  内容:\n")
        # 显示前 200 字符，避免输出过长
        preview = chunk.text[:200]
        if len(chunk.text) > 200:
            preview += "..."
        print(f"    {preview}")
        print()
    
    print(f"{'='*80}")
    print(f"✅ 分块质量检查完成！")
    print(f"{'='*80}\n")
    
    # 质量建议
    print("💡 质量检查要点:")
    print("   1. 每个块是否语义完整（没有在句子中间截断）")
    print("   2. 块长度是否合理（300-500 字符）")
    print("   3. 是否有过多重复内容（overlap 是否合理）")
    print("   4. 专有名词是否被正确识别（未被切碎）")


def main():
    parser = argparse.ArgumentParser(description="检查文档分块质量")
    parser.add_argument("doc_path", help="文档路径")
    parser.add_argument("--num", type=int, default=5, help="显示的分块数量（默认 5）")
    
    args = parser.parse_args()
    
    doc_path = Path(args.doc_path)
    if not doc_path.exists():
        print(f"❌ 文件不存在: {args.doc_path}")
        return
    
    check_chunks(str(doc_path), args.num)


if __name__ == "__main__":
    main()
