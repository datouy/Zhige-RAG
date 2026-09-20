# 中文 RAG 系统优化指南

本文档说明已完成的优化配置和使用建议。

## ✅ 已完成的优化

### 1. 入库后自动检查分块质量

**功能**：入库脚本现在会自动打印前 5 个分块样例，方便检查切分质量。

**使用方法**：
```bash
# 入库时自动展示分块样例
python scripts/ingest.py data/raw/your_document.pdf

# 入库某个目录
python scripts/ingest.py data/raw --recursive
```

**输出示例**：
```
📊 分块质量检查 - 样例展示
==================================================
总分块数: 42
平均长度: 385.2 字符
最小长度: 156
最大长度: 498

前 5 个分块预览:
==================================================
分块 1:
  长度: 392 字符
  来源: document.pdf
  标题: 第一章 系统概述
  内容预览:
    本系统是一个面向中文知识库的 RAG 检索增强生成系统...
```

**质量检查要点**：
- ✓ 每个块语义完整（未在句子中间截断）
- ✓ 块长度合理（300-500 字符为宜）
- ✓ 专有名词未被切碎
- ✓ 有适当的上下文重叠

**独立检查工具**：
```bash
# 查看文档分块效果（不入库）
python scripts/check_chunks.py data/raw/test.pdf

# 查看更多分块
python scripts/check_chunks.py data/raw/test.pdf --num 10
```

---

### 2. 自定义词典配置

**位置**：`config/custom_dict.txt`

**功能**：防止专有名词被 jieba 分词切碎，提升分块质量。

**已内置词典分类**：
- AI/机器学习术语（RAG系统、向量数据库、语义检索等）
- 编程语言（Java、JavaScript、Python 等）
- 框架/库（Spring Boot、Django、React 等）
- 数据库（MySQL、PostgreSQL、Redis 等）
- 云服务/DevOps（Docker、Kubernetes、CI/CD 等）
- 设计模式（单例模式、工厂模式等）
- 架构设计（微服务、分布式系统、高并发等）
- 常见误切词（"不知道"、"怎么样"等）

**添加自定义词汇**：
1. 编辑 `config/custom_dict.txt`
2. 每行添加一个词（支持 `#` 注释）
3. 重新运行入库脚本，词典会自动加载

**示例**：
```txt
# 公司产品名
智能客服系统
知识管理平台
业务中台

# 业务术语
用户画像
转化率
A/B测试
```

**自动加载**：入库脚本和分块器会自动读取该文件，无需手动调用。

---

### 3. 速度与精度调优

#### 🚀 加速模式（优先响应速度）

**配置**：`config/config.yaml`
```yaml
reranker:
  enabled: false    # 关闭 reranker，速度提升 2-3 倍
```

**适用场景**：
- 快速原型开发
- 响应时间敏感的应用
- 知识库内容质量较高、误召回少

#### 🎯 精准模式（优先检索精度）

**配置**：`config/config.yaml`
```yaml
reranker:
  enabled: true     # 开启 reranker
  top_n: 3          # 重排后返回 Top 3

llm:
  generation:
    temperature: 0.1   # 降低温度至 0.1
```

**适用场景**：
- 对答案准确性要求高
- 知识库内容复杂、噪声较多
- 可以接受稍慢的响应时间

---

### 4. 已优化的默认配置

#### 文本分块参数（已调优）
```yaml
text_splitter:
  chunk_size: 400        # 中文最佳长度（300-500）
  chunk_overlap: 50      # 块间重叠，保持上下文
  chinese_separators:    # 中文标点优先
    - "\n\n"             # 段落
    - "\n"               # 换行
    - "。"               # 句号
    - "！"               # 感叹号
    - "？"               # 问号
    - "；"               # 分号
    - "，"               # 逗号
```

#### 混合检索（已启用）
```yaml
vector_store:
  hybrid:
    enabled: true        # 向量 + BM25 混合检索
    fetch_k: 20          # BM25 召回数
```

#### LLM 生成参数（已优化为精准模式）
```yaml
llm:
  generation:
    temperature: 0.1     # 低温，更精准的事实型回答
    top_p: 0.9
    max_new_tokens: 256
```

---

## 📊 性能对比

| 配置 | 响应时间 | 准确率 | 适用场景 |
|------|---------|--------|---------|
| reranker: false + temperature: 0.1 | 快（基准） | 中 | 快速开发、简单问答 |
| reranker: true + temperature: 0.1 | 慢 2-3x | 高 | 生产环境、复杂检索 |
| reranker: false + temperature: 0.3 | 快（基准） | 中 | 创意生成、开放问答 |

---

## 🎯 最佳实践建议

### 开发阶段
```yaml
# config/config.yaml
reranker:
  enabled: false          # 快速迭代
llm:
  generation:
    temperature: 0.1      # 保持精准
```

### 生产部署
```yaml
# config/config.yaml
reranker:
  enabled: true           # 提升准确率
  top_n: 3
llm:
  generation:
    temperature: 0.1      # 事实型问答
```

### 创意生成场景
```yaml
llm:
  generation:
    temperature: 0.5      # 适度发散
    top_p: 0.95
```

---

## 🔧 快速调优流程

1. **入库测试**
   ```bash
   python scripts/ingest.py data/raw/test_doc.pdf
   # 查看输出的分块样例，确认质量
   ```

2. **检查分块质量**
   - 块长度是否合理（300-500 字符）
   - 专有名词是否完整（未被切碎）
   - 语义是否完整（未在句中截断）

3. **添加自定义词典**
   - 发现被切碎的专有名词
   - 添加到 `config/custom_dict.txt`
   - 重新入库

4. **调整检索精度**
   - 测试查询效果
   - 如果召回质量差：开启 reranker
   - 如果生成不够精准：降低 temperature

5. **性能优化**
   - 响应慢：关闭 reranker
   - 准确率低：开启 reranker + 降低 temperature

---

## 📝 常见问题

### Q: 分块质量不好怎么办？
A: 
1. 检查 `chunk_size` 是否合理（推荐 300-500）
2. 添加领域专有名词到 `config/custom_dict.txt`
3. 运行 `scripts/check_chunks.py` 独立检查

### Q: 检索召回不准确？
A: 
1. 开启 reranker: `enabled: true`
2. 确认混合检索已启用：`hybrid.enabled: true`
3. 调整 `retrieval.top_k` 和 `reranker.top_n`

### Q: 响应速度太慢？
A: 
1. 关闭 reranker: `enabled: false`
2. 降低 `retrieval.top_k`（默认 8）
3. 检查模型是否使用了量化（quantization）

### Q: 生成内容不够准确？
A: 
1. 降低 temperature 至 0.1
2. 开启 reranker 过滤噪声
3. 提高 `verification.min_grounding`（最小接地度）

---

## 📚 相关文档

- 完整配置说明：`config/config.yaml`
- 分块工具：`scripts/check_chunks.py`
- 入库脚本：`scripts/ingest.py`
- 文本分块器：`src/text_splitter.py`
