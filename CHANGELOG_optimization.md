# 优化变更日志

本文档记录系统优化相关的变更。

## 2026-09-20 - 分块质量与性能优化

### ✨ 新增功能

#### 1. 入库后自动检查分块质量
- **文件**: `scripts/ingest.py`
- **功能**: 入库完成后自动打印前 5 个分块样例
- **输出内容**:
  - 总分块数、平均长度、最小/最大长度
  - 每个分块的长度、来源、标题
  - 前 200 字符内容预览
  - 质量检查要点提示
- **使用**: 运行 `python scripts/ingest.py <path>` 即可自动展示

#### 2. 扩展自定义词典
- **文件**: `config/custom_dict.txt`
- **新增分类**:
  - AI/机器学习术语（50+ 词）
  - 编程语言（15+ 词）
  - 框架/库（30+ 词）
  - 数据库（20+ 词）
  - 云服务/DevOps（20+ 词）
  - 设计模式（18 词）
  - 架构设计（15+ 词）
  - 前端技术（10+ 词）
  - 业务术语示例（20+ 词）
  - 常见误切词（30+ 词）
- **总词数**: 从 42 个扩展到 230+ 个
- **效果**: 显著减少专有名词被 jieba 误切

#### 3. 自定义词典自动加载
- **文件**: `scripts/ingest.py`
- **改动**: `build_splitter()` 函数
- **功能**: 分块器初始化时自动加载 `config/custom_dict.txt`
- **参数**: 新增 `custom_dict_path="config/custom_dict.txt"`

#### 4. 文本分块器支持词典路径参数
- **文件**: `src/text_splitter.py`
- **改动**: `ChineseTextSplitter.__init__()`
- **新增参数**: `custom_dict_path: Optional[str] = None`
- **功能**: 同时支持传入词表列表和词典文件路径，自动合并

### ⚙️ 配置优化

#### 1. LLM 温度调低以提升精准度
- **文件**: `config/config.yaml`
- **改动**: `llm.generation.temperature: 0.15 → 0.1`
- **效果**: 更精准的事实型回答，减少幻觉
- **适用场景**: 知识库问答、技术文档检索

#### 2. Reranker 默认关闭以提升速度
- **文件**: `config/config.yaml`
- **改动**: `reranker.enabled: true → false`
- **原因**: 
  - Reranker 会增加 2-3 倍响应时间
  - 对于质量较好的知识库，向量+BM25 混合检索已足够
  - 用户可根据需求在配置中开启
- **注释**: 添加速度与精度权衡说明

### 📚 文档新增

#### 1. 优化指南文档
- **文件**: `README_optimization.md`
- **内容**:
  - 已完成优化的详细说明
  - 入库质量检查使用方法
  - 自定义词典配置指南
  - 速度与精度调优建议
  - 性能对比表格
  - 最佳实践建议
  - 快速调优流程
  - 常见问题 FAQ

#### 2. 变更日志
- **文件**: `CHANGELOG_optimization.md`（本文件）
- **内容**: 记录所有优化相关的变更

#### 3. README 更新
- **文件**: `README.md`
- **改动**: 文档导航中新增优化指南链接

### 🔧 代码改进

#### 1. 入库脚本新增分块预览函数
```python
def _print_chunk_samples(chunks, num_samples: int = 5)
```
- 打印分块统计信息
- 展示前 N 个分块样例
- 提供质量检查要点

#### 2. 文本分块器参数增强
```python
ChineseTextSplitter(
    # ... 原有参数 ...
    custom_dict_path: Optional[str] = None,  # 新增
)
```
- 支持直接传入词典文件路径
- 自动合并传入的词表和文件词表

### 📊 性能对比

| 配置组合 | 响应时间 | 准确率 | 推荐场景 |
|---------|---------|--------|---------|
| reranker: false<br>temperature: 0.1 | **快** (基准) | 中高 | 开发阶段、快速原型 |
| reranker: true<br>temperature: 0.1 | 慢 2-3x | **高** | 生产环境、复杂检索 |
| reranker: false<br>temperature: 0.3 | 快 | 中 | 创意生成、开放问答 |

### 🎯 使用建议

#### 快速开发模式
```yaml
reranker:
  enabled: false          # 速度优先
llm:
  generation:
    temperature: 0.1      # 保持精准
```

#### 生产部署模式
```yaml
reranker:
  enabled: true           # 精度优先
  top_n: 3
llm:
  generation:
    temperature: 0.1      # 事实型问答
```

### 🔄 迁移指南

#### 现有用户升级步骤
1. 拉取最新代码
2. 查看 `config/custom_dict.txt` 新增词汇
3. 根据领域添加自己的专有名词
4. 根据需求调整 `config/config.yaml`:
   - 需要速度：保持 `reranker.enabled: false`
   - 需要精度：改为 `reranker.enabled: true`
5. 重新入库测试文档，观察分块质量输出

#### 无需修改的部分
- 现有入库脚本调用方式不变
- API 接口无变更
- 数据库结构无变更

### ⚠️ 注意事项

1. **词典生效时机**: 自定义词典在入库时生效，需重新入库才能应用到已有文档
2. **Reranker 性能**: 开启 reranker 会显著增加响应时间，请根据实际需求权衡
3. **Temperature 调整**: 降低 temperature 会使回答更保守，适合事实型问答
4. **分块样例数量**: 入库时固定显示 5 个样例，如需更多可使用 `check_chunks.py` 工具

### 🐛 Bug 修复

无（本次为纯优化变更）

### 📝 技术债务

无新增技术债务

---

## 未来优化方向

### 计划中（待实施）

1. **动态词典更新**: 支持在 UI 中添加词汇，无需重启
2. **分块质量评分**: 自动计算分块质量分数（语义完整性、长度合理性等）
3. **智能参数推荐**: 根据文档类型自动推荐 chunk_size 和 overlap
4. **A/B 测试框架**: 自动对比不同配置的检索效果
5. **分块可视化**: 在 UI 中展示文档分块结果，支持手动调整

### 社区建议收集

欢迎在 Issue 中提出优化建议！

---

**最后更新**: 2026-09-20  
**版本**: v0.2.0  
**贡献者**: @Dtou
