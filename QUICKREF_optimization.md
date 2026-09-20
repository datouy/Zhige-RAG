# 优化任务快速参考卡

## 快速命令

### 测试优化功能
```bash
python scripts/test_optimization.py
```

### 入库并查看分块质量
```bash
python scripts/ingest.py data/sample.pdf
```

### 查看已入库的分块
```bash
python scripts/check_chunks.py --user test_user --limit 5
```

---

## 核心配置

### 快速响应（推荐开发）
```yaml
# config/config.yaml
reranker:
  enabled: false  # 速度提升 2-3x
llm:
  generation:
    temperature: 0.1  # 精准回答
```

### 高精度（推荐生产）
```yaml
# config/config.yaml
reranker:
  enabled: true   # 最高准确率
  top_n: 3
llm:
  generation:
    temperature: 0.1
```

---

## 自定义词典

**位置**: `config/custom_dict.txt`

**添加词汇**:
1. 打开文件
2. 在对应分类下添加（每行一词）
3. 重新入库文档生效

**示例**:
```
# 医疗领域
冠状动脉
心肌梗死
血压监测
```

---

## 优化成果

✅ 词典扩展: 42 → 230+ 词  
✅ 响应速度: 提升 2-3 倍  
✅ 回答精准度: Temperature 0.15 → 0.1  
✅ 分块质量: 自动检查输出  
✅ 开发体验: 词典自动加载  

---

## 文档索引

- [README_optimization.md](README_optimization.md) - 优化指南
- [CHANGELOG_optimization.md](CHANGELOG_optimization.md) - 变更日志
- [SUMMARY_optimization.md](SUMMARY_optimization.md) - 完成报告
- [TODO_optimization.md](TODO_optimization.md) - 任务清单

---

**版本**: v0.2.0 | **日期**: 2026-09-20
