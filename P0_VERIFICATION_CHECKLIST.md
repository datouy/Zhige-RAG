# P0 级修复验证清单

## 📋 验证项目

### 1️⃣ P0-1: 死代码清理验证

- [ ] **CI 配置检查**
  ```bash
  grep -n "ignore" .github/workflows/ci.yml
  # 应该没有 --ignore=tests/test_agent.py 或 --ignore=tests/test_rag_pipeline.py
  ```

- [ ] **测试运行验证**
  ```bash
  python -m pytest tests/test_agent.py -v
  # 应该运行 30+ 个 agent 测试
  
  python -m pytest tests/test_integration.py -v
  # 应该运行 26+ 个集成测试
  ```

- [ ] **导入检查**
  ```bash
  grep -r "from src.executor.llm_executor" . --exclude-dir=.git
  # 应该没有任何引用
  ```

**预期结果**: ✅ 无测试忽略，无死代码引用

---

### 2️⃣ P0-2: 限流修复验证

- [ ] **可信代理逻辑测试**
  ```bash
  python -m pytest tests/test_rate_limit.py -v
  # 应该全部通过（5/5）
  ```

- [ ] **模拟反向代理测试**
  ```bash
  # 测试 1: 可信代理传递真实 IP
  curl -X POST http://localhost:8000/api/v1/auth/login \
    -H "Content-Type: application/json" \
    -H "X-Forwarded-For: 203.0.113.5" \
    -d '{"login":"test","password":"wrong"}'
  
  # 测试 2: 公网 IP 伪造 X-Forwarded-For（应被忽略）
  # 从公网直接访问时，X-Forwarded-For 应被忽略
  ```

- [ ] **限流端点验证**
  ```bash
  # /login: 每分钟 5 次
  for i in {1..10}; do
    curl -X POST http://localhost:8000/api/v1/auth/login \
      -H "Content-Type: application/json" \
      -d '{"login":"test","password":"wrong"}' \
      -w "\nStatus: %{http_code}\n"
    sleep 1
  done
  # 应该在第 6 次返回 429 Too Many Requests
  
  # /refresh: 每分钟 10 次（需要有效 refresh_token）
  # /logout: 每分钟 30 次（需要认证）
  ```

- [ ] **环境变量配置**
  ```bash
  echo $TRUSTED_PROXIES
  # 应该包含内网网段，例如: 10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
  ```

**预期结果**: ✅ 限流正常工作，X-Forwarded-For 伪造被防护

---

### 3️⃣ P0-3: 测试覆盖验证

- [ ] **全量测试运行**
  ```bash
  python -m pytest tests/ -v --tb=no
  # 应该运行全部 377 个测试，无跳过
  ```

- [ ] **覆盖率检查**
  ```bash
  python -m pytest tests/ --cov=src --cov=api --cov-report=term-missing
  # 应该达到 90%+ 覆盖率
  ```

- [ ] **关键模块测试**
  ```bash
  # Agent 测试
  python -m pytest tests/test_agent.py -v
  # 应该 30/30 通过
  
  # 集成测试
  python -m pytest tests/test_integration.py -v
  # 应该 26/26 通过
  
  # 认证测试
  python -m pytest tests/test_auth.py -v
  # 应该全部通过
  ```

- [ ] **双模式兼容性**
  ```bash
  # 本地模式（LOCAL_MODE=true）
  export LOCAL_MODE=true
  python -m pytest tests/test_integration.py -v
  
  # 多租户模式（LOCAL_MODE=false）
  export LOCAL_MODE=false
  python -m pytest tests/test_integration.py -v
  
  # 两种模式都应该通过
  ```

**预期结果**: ✅ 377/377 测试通过，覆盖率 95%+

---

## 🧪 自动验证脚本

### Windows (PowerShell)

```powershell
# verify_p0_fixes.ps1
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  P0 级修复验证" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# P0-1: 检查 CI 配置
Write-Host "1️⃣ 验证 P0-1: 死代码清理..." -ForegroundColor Yellow
$ciContent = Get-Content .github\workflows\ci.yml -Raw
if ($ciContent -match "--ignore=tests/test_agent.py") {
    Write-Host "❌ 失败: CI 仍然忽略 test_agent.py" -ForegroundColor Red
} else {
    Write-Host "✅ 通过: CI 不再忽略核心测试" -ForegroundColor Green
}

# P0-2: 运行限流测试
Write-Host ""
Write-Host "2️⃣ 验证 P0-2: 限流修复..." -ForegroundColor Yellow
python -m pytest tests/test_rate_limit.py -v --tb=no
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 通过: 限流测试全部通过" -ForegroundColor Green
} else {
    Write-Host "❌ 失败: 限流测试有失败" -ForegroundColor Red
}

# P0-3: 运行全量测试
Write-Host ""
Write-Host "3️⃣ 验证 P0-3: 测试覆盖..." -ForegroundColor Yellow
python -m pytest tests/ -v --tb=no -q 2>&1 | Select-String "passed"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 通过: 全量测试通过" -ForegroundColor Green
} else {
    Write-Host "❌ 失败: 有测试失败" -ForegroundColor Red
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  验证完成" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
```

### Linux/Mac (Bash)

```bash
#!/bin/bash
# verify_p0_fixes.sh

echo "====================================="
echo "  P0 级修复验证"
echo "====================================="
echo ""

# P0-1: 检查 CI 配置
echo "1️⃣ 验证 P0-1: 死代码清理..."
if grep -q "\-\-ignore=tests/test_agent.py" .github/workflows/ci.yml; then
    echo "❌ 失败: CI 仍然忽略 test_agent.py"
    exit 1
else
    echo "✅ 通过: CI 不再忽略核心测试"
fi

# P0-2: 运行限流测试
echo ""
echo "2️⃣ 验证 P0-2: 限流修复..."
python -m pytest tests/test_rate_limit.py -v --tb=no
if [ $? -eq 0 ]; then
    echo "✅ 通过: 限流测试全部通过"
else
    echo "❌ 失败: 限流测试有失败"
    exit 1
fi

# P0-3: 运行全量测试
echo ""
echo "3️⃣ 验证 P0-3: 测试覆盖..."
python -m pytest tests/ -v --tb=no
if [ $? -eq 0 ]; then
    echo "✅ 通过: 全量测试通过"
else
    echo "❌ 失败: 有测试失败"
    exit 1
fi

echo ""
echo "====================================="
echo "  验证完成 ✅"
echo "====================================="
```

---

## 📊 验证报告模板

```
P0 级修复验证报告
验证时间: ____________________
验证人员: ____________________

P0-1: 死代码清理
□ CI 配置正确
□ Agent 测试运行
□ 集成测试运行
□ 无死代码引用
结论: ____________________

P0-2: 限流修复
□ 限流测试通过
□ 可信代理识别正确
□ X-Forwarded-For 防护生效
□ 环境变量已配置
结论: ____________________

P0-3: 测试覆盖
□ 全量测试通过
□ 覆盖率达标
□ 双模式兼容
□ 关键模块测试通过
结论: ____________________

总体评估:
□ 全部通过，可进入下一阶段
□ 部分失败，需要返工

签名: ____________________
日期: ____________________
```

---

## 🚀 后续步骤

验证通过后：
1. ✅ 代码审查
2. ✅ 提交代码: `commit_p0_fixes.bat` / `bash commit_p0_fixes.sh`
3. ✅ 推送到远程: `git push`
4. ✅ 部署到测试环境
5. ✅ 生产环境配置 `TRUSTED_PROXIES`
6. ✅ 最终验收测试

---

生成时间: 2026-09-24 14:48
