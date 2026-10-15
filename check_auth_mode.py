#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查当前认证模式（本地单用户 vs 多用户）"""
import sys
import io
from pathlib import Path

# 修复 Windows 终端编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.middleware.auth import auth_enabled
from src.utils import load_config

def main():
    enabled = auth_enabled()
    cfg = load_config("config/config.yaml")
    
    print("=" * 50)
    print("知阁认证模式检查")
    print("=" * 50)
    
    if enabled:
        print("\n✓ 当前模式: 多用户认证模式")
        print("   - 用户需要注册/登录才能使用")
        print("   - 每个用户的数据相互隔离")
        print("\n如需切换到本地单用户模式:")
        print("   1. 修改 config/config.yaml: auth.enabled: false")
        print("   2. 或设置环境变量: AUTH_ENABLED=0")
    else:
        print("\n✓ 当前模式: 本地单用户模式（开箱即用）")
        print("   - 用户无需注册/登录")
        print("   - 所有数据归属本地用户")
        print("   - 无配额限制")
        print("\n如需切换到多用户模式:")
        print("   1. 修改 config/config.yaml: auth.enabled: true")
        print("   2. 或设置环境变量: AUTH_ENABLED=1")
    
    print("\n配置文件: config/config.yaml")
    print(f"   auth.enabled = {cfg.get('auth', {}).get('enabled', False)}")
    print("=" * 50)

if __name__ == "__main__":
    main()
