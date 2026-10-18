# 🎉 知阁 · 开箱即用指南

**无需登录！无需配置！拉下来就能用！**

---

## 🚀 4 步启动

### 第一步：安装项目

Windows 用户可以直接双击 `setup.bat`，选择“快速安装”。

也可以手动安装：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements-quick.txt
```

### 第二步：下载模型（推荐 Ollama，无需 GPU）

```bash
# 1. 安装 Ollama（一个命令，无需 GPU）
# 访问 https://ollama.com/download

# 2. 下载模型（运行一次即可，后续不用再跑）
ollama pull qwen3:4b-instruct-2507-q4_K_M  # 非思考指令模型，约 2.5 GB
ollama pull nomic-embed-text  # 向量模型（用于语义搜索）
```

**为什么推荐 Ollama？**
- ✅ 不需要 GPU，CPU 就能跑
- ✅ 不需要下载 GB 级权重文件
- ✅ 完全免费，无需 API Key
- ✅ 完全离线可用

### 第三步：启动服务

```bash
# 进入项目目录
cd Zhige-RAG

# 方式 A：一键启动（推荐）
python scripts\quickstart.py

# 方式 B：如果模型不可用，会给出提示，按提示配置后重试
```

### 第四步：打开浏览器

```
http://localhost:8000/
```

**然后就可以直接上传文档、提问了！** 

---

## 📱 你会看到的界面

### 首次访问
1. **欢迎界面** - 显示"欢迎使用知阁"
2. 点击 **"开始使用"** 按钮
3. 进入主界面

### 主界面功能
- 📄 **上传文档** - 支持 PDF、Word、Markdown、TXT
- 💬 **智能问答** - 基于你的文档回答问题
- 🌐 **网页导入** - 直接抓取网页内容
- 📚 **多知识库** - 按主题分类管理文档

---

## ⚙️ 高级配置（可选）

### 切换到云端 API

如果你有 API Key，想要更好的回答质量：

```bash
# 设置环境变量后启动
$env:OPENAI_API_KEY = "你的-api-key"
python scripts\quickstart.py
```

或者在界面中点击左下角 **"模型设置"** 进行配置。

---

## 🔧 常见问题

### Q: 启动报错 "缺少 xxx"
```bash
# 安装依赖
pip install -r requirements-quick.txt
```

### Q: Ollama 模型下载慢
```bash
# 使用国内镜像（如果有的话）
# 或者使用云端 API
```

### Q: 浏览器打不开 http://localhost:8000/
```bash
# 确保后端正在运行（看到 "Uvicorn running on http://0.0.0.0:8000"）
```

### Q: 需要登录吗？
- 不需要。本服务没有任何登录 / 注册入口，打开就能用
- 所有数据归属固定的「本地用户」，保存在本机

---

## 📁 项目结构

```
Zhige-RAG/
├── config/
│   └── config.yaml      # 配置文件（默认无需修改）
├── ui/
│   └── web/             # Web 前端界面
│       ├── index.html  # 主页面
│       ├── upload.html # 上传页面
│       └── docs.html   # 使用文档
├── api/                # 后端 API
├── src/                # 核心代码
├── data/               # 文档数据存储
└── scripts/
    └── quickstart.py   # 一键启动脚本
```

---

## 🎯 隐私说明

- **所有数据都保存在你的电脑上**
- 不上传到任何服务器
- 不需要登录账号
- 不收集任何使用信息

---

## 📞 获取帮助

- 查看完整文档: `README.md`
- 诊断问题: `python scripts/doctor.py`

---

**祝你使用愉快！🎉**
