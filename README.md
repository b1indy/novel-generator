# 网络小说生成器

基于 DeepSeek-v4-Flash（1M 上下文）的网络小说生成器，按卷生成，配套记忆系统与双维度审计，支撑 200 万字以上超长篇小说创作。

## 特性

- **按卷生成** — 利用 1M 上下文窗口，每卷 10-20 章内逻辑通畅，跨卷通过小结+末两章衔接
- **记忆系统** — 角色表、物品表、伏笔表实时追踪，每章生成后自动更新，杜绝剧情 bug
- **双维度审计** — 单次 LLM 调用同时检查逻辑一致性与 AI 写作痕迹，节省 token
- **灵活大纲** — 总大纲 + 卷大纲两级结构，写作中途可优化调整
- **风格配置** — 内置 5 种作家风格预设 + 6 种小说类型，支持自定义
- **每章标题 + 导语** — 大纲生成后附带约 100 字导语简介

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`，填入 DeepSeek API key：

```yaml
llm:
  api_key: "sk-your-deepseek-api-key"
  base_url: "https://api.deepseek.com"
  model: "deepseek-chat"
```

### 3. 创建小说

```bash
python cli.py new
```

交互式引导：设定书名 → 选择类型/风格 → 生成总大纲 → 生成第一卷大纲 → 开始写作。

## 命令说明

| 命令 | 说明 |
|------|------|
| `python cli.py new` | 创建新小说（交互式引导） |
| `python cli.py list` | 列出所有小说 |
| `python cli.py outline --novel <名称>` | 查看大纲 |
| `python cli.py outline --novel <名称> --volume 3` | 生成第 3 卷大纲 |
| `python cli.py write --novel <名称> --volume 1` | 生成第 1 卷 |
| `python cli.py audit --novel <名称> --volume 1` | 审计第 1 卷 |
| `python cli.py audit --novel <名称> --volume 1 --fix` | 审计并自动修复 |
| `python cli.py status --novel <名称>` | 查看进度与状态 |
| `python cli.py continue --novel <名称>` | 继续生成下一卷 |

## 内置风格

### 作家风格
| 风格 | 特点 |
|------|------|
| 辰东式热血 | 快节奏，宏大世界，悲壮英雄主义 |
| 猫腻式文艺 | 文学性强，哲学思辨，人物复杂 |
| 番茄式爽文 | 实力进阶快，快意恩仇，节奏极快 |
| 老鹰式细节 | 细腻世界构建，政治博弈，慢热 |
| 唐家三少式 | 感情线丰富，等级体系清晰，团队作战 |

### 小说类型
玄幻 · 都市 · 科幻 · 仙侠 · 历史 · 悬疑

## 项目结构

```
src/
├── llm/          # DeepSeek API 客户端（流式/重试/token计数）
├── config/       # 风格配置
├── outline/      # 大纲生成与管理
├── generator/    # 小说生成（单章 + 整卷编排）
├── auditor/      # 双维度审计Agent
├── memory/       # 记忆系统（角色/物品/伏笔/卷小结）
└── storage/      # 存储层（Markdown + JSON）
```

## 工作流程

```
设定风格 → 生成总大纲 → 生成卷大纲(+导语)
→ 逐章生成（更新记忆表）→ 整卷审计（逻辑+AI味）
→ 生成卷小结 → 下一卷（注入前卷上下文）→ 循环
```
