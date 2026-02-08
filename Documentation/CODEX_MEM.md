# Codex-Mem (Codex 专属长期记忆层)

## 目标

在本地为 Codex 提供跨会话长期记忆，并和 `repo_knowledge` 融合，降低重复上下文 token 开销。

核心能力：
- 5 生命周期钩子：`session-start / user-prompt-submit / post-tool-use / stop / session-end`
- 三层渐进检索：
  - `search`（紧凑索引层）
  - `timeline`（时序上下文层）
  - `get-observations`（完整详情层）
- 融合检索：`ask` 同时拉 Memory + Repo 代码上下文
- MCP 直连：`Scripts/codex_mem_mcp.py` 提供 `mem_*` 工具
- Skill 封装：`Skills/codex-mem/SKILL.md`

## 快速开始

```bash
bash Scripts/codex_mem.sh init --project hopenote
```

写入一个完整会话：

```bash
bash Scripts/codex_mem.sh session-start s1 --project hopenote --title "S1"
bash Scripts/codex_mem.sh prompt s1 "先阅读 readmefirst.md 并梳理主流程" --project hopenote
bash Scripts/codex_mem.sh tool s1 shell "grep HomeTabView+Logic.swift" --project hopenote --title "检索 Home 入口" --compact
bash Scripts/codex_mem.sh stop s1 --project hopenote --content "阶段性停止"
bash Scripts/codex_mem.sh session-end s1 --project hopenote
```

三层检索：

```bash
# Stage 1
bash Scripts/codex_mem.sh search "home streaming orchestrator" --project hopenote --limit 20

# Stage 2
bash Scripts/codex_mem.sh timeline E12 --before 5 --after 5

# Stage 3
bash Scripts/codex_mem.sh get E12 O3
```

融合昨天的 repo 检索：

```bash
bash Scripts/codex_mem.sh ask "Home streaming 的入口和状态更新链路" --project hopenote

# 或沿用原入口
bash Scripts/repo_knowledge.sh ask-plus "Home streaming 的入口和状态更新链路" --project hopenote
```

MCP 挂载（Codex）：

```bash
codex mcp add codex-mem -- python3 /ABS/PATH/hopeNote/Scripts/codex_mem_mcp.py --root /ABS/PATH/hopeNote --project-default hopenote
```

## 数据与隐私

- 数据仅存本地：`<repo>/.codex_mem/codex_mem.sqlite3`
- 存储结构：`sessions / events / observations + FTS5`
- 向量为本地哈希语义向量（无外部模型依赖）

## 设计说明

- `search` 默认输出 ID + 标题 + 类型 + 分数，控制 token。
- `timeline` 只在选中目标后扩展邻域上下文。
- `get-observations` 才拉完整内容，避免一次性塞满上下文窗口。
- `ask` 将 Memory 三层检索与 `Scripts/repo_knowledge.py query --json` 结果融合，并返回 token 估算。
