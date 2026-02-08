# codex-mem

Codex 专属长期记忆系统（本地优先），包含：
- 生命周期记忆钩子（SessionStart/UserPromptSubmit/PostToolUse/Stop/SessionEnd）
- 三层渐进检索（`search -> timeline -> get-observations`）
- 与 `repo_knowledge` 融合检索（`ask` / `mem_ask`）
- MCP server（可直接挂到 Codex）
- Skill 包（可社区分发复用）

## 目录

- `Scripts/codex_mem.py`：核心 CLI
- `Scripts/codex_mem_mcp.py`：MCP server（stdio）
- `Scripts/codex_mem.sh`：快捷命令入口
- `Scripts/codex_mem_smoketest.py`：模拟端到端测试
- `Scripts/repo_knowledge.py`：代码仓检索（融合层依赖）
- `Skills/codex-mem/`：技能定义与 `openai.yaml`
- `Documentation/CODEX_MEM.md`：详细用法

## 快速开始

```bash
bash Scripts/codex_mem.sh init --project demo
bash Scripts/codex_mem.sh session-start s1 --project demo --title "Session 1"
bash Scripts/codex_mem.sh prompt s1 "先阅读 readmefirst.md" --project demo
bash Scripts/codex_mem.sh tool s1 shell "grep -n HomeViewModel.swift" --project demo --compact
bash Scripts/codex_mem.sh session-end s1 --project demo

bash Scripts/codex_mem.sh search "Home streaming orchestrator" --project demo
bash Scripts/codex_mem.sh ask "Home streaming 的入口和状态更新链路" --project demo
```

## MCP 挂载（Codex）

```bash
codex mcp add codex-mem -- python3 /ABS/PATH/codex-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/codex-mem --project-default demo
```

## 模拟测试

```bash
python3 Scripts/codex_mem_smoketest.py --root .
```

