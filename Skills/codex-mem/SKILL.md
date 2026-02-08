---
name: codex-mem
description: Codex 专属长期记忆技能。提供 Session 生命周期记录、三层渐进检索（search/timeline/get_observations）以及与 repo_knowledge 的融合检索 ask。
---

# Codex-Mem Skill

## 北极星

让 Codex 在新会话里不再“失忆”，并且只注入必要上下文，降低 token 消耗，同时保持代码事实准确性。

## 何时使用

- 需要跨会话保留项目决策、排障轨迹、已完成工作、后续计划
- 需要在大仓库中做“先粗后细”的上下文检索，避免一次性塞满窗口
- 需要把历史记忆与仓库实时代码检索合并回答

## 接入方式（MCP）

1. 启动 MCP server（stdio）：

```bash
python3 /ABS/PATH/hopeNote/Scripts/codex_mem_mcp.py --root /ABS/PATH/hopeNote --project-default hopenote
```

2. 注册到 Codex（示例命令，按你的本地路径替换）：

```bash
codex mcp add codex-mem -- python3 /ABS/PATH/hopeNote/Scripts/codex_mem_mcp.py --root /ABS/PATH/hopeNote --project-default hopenote
```

3. 验证：
- `tools/list` 应出现 `mem_search / mem_timeline / mem_get_observations / mem_ask`
- 调用 `mem_session_start` + `mem_user_prompt_submit` 成功写入本地库

## 推荐工作流（三层渐进）

1. `mem_search(query)`  
拿到紧凑结果（ID+标题+类型+分数），先做候选集。

2. `mem_timeline(id)`  
围绕候选 ID 看上下文邻域，确认是否真相关。

3. `mem_get_observations(ids)`  
仅对确认目标拉全量细节。

4. `mem_ask(question)`  
自动融合 Memory + `repo_knowledge` 代码片段，输出建议上下文 prompt。

## 生命周期钩子建议

- 会话开始：`mem_session_start`
- 用户发问：`mem_user_prompt_submit`
- 每次工具后：`mem_post_tool_use`
- 暂停：`mem_stop`
- 会话结束：`mem_session_end`

## 安全与隐私

- 数据仅本地：`<repo>/.codex_mem/codex_mem.sqlite3`
- 给 `mem_post_tool_use` 传 `tags` 包含以下任一值会跳过记录：  
  `no_mem`, `private`, `sensitive`, `secret`
