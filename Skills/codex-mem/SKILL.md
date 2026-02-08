---
name: codex-mem
description: Persistent memory skill for Codex with lifecycle capture, progressive retrieval, and fused memory+repo context.
---

# Codex-Mem Skill

## Mission

Give Codex reliable cross-session memory while keeping context token-efficient and code-grounded.

## When to trigger this skill

Use this skill when a workflow needs one or more of the following:
- Continuity across sessions (decisions, findings, completed work, next steps)
- Progressive memory retrieval instead of full-history stuffing
- Fused retrieval from memory + live repository code

## MCP setup

1. Start MCP server:

```bash
python3 /ABS/PATH/codex-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/codex-mem --project-default my-project
```

2. Register server in Codex:

```bash
codex mcp add codex-mem -- python3 /ABS/PATH/codex-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/codex-mem --project-default my-project
```

3. Verify tools:
- `mem_search`
- `mem_timeline`
- `mem_get_observations`
- `mem_ask`

## Recommended retrieval sequence

1. `mem_search(query)`
2. `mem_timeline(id)`
3. `mem_get_observations(ids)`
4. `mem_ask(question)` when code-level grounding is required

## Recommended lifecycle capture sequence

- `mem_session_start`
- `mem_user_prompt_submit`
- `mem_post_tool_use` (after each meaningful tool step)
- `mem_stop` (optional)
- `mem_session_end`

## Privacy rule

Skip memory writes for sensitive content by passing blocked tags:
- `no_mem`
- `private`
- `sensitive`
- `secret`

## Notes

- Memory is local-only by default (`.codex_mem/codex_mem.sqlite3`).
- Use `--compact` for large tool outputs to reduce storage and retrieval token cost.
