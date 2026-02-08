---
name: codex-mem
description: Persistent memory skill for Codex with progressive retrieval, mem-search, local viewer, and dual-tag privacy controls.
---

# Codex-Mem Skill

## Mission

Keep Codex sessions continuous, retrieval-efficient, and code-grounded.

## Trigger Conditions

Use this skill when work needs:
- cross-session continuity
- natural-language memory lookup (mem-search)
- progressive retrieval instead of full-history stuffing
- fused memory + repository context

## MCP Setup

Start server:

```bash
python3 /ABS/PATH/codex-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/codex-mem --project-default my-project
```

Register in Codex:

```bash
codex mcp add codex-mem -- python3 /ABS/PATH/codex-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/codex-mem --project-default my-project
```

## Recommended Retrieval Sequence

1. `mem_search` or `mem_nl_search`
2. `mem_timeline`
3. `mem_get_observations`
4. `mem_ask` when repository grounding is needed

## Recommended Lifecycle Capture Sequence

1. `mem_session_start`
2. `mem_user_prompt_submit`
3. `mem_post_tool_use` after meaningful tool actions
4. `mem_stop` optional checkpoint
5. `mem_session_end`

## Runtime Mode Controls

- `mem_config_get`
- `mem_config_set` (`stable`/`beta`, refresh interval, endless mode)

## Privacy Policy Rules

- semantic labels: `tags`
- policy labels: `privacy_tags`

Policy labels:
- block write: `no_mem`, `block`, `skip`, `secret_block`
- private visibility: `private`, `sensitive`, `secret`
- redaction: `redact`, `mask`, `sensitive`, `secret`

Default retrieval excludes private records unless explicitly requested.
