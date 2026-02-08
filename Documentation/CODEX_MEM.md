# Codex-Mem Operational Guide

This guide documents production-style usage of `codex-mem` for Codex workflows.

## 1) Goals

- Preserve useful engineering context across sessions.
- Reduce repeated prompt/context boilerplate.
- Keep retrieval token-efficient via progressive disclosure.
- Ground follow-up answers in both memory history and live code evidence.

## 2) Components

- `Scripts/codex_mem.py`: core local memory engine and CLI
- `Scripts/codex_mem_mcp.py`: MCP server exposing `mem_*` tools
- `Scripts/codex_mem.sh`: command wrapper
- `Scripts/repo_knowledge.py`: code retrieval engine used by fused `ask`
- `Skills/codex-mem/`: reusable skill package for Codex flows

## 3) Setup

```bash
bash Scripts/codex_mem.sh init --project my-project
```

Data location:
- `<repo>/.codex_mem/codex_mem.sqlite3`

## 4) Lifecycle Pattern

Recommended event sequence per session:

1. `session-start`
2. `user-prompt-submit`
3. `post-tool-use` (repeat)
4. `stop` (optional checkpoints)
5. `session-end`

Example:

```bash
bash Scripts/codex_mem.sh session-start s100 --project my-project --title "Refactor stream pipeline"
bash Scripts/codex_mem.sh prompt s100 "Review current state and isolate bottlenecks" --project my-project
bash Scripts/codex_mem.sh tool s100 shell "rg -n 'stream' App" --project my-project --compact
bash Scripts/codex_mem.sh stop s100 --project my-project --content "checkpoint after search"
bash Scripts/codex_mem.sh session-end s100 --project my-project
```

## 5) Progressive Retrieval Workflow

### Layer 1: Search

```bash
bash Scripts/codex_mem.sh search "streaming orchestration" --project my-project --limit 20
```

Returns compact index candidates with IDs and scores.

### Layer 2: Timeline

```bash
bash Scripts/codex_mem.sh timeline E42 --before 5 --after 5
```

Adds temporal context around selected IDs.

### Layer 3: Full observations

```bash
bash Scripts/codex_mem.sh get E42 O8 O9
```

Fetches full payload only for selected IDs.

## 6) Fused Retrieval (Memory + Repo)

```bash
bash Scripts/codex_mem.sh ask "What is the current stream update chain from input to persistence?" --project my-project
```

`ask` combines:
- memory context from this project
- code chunks from `repo_knowledge.py`
- token estimate breakdown

## 7) MCP Integration

### Start MCP server

```bash
python3 Scripts/codex_mem_mcp.py --root . --project-default my-project
```

### Register in Codex

```bash
codex mcp add codex-mem -- python3 /ABS/PATH/codex-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/codex-mem --project-default my-project
```

### Tools exposed

- `mem_search`
- `mem_timeline`
- `mem_get_observations`
- `mem_ask`
- `mem_session_start`
- `mem_user_prompt_submit`
- `mem_post_tool_use`
- `mem_stop`
- `mem_session_end`
- `mem_summarize_session`

## 8) Privacy Controls

`mem_post_tool_use` can skip writes when tags include blocked labels:
- `no_mem`
- `private`
- `sensitive`
- `secret`

Example:

```bash
bash Scripts/codex_mem.sh tool s100 shell "cat .env" --project my-project --tag private
```

## 9) Output Compaction for Heavy Logs

Use compaction on large tool outputs:

```bash
bash Scripts/codex_mem.sh tool s100 shell "npm test" --project my-project --compact --compact-chars 1200
```

Compaction stores:
- head slice
- signal lines (errors/warnings/failures/etc.)
- tail slice

## 10) Validation

Run smoke test:

```bash
python3 Scripts/codex_mem_smoketest.py --root .
```

Validation scope:
- lifecycle capture
- search/timeline/get
- MCP initialize and tool calls
- non-empty retrieval output

## 11) Distribution

For release and publishing workflow, see:
- `PUBLISH.md`

## 12) Known Constraints

- No web dashboard in current version (CLI + MCP only)
- Default vectors are local hash embeddings
- Fused ask expects `repo_knowledge.py` to be present in `Scripts/`
