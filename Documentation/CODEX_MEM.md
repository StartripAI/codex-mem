# Dev-Mem Operational Guide

This guide describes production-style operation of `dev-mem` in IDE/CLI workflows.

## 1) Objectives

- Preserve engineering memory across sessions
- Reduce repeated context boilerplate
- Keep retrieval token-efficient via progressive disclosure
- Ground outputs in memory evidence + live repository evidence

## 2) Components

- `Scripts/codex_mem.py`: core engine and CLI
- `Scripts/codex_mem_mcp.py`: MCP server
- `Scripts/codex_mem_web.py`: local web viewer
- `Scripts/dev_mem.sh`: shell wrapper
- `Scripts/repo_knowledge.py`: code retrieval engine used by `ask`
- `Skills/dev-mem/`: Codex skill package

## 2.5) Cross-Repo Entrypoint (Hard Requirement)

For target repositories outside this repo, execute only via `run-target` with explicit target root:

```bash
bash /ABS/PATH/TO/dev-mem/Scripts/dev_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "learn this project: north star, architecture, module map, entrypoint, main flow, persistence, ai generation, risks."
```

Hard rules:
- do not bypass `dev_mem.sh run-target` in cross-repo runs
- do not omit explicit target root path
- do not return non-executable guidance

One-click natural-language entry:

```bash
bash /ABS/PATH/TO/dev-mem/Scripts/dev_mem.sh \
  run-target-auto "learn this project: north star, architecture, module map, entrypoint, main flow, persistence, ai generation, risks"
```

When root detection fails, output token is:

```text
TARGET_ROOT_REQUIRED
```

Single-model runtime switch (per run):

```bash
bash /ABS/PATH/TO/dev-mem/Scripts/dev_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --executor codex \
  --question "learn this project: architecture, persistence, risks"
```

## 3) Initialization

```bash
bash Scripts/dev_mem.sh init --project my-project
```

Data location:
- `<repo>/.codex_mem/codex_mem.sqlite3`

## 4) Lifecycle Capture Pattern

Recommended sequence per session:

1. `session-start`
2. `user-prompt-submit`
3. `post-tool-use` (repeat)
4. `stop` (optional)
5. `session-end`

Example:

```bash
bash Scripts/dev_mem.sh session-start s100 --project my-project --title "Refactor stream pipeline"
bash Scripts/dev_mem.sh prompt s100 "Map current bottlenecks and propose safe migration order" --project my-project
bash Scripts/dev_mem.sh tool s100 shell "rg -n 'stream' App" --project my-project --compact
bash Scripts/dev_mem.sh stop s100 --project my-project --content "checkpoint after search"
bash Scripts/dev_mem.sh session-end s100 --project my-project
```

## 5) Progressive Retrieval Workflow

### Layer 1: compact retrieval

```bash
bash Scripts/dev_mem.sh search "streaming orchestration" --project my-project --limit 20
```

### Layer 1 (NL): mem-search

```bash
bash Scripts/dev_mem.sh mem-search "what bugs were fixed this week" --project my-project --limit 20
```

### Layer 2: timeline neighborhood

```bash
bash Scripts/dev_mem.sh timeline E42 --before 5 --after 5
```

### Layer 3: full details

```bash
bash Scripts/dev_mem.sh get E42 O8 O9
```

## 6) Fused Retrieval (Memory + Repo)

```bash
bash Scripts/dev_mem.sh ask "What is the stream update chain from input to persisted output?" --project my-project
```

`ask` fuses:
- memory shortlist + detail records
- top-k repository chunks from `repo_knowledge.py`
- token estimate breakdown
- graph-lite persisted neighbor priors (`graph_lite_edges` in sqlite)
- onboarding coverage recovery loop (`--coverage-retry-max`)

Example with executor + recovery controls:

```bash
bash Scripts/dev_mem.sh run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --executor codex \
  --question "learn this project: architecture, persistence, risks" \
  -- --coverage-retry-max 3
```

## 7) Runtime Modes (Stable/Beta)

Read runtime config:

```bash
bash Scripts/dev_mem.sh config-get
```

Update runtime config:

```bash
bash Scripts/dev_mem.sh config-set --channel beta --viewer-refresh-sec 2 --beta-endless-mode on
```

Behavior summary:
- `stable`: only explicit compaction (`--compact`)
- `beta` + endless mode: auto-compaction for heavy tool outputs

## 8) Dual-Tag Privacy Model

`post-tool-use` supports two tag dimensions:

- semantic tags via `--tag`
- privacy tags via `--privacy-tag`

Privacy controls:
- block write: `no_mem`, `block`, `skip`, `secret_block`
- private visibility: `private`, `sensitive`, `secret`
- redaction: `redact`, `mask`, `sensitive`, `secret`

Example:

```bash
bash Scripts/dev_mem.sh tool s100 shell "credential=<REDACTED_VALUE>" --project my-project \
  --tag auth --privacy-tag private --privacy-tag redact
```

Default retrieval excludes private records unless `--include-private` is set.

## 9) Local Web Viewer

Start:

```bash
bash Scripts/dev_mem.sh web --project-default my-project --host 127.0.0.1 --port 37777
```

Capabilities:
- real-time memory stream
- session summaries
- natural language mem-search
- runtime config controls (stable/beta, refresh interval, endless mode)

## 10) MCP Integration

Start MCP server:

```bash
python3 Scripts/codex_mem_mcp.py --root . --project-default my-project
```

Register with Codex:

```bash
codex mcp add dev-mem -- python3 /ABS/PATH/TO/dev-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/TO/dev-mem --project-default my-project
```

Key tools:
- retrieval: `mem_search`, `mem_nl_search`, `mem_timeline`, `mem_get_observations`, `mem_ask`, `mem_export_session`
- config: `mem_config_get`, `mem_config_set`
- lifecycle: `mem_session_start`, `mem_user_prompt_submit`, `mem_post_tool_use`, `mem_stop`, `mem_session_end`, `mem_summarize_session`

## 11) Session Export for Sharing

Export one session (anonymized by default):

```bash
bash Scripts/dev_mem.sh export-session s100 --anonymize on --output /tmp/s100_export.json
```

Use cases:
- issue reproduction handoff
- support debugging package
- sanitized knowledge sharing

## 12) Validation

Run comprehensive simulation:

```bash
python3 Scripts/codex_mem_smoketest.py --root .
```

Validation coverage:
- lifecycle hooks
- NL mem-search
- privacy block/private/redact behaviors
- config changes
- web API
- MCP tools

## 13) Known Constraints

- Local hash-based vectors are lightweight by design, not external embedding APIs
- Web viewer is local and unauthenticated by default (bind to loopback host)
- Fused `ask` assumes `Scripts/repo_knowledge.py` is present
- Executor mode is strictly single-model per run (`none|codex|claude`)
