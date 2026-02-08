# codex-mem

Persistent memory, progressive retrieval, and Codex-native workflow tooling.

`codex-mem` is a local-first memory layer for Codex that keeps cross-session context usable without stuffing entire history into every prompt.

It is inspired by the same pain point that made Claude-Mem popular, but implemented for the Codex toolchain (CLI, MCP, Skills, and local automation).

## North Star

Make every new Codex session feel like a continuation of real engineering work:
- keep durable memory of decisions, tool outputs, and outcomes
- retrieve only what is relevant (progressive disclosure)
- fuse memory with repository evidence for accurate follow-up answers

## What You Get

### Core capabilities
- Local-first persistence with SQLite + FTS5 + lightweight semantic vectors
- Five lifecycle hooks:
  - `session-start`
  - `user-prompt-submit`
  - `post-tool-use`
  - `stop`
  - `session-end`
- Three-stage retrieval:
  - `search` (compact IDs/titles/scores)
  - `timeline` (temporal neighborhood)
  - `get-observations` (full details by selected IDs)
- Fused retrieval via `ask`:
  - memory context + `repo_knowledge.py` code chunks

### UX features (parity-focused)
- Built-in natural language memory search:
  - `nl-search` / `mem-search`
  - supports queries like: "what bugs were fixed", "last week", "this week", "today"
- Local Web UI:
  - real-time memory stream
  - session summaries
  - natural-language search
  - runtime channel switch (`stable` / `beta`)
- Dual-tag privacy model:
  - semantic tags (`--tag`) for classification
  - privacy tags (`--privacy-tag`) for access policy
- Endless-mode style compaction (beta):
  - auto-compacts high-volume tool output to reduce stored and retrieved token load

### Codex integration
- MCP server (`Scripts/codex_mem_mcp.py`) with `mem_*` tools
- Skill package (`Skills/codex-mem/`) for reproducible retrieval workflow
- Shell wrapper (`Scripts/codex_mem.sh`) for short operational commands

## Repository Layout

- `Scripts/codex_mem.py`: core memory engine and CLI
- `Scripts/codex_mem_mcp.py`: MCP server
- `Scripts/codex_mem_web.py`: local web viewer
- `Scripts/codex_mem_smoketest.py`: end-to-end simulation test
- `Scripts/repo_knowledge.py`: repository retrieval used by fused `ask`
- `Skills/codex-mem/`: skill docs + agent config
- `Documentation/`: deep docs for install, architecture, MCP tools, and troubleshooting

## Quick Start

### 1) Initialize memory store

```bash
bash Scripts/codex_mem.sh init --project demo
```

### 2) Capture a session lifecycle

```bash
bash Scripts/codex_mem.sh session-start s1 --project demo --title "Streaming refactor"
bash Scripts/codex_mem.sh prompt s1 "Map pipeline from entry to persistence" --project demo
bash Scripts/codex_mem.sh tool s1 shell "rg -n 'HomeStreamOrchestrator'" --project demo --title "Find orchestrator" --compact
bash Scripts/codex_mem.sh stop s1 --project demo --content "checkpoint"
bash Scripts/codex_mem.sh session-end s1 --project demo
```

### 3) Retrieve progressively

```bash
# Layer 1: compact candidates
bash Scripts/codex_mem.sh search "orchestrator streaming" --project demo --limit 20

# Layer 1 (natural language): mem-search
bash Scripts/codex_mem.sh mem-search "what bugs were fixed today" --project demo --limit 20

# Layer 2: timeline around chosen item
bash Scripts/codex_mem.sh timeline E12 --before 5 --after 5

# Layer 3: full details for chosen IDs
bash Scripts/codex_mem.sh get E12 O3
```

### 4) Fuse memory with code retrieval

```bash
bash Scripts/codex_mem.sh ask "What is the current end-to-end generation path?" --project demo
```

## Web UI (Local Viewer)

Start viewer:

```bash
bash Scripts/codex_mem.sh web --project-default demo --host 127.0.0.1 --port 37777
```

Then open:
- `http://127.0.0.1:37777/`

Viewer provides:
- real-time memory stream (with visibility badge)
- session summary panel
- mem-search query box
- runtime config controls:
  - channel: `stable` / `beta`
  - refresh interval
  - `beta endless mode` toggle

## Dual-Tag Privacy Model

### Semantic tags
Use `--tag` for topic/category indexing and retrieval semantics.

Example:
```bash
bash Scripts/codex_mem.sh tool s1 shell "Fixed race in parser" --project demo --tag bugfix --tag parser
```

### Privacy tags
Use `--privacy-tag` for storage/visibility policy.

Supported policy behavior:
- block write:
  - `no_mem`, `block`, `skip`, `secret_block`
- mark record private:
  - `private`, `sensitive`, `secret`
- redact sensitive patterns:
  - `redact`, `mask`, `sensitive`, `secret`

Example:
```bash
bash Scripts/codex_mem.sh tool s1 shell "token=abc123" --project demo --privacy-tag private --privacy-tag redact
```

By default, retrieval hides private records. Use `--include-private` when needed.

## Runtime Config and Beta Endless Mode

Read config:
```bash
bash Scripts/codex_mem.sh config-get
```

Set config:
```bash
bash Scripts/codex_mem.sh config-set --channel beta --viewer-refresh-sec 2 --beta-endless-mode on
```

Behavior:
- `stable`: explicit compaction only (`--compact`)
- `beta` + endless mode `on`: auto-compaction for high-volume tool outputs

## MCP Server

Run server:

```bash
python3 Scripts/codex_mem_mcp.py --root . --project-default demo
```

Register in Codex:

```bash
codex mcp add codex-mem -- python3 /ABS/PATH/codex-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/codex-mem --project-default demo
```

### Exposed MCP tools

Retrieval:
- `mem_search`
- `mem_nl_search`
- `mem_timeline`
- `mem_get_observations`
- `mem_ask`

Runtime config:
- `mem_config_get`
- `mem_config_set`

Lifecycle:
- `mem_session_start`
- `mem_user_prompt_submit`
- `mem_post_tool_use`
- `mem_stop`
- `mem_session_end`
- `mem_summarize_session`

## Skill Integration

Skill location:
- `Skills/codex-mem/SKILL.md`

Recommended skill retrieval sequence:
1. `mem_search` or `mem_nl_search`
2. `mem_timeline`
3. `mem_get_observations`
4. `mem_ask` for memory+code grounded answers

## Token Strategy

Token savings come from retrieval discipline, not from aggressive truncation alone:
- Layer 1 IDs/summaries first
- timeline only around selected IDs
- full details only for final shortlist
- compact tool outputs on write path
- bound code context with `code_top_k` and module limits

This design keeps context quality while reducing unnecessary context payload.

## Data Model

Storage path:
- `<repo>/.codex_mem/codex_mem.sqlite3`

Tables:
- `meta`
- `sessions`
- `events`
- `observations`
- `events_fts`
- `observations_fts`

Important metadata fields:
- runtime config: `channel`, `viewer_refresh_sec`, `beta_endless_mode`
- privacy metadata per record: visibility, tags, redaction flags

## Simulation / Validation

Run one command:

```bash
python3 Scripts/codex_mem_smoketest.py --root .
```

Smoke test validates:
- lifecycle capture
- natural-language mem-search
- privacy blocking + private visibility filtering
- runtime config set/get
- web API endpoints
- MCP tool availability + retrieval calls

## Codex Plugin / Extension Path

Current practical integration path for Codex is:
- MCP server (`codex_mem_mcp.py`)
- Skills package (`Skills/codex-mem`)
- GitHub distribution + release notes

If an official extension marketplace flow is available in your Codex environment, keep this repo structured around:
- deterministic install commands
- explicit MCP registration snippet
- compatibility matrix in docs
- smoke test command for reviewers

## Roadmap

- Incremental compaction policies by tool type
- Optional external embedding provider adapters
- Browser-based timeline drilling and detail panes
- Export/import utilities for memory snapshots
- Release automation for reproducible installation bundles

## Documentation

- `Documentation/CODEX_MEM.md`
- `Documentation/INSTALLATION.md`
- `Documentation/MCP_TOOLS.md`
- `Documentation/ARCHITECTURE.md`
- `Documentation/TROUBLESHOOTING.md`
- `PUBLISH.md`

## Contributing

1. Create a feature branch from `codex/init`
2. Keep changes local-first and deterministic
3. Update docs with concrete command examples
4. Run `python3 Scripts/codex_mem_smoketest.py --root .`
5. Include smoke output summary in PR

## License

No explicit open-source license file is included yet. Add one before broad public redistribution.
