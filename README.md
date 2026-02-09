# codex-mem

Cut context tokens by **99.84%**  (median local benchmark).
Codex-native persistent memory with progressive retrieval, local viewer UX, and MCP-ready integration.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Token Savings](https://img.shields.io/badge/token_saving-99.84%25-16a34a)
![Startup](https://img.shields.io/badge/startup_to_first_context-59.5ms-0ea5e9)
![Startup Speedup](https://img.shields.io/badge/startup_speedup-1.32x-0891b2)
![MCP](https://img.shields.io/badge/MCP-ready-0A7C66)
![Runtime](https://img.shields.io/badge/runtime-stable%20%7C%20beta-0f766e)
![Smoke Test](https://img.shields.io/badge/smoke_test-passing-16a34a)
![Local First](https://img.shields.io/badge/storage-local--first-334155)
![License](https://img.shields.io/badge/license-not%20set-lightgrey)

[Quick Start](#quick-start) • [Prompt Playbook](Documentation/PROMPT_PLAYBOOK_EN.md) • [Comparison](#comparison-table) • [Release Notes](RELEASE_NOTES.md) • [Docs](Documentation/CODEX_MEM.md)

Benchmark source:
- [`Documentation/benchmarks/marketing_claims_20260208.json`](Documentation/benchmarks/marketing_claims_20260208.json)
- Reproduce with: `python3 Scripts/benchmark_marketing_claim.py --root . --out Documentation/benchmarks/marketing_claims_20260208.json`

Scenario savings (2026-02-09):
- Cold start (lean): **63.98%** token saving
- Cold start (deeper context): **72.26%** token saving
- Daily Q&A (standard): **99.84%** token saving
- Daily Q&A (deep retrieval): **99.70%** token saving
- Incident forensics (wide detail pull): **88.97%** token saving
- Full matrix: [`Documentation/benchmarks/scenario_savings_20260209.json`](Documentation/benchmarks/scenario_savings_20260209.json)

## Why codex-mem

Most coding assistants lose operational memory between sessions.  
`codex-mem` makes new Codex sessions feel continuous by capturing lifecycle evidence, retrieving context progressively, and fusing memory with live repository facts.

North star:
- less repeated explanation
- less wasted context tokens
- more accurate follow-up reasoning from real prior work

## Launch Asset Production Kit

- Playbook: [`Documentation/LAUNCH_ASSET_PLAYBOOK.md`](Documentation/LAUNCH_ASSET_PLAYBOOK.md)
- Asset root: [`Assets/LaunchKit/README.md`](Assets/LaunchKit/README.md)
- GIF shotlist template: [`Assets/LaunchKit/gif/spec/SHOTLIST_TEMPLATE.md`](Assets/LaunchKit/gif/spec/SHOTLIST_TEMPLATE.md)
- PRD screenshot copy template: [`Assets/LaunchKit/screenshots/prd-copy/PRD_SCREENSHOT_COPY_TEMPLATE.md`](Assets/LaunchKit/screenshots/prd-copy/PRD_SCREENSHOT_COPY_TEMPLATE.md)

## Core Capabilities

### Lifecycle capture
- five hooks:
  - `session-start`
  - `user-prompt-submit`
  - `post-tool-use`
  - `stop`
  - `session-end`
- automatic session summary observations at close

### Progressive disclosure retrieval
- Layer 1: `search` / `mem-search` (compact shortlist)
- Layer 2: `timeline` (temporal neighborhood)
- Layer 3: `get-observations` (full details by selected IDs)

### Fused memory + code grounding
- `ask` combines memory shortlist with code context from `repo_knowledge.py`

### UX and operations
- built-in natural-language search (`nl-search` / `mem-search`)
- local web viewer (stream, summary, search, config)
- runtime channel config (`stable` / `beta`)
- endless-mode style auto-compaction in beta
- dual-tag privacy model (`--tag`, `--privacy-tag`)

### Codex integrations
- MCP server with `mem_*` tools
- skill package for reusable retrieval workflow
- CLI wrapper for repeatable operations

## Comparison Table

| Capability | codex-mem | Basic session-only chat memory | Codex-Mem target parity with Claude-Mem-style workflow |
|---|---|---|---|
| Cross-session persistence | ✅ Local SQLite + FTS + vectors | ❌ | ✅ |
| 3-layer progressive retrieval | ✅ | ❌ | ✅ |
| Natural-language memory query | ✅ `mem-search` | ❌ | ✅ |
| Real-time local web viewer | ✅ | ❌ | ✅ |
| Stable/Beta runtime switch | ✅ | ❌ | ✅ |
| Endless-style compaction mode | ✅ (beta) | ❌ | ✅ |
| Dual-tag privacy controls | ✅ semantic + policy tags | ❌ | ✅ |
| MCP tool surface for Codex | ✅ | ❌ | ✅ |
| Smoke-testable install validation | ✅ | ❌ | ✅ |

## What’s New (Latest)

See full history in [RELEASE_NOTES.md](RELEASE_NOTES.md).

Highlights in `v0.3.0`:
- launch asset production toolkit (`Assets/LaunchKit/`)
- CI asset gate (`.github/workflows/asset-gate.yml`)
- mem export surface (`export-session`, `mem_export_session`)
- web viewer upgrades (PRD caption copy + recording guide mode)
- launch automation scripts (`make_gifs`, `validate_assets`, `social_pack`, `snapshot_docs`)
- benchmark + roadmap + compatibility + security documentation set

## Release Rhythm Template

Every release batch follows one fixed package:
- Release Notes update
- 3 GIFs (`gif_01`, `gif_02`, `gif_03`)
- 3 final screenshots
- 1 comparison table update in README

Reference:
- [`Documentation/RELEASE_RHYTHM.md`](Documentation/RELEASE_RHYTHM.md)

## Quick Start

### 1) Initialize memory

```bash
bash Scripts/codex_mem.sh init --project demo
```

### 2) Capture a full session lifecycle

```bash
bash Scripts/codex_mem.sh session-start s1 --project demo --title "Streaming refactor"
bash Scripts/codex_mem.sh prompt s1 "Map end-to-end generation and persistence path" --project demo
bash Scripts/codex_mem.sh tool s1 shell "rg -n 'HomeStreamOrchestrator'" --project demo --title "Locate orchestrator" --compact
bash Scripts/codex_mem.sh stop s1 --project demo --content "checkpoint"
bash Scripts/codex_mem.sh session-end s1 --project demo
```

### 3) Retrieve progressively

```bash
# Layer 1 compact search
bash Scripts/codex_mem.sh search "orchestrator streaming" --project demo --limit 20

# Layer 1 natural-language search
bash Scripts/codex_mem.sh mem-search "what bugs were fixed this week" --project demo --limit 20

# Layer 2 timeline
bash Scripts/codex_mem.sh timeline E12 --before 5 --after 5

# Layer 3 full details
bash Scripts/codex_mem.sh get E12 O3
```

### 4) Ask with memory + code fusion

```bash
bash Scripts/codex_mem.sh ask "What is the current generation chain from input to persisted output?" --project demo
```

## Local Viewer

Start:

```bash
bash Scripts/codex_mem.sh web --project-default demo --host 127.0.0.1 --port 37777
```

Open:
- `http://127.0.0.1:37777/`

Viewer panels:
- real-time memory stream
- session summaries
- NL mem-search results
- runtime mode controls (`stable` / `beta`, refresh interval, endless mode)

## Dual-Tag Privacy Model

`post-tool-use` supports two tag lanes:

- semantic tags: `--tag`
- privacy policy tags: `--privacy-tag`

Policy behavior:
- block write:
  - `no_mem`, `block`, `skip`, `secret_block`
- private visibility:
  - `private`, `sensitive`, `secret`
- redact sensitive values:
  - `redact`, `mask`, `sensitive`, `secret`

Example:

```bash
bash Scripts/codex_mem.sh tool s1 shell "credential=<REDACTED_VALUE>" --project demo \
  --tag auth \
  --privacy-tag private \
  --privacy-tag redact
```

Default retrieval hides private records unless `--include-private` is passed.

## Runtime Modes

Read:

```bash
bash Scripts/codex_mem.sh config-get
```

Set:

```bash
bash Scripts/codex_mem.sh config-set --channel beta --viewer-refresh-sec 2 --beta-endless-mode on
```

Mode behavior:
- `stable`: compaction only when explicitly requested (`--compact`)
- `beta` + endless mode `on`: auto-compaction for high-volume tool outputs

## MCP Server

Run:

```bash
python3 Scripts/codex_mem_mcp.py --root . --project-default demo
```

Register with Codex:

```bash
codex mcp add codex-mem -- python3 /ABS/PATH/codex-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/codex-mem --project-default demo
```

### MCP tools

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
- `mem_export_session`

## Repository Structure

- `Scripts/codex_mem.py` core engine and CLI
- `Scripts/codex_mem_mcp.py` MCP server
- `Scripts/codex_mem_web.py` local web app
- `Scripts/codex_mem_smoketest.py` end-to-end simulation
- `Scripts/repo_knowledge.py` repository context retrieval
- `Scripts/make_gifs.sh` media pipeline (source -> webm/gif/poster)
- `Scripts/validate_assets.py` asset + README gate checks
- `Scripts/load_demo_data.py` one-click demo dataset loader
- `Scripts/redact_screenshot.py` OCR-based screenshot redaction
- `Scripts/compare_search_modes.py` search vs mem-search comparison runner
- `Scripts/snapshot_docs.sh` release snapshot utility
- `Skills/codex-mem/` skill package
- `Documentation/` deep operational docs

## Validation

Run one command:

```bash
python3 Scripts/codex_mem_smoketest.py --root .
```

Smoke test verifies:
- lifecycle capture
- NL mem-search
- privacy block/private/redact handling
- stable/beta config updates
- web APIs
- MCP tool registration and calls

## Launch Ops Commands

```bash
# 1) load sanitized recording dataset
bash Scripts/load_demo_data.sh --reset

# 2) render GIF bundle from source clips
bash Scripts/make_gifs.sh --fps 12 --width 1200

# 3) validate media + README links
python Scripts/validate_assets.py --check-readme --strict

# 4) snapshot docs/media per release
bash Scripts/snapshot_docs.sh v0.3.0

# 5) generate social copy pack
python Scripts/generate_social_pack.py --version v0.3.0
```

## Token Efficiency Strategy

Token reduction comes from retrieval discipline:
- compact shortlist first
- timeline around selected IDs only
- full payload fetch only for shortlisted IDs
- bounded repo context for fused ask
- optional/automatic output compaction for verbose tool logs

## Documentation

- [Operational Guide](Documentation/CODEX_MEM.md)
- [Installation](Documentation/INSTALLATION.md)
- [MCP Tools](Documentation/MCP_TOOLS.md)
- [Architecture](Documentation/ARCHITECTURE.md)
- [Launch Asset Playbook](Documentation/LAUNCH_ASSET_PLAYBOOK.md)
- [Release Rhythm](Documentation/RELEASE_RHYTHM.md)
- [Compatibility Matrix](Documentation/COMPATIBILITY_MATRIX.md)
- [Failure Case Library](Documentation/FAILURE_CASE_LIBRARY.md)
- [First-Screen Conversion Spec](Documentation/FIRST_SCREEN_CONVERSION_SPEC.md)
- [Troubleshooting](Documentation/TROUBLESHOOTING.md)
- [Release Notes](RELEASE_NOTES.md)
- [Publishing Guide](PUBLISH.md)
- [Public Roadmap](roadmap/public-roadmap.md)
- [Benchmarks](BENCHMARKS.md)
- [Security Policy](SECURITY.md)
- [Asset Contribution Guide](CONTRIBUTING_ASSETS.md)

## Contributing

1. Branch from `codex/init`
2. Keep changes local-first and deterministic
3. Update docs with runnable examples
4. Run `python3 Scripts/codex_mem_smoketest.py --root .`
5. Include smoke output summary in PR

## License

No explicit open-source license file is included yet.  
Add one before broad redistribution.
