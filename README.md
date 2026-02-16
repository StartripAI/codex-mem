# codex-mem

`codex-mem` gives Codex persistent project memory so each new session starts with evidence, not re-explaining everything.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-ready-0A7C66)
![Prompt Compaction](https://img.shields.io/badge/compact_prompt-53.28%25_less_tokens-16a34a)
![Smoke Test](https://img.shields.io/badge/smoke_test-passing-16a34a)

[Quick Start](#quick-start) • [Prompt SOP](Documentation/PROMPT_PLAYBOOK_EN.md) • [Architecture](Documentation/ARCHITECTURE.md) • [MCP Tools](Documentation/MCP_TOOLS.md) • [Benchmarks](BENCHMARKS.md) • [Release Notes](RELEASE_NOTES.md)

## Why This Exists

Without memory, Codex workflows degrade over time:
- repeated context dumping
- higher token spend
- weaker follow-up accuracy

`codex-mem` solves this with local-first lifecycle capture + progressive retrieval + repository grounding.

## Required Target-Project Entrypoint

For cross-repository usage, always run through `codex_mem.sh run-target` with an explicit target root.

```bash
bash /ABS/PATH/TO/codex-mem/Scripts/codex_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "learn this project: north star, architecture, module map, entrypoint, main flow, persistence, ai generation, risks."
```

Hard rules:
- do not bypass `codex_mem.sh run-target` for cross-repo onboarding
- do not run without explicit `run-target "/ABS/PATH/TO/TARGET_PROJECT"`
- do not accept non-executable guidance as output
- for natural-language requests, resolver must auto-detect target root (path in user text first, then workspace root); if unresolved, return `TARGET_ROOT_REQUIRED`
- only callable prompts are allowed; non-callable prompt text is invalid output

## What You Get

1. Cross-session memory with SQLite + FTS + deterministic vectors.
2. Progressive retrieval pipeline:
   - Layer 1: `search` / `mem-search`
   - Layer 2: `timeline`
   - Layer 3: `get-observations`
3. `ask` that fuses memory evidence with live repo evidence.
4. Built-in compact prompt system (default):
   - profile mapping (`onboarding`, `daily_qa`, `bug_triage`, `implementation`)
   - onboarding coverage gate (`entrypoint`, `persistence`, `ai_generation`)
   - token budgeting + compact rendering

## Measured Impact (2026-02-11)

| Metric | Result | Source |
|---|---:|---|
| Onboarding pack vs `ask` context | 94.71% reduction | [`Documentation/benchmarks/onboarding_pack_codex_mem_rich_20260211.json`](Documentation/benchmarks/onboarding_pack_codex_mem_rich_20260211.json) |
| Repo grounding vs full corpus | 97.76% reduction | [`Documentation/benchmarks/repo_onboarding_codex_mem_20260211.json`](Documentation/benchmarks/repo_onboarding_codex_mem_20260211.json) |
| Daily Q&A context savings | 99.84% reduction | [`Documentation/benchmarks/marketing_claims_20260211.json`](Documentation/benchmarks/marketing_claims_20260211.json) |
| Compact prompt vs legacy prompt | 53.28% fewer tokens | [`Documentation/benchmarks/prompt_compaction_20260211.json`](Documentation/benchmarks/prompt_compaction_20260211.json) |

More detail:
- scenario matrix: [`Documentation/benchmarks/scenario_savings_20260211.json`](Documentation/benchmarks/scenario_savings_20260211.json)
- methodology notes: [`Documentation/benchmarks/MEASURED_SAVINGS_20260211.md`](Documentation/benchmarks/MEASURED_SAVINGS_20260211.md)

## 60-Second Start

```bash
bash /ABS/PATH/TO/codex-mem/Scripts/codex_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "learn this project: north star, architecture, module map, entrypoint, main flow, persistence, ai generation, risks."
```

Default `ask` behavior:
- `--prompt-style compact`
- `--mapping-fallback auto`

For regression comparison:

```bash
bash /ABS/PATH/TO/codex-mem/Scripts/codex_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "same question" \
  -- --prompt-style legacy
```

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
bash Scripts/codex_mem.sh ask "map entrypoint and persistence chain" --project demo
```

Compact mode is default. Force legacy format when comparing behavior:

```bash
python3 Scripts/codex_mem.py --root . ask "learn architecture and top risks" --project demo --prompt-style legacy
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

Domain isolation gate:

```bash
python3 Scripts/check_domain_isolation.py --root .
```

Prompt compaction benchmark:

```bash
python3 Scripts/benchmark_prompt_compaction.py --root . --runs 3
```

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
