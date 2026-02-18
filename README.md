# Lakeside-mem

## Token is the hard currency of modern work and research.

Lakeside-mem is a local-first memory runtime for software teams.
It keeps coding context persistent, retrieves only what matters, and drives stable next actions with evidence.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-ready-0A7C66)

[Quick Start](#quick-start) • [Most-Used Scenarios](#most-used-scenarios) • [Landing Page](Documentation/LAKESIDE_MEM_SHOWCASE.html) • [CLI Guide](Documentation/CODEX_MEM.md) • [Architecture](Documentation/ARCHITECTURE.md) • [Benchmarks](BENCHMARKS.md)

Legacy compatibility:
- `Scripts/codex_mem.sh` still works.
- `Scripts/dev_mem.sh` still works.

## What It Does

- One-click target repository execution (`run-target` / `run-target-auto`)
- Persistent project memory across sessions
- Hybrid retrieval: lexical + structure + graph-lite + optional embeddings
- Evidence-grounded output with file/symbol traceability
- Stable output contract with mandatory `forced_next_input`

## Quick Start

### 1) Initialize local memory

```bash
bash Scripts/lakeside_mem.sh init --project demo
```

### 2) Run against a target repository

```bash
bash /ABS/PATH/TO/lakeside-mem/Scripts/lakeside_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "learn this project: goal, architecture, module map, entrypoint, main flow, persistence, ai generation, tests, risks"
```

### 3) Natural language auto mode

```bash
bash /ABS/PATH/TO/lakeside-mem/Scripts/lakeside_mem.sh \
  run-target-auto "learn this project deeply and return evidence-backed conclusions"
```

## Most-Used Scenarios

These are the most frequent workflows across modern coding-agent usage patterns (onboarding, issue fixing, debug loops, implementation, and repo-scale maintenance).

### 1) New repo onboarding (first day)

```bash
bash Scripts/lakeside_mem.sh run-target "/ABS/PATH/TO/TARGET_PROJECT" --project target \
  --question "learn this project: goal, architecture, module map, entrypoint, main flow, persistence, ai generation, tests, risks"
```

### 2) Bug triage from a failing issue

```bash
bash Scripts/lakeside_mem.sh run-target "/ABS/PATH/TO/TARGET_PROJECT" --project target \
  --question "triage this bug: repro path, root cause chain, minimal-risk fix, validation checklist"
```

### 3) Debug a failing test or terminal error

```bash
bash Scripts/lakeside_mem.sh run-target "/ABS/PATH/TO/TARGET_PROJECT" --project target \
  --question "debug this failure: isolate failing path, show exact evidence, propose smallest fix"
```

### 4) Implement a feature with guardrails

```bash
bash Scripts/lakeside_mem.sh run-target "/ABS/PATH/TO/TARGET_PROJECT" --project target \
  --question "implement this feature with minimal patch, compatibility boundaries, tests, and rollout notes"
```

### 5) Refactor safely in a large codebase

```bash
bash Scripts/lakeside_mem.sh run-target "/ABS/PATH/TO/TARGET_PROJECT" --project target \
  --question "plan and execute a safe refactor: dependency map, blast radius, incremental steps, regression checks"
```

### 6) PR review and risk scan

```bash
bash Scripts/lakeside_mem.sh run-target "/ABS/PATH/TO/TARGET_PROJECT" --project target \
  --question "review this change: behavior diffs, hidden risks, missing tests, and release impact"
```

### 7) Incident forensics and timeline reconstruction

```bash
bash Scripts/lakeside_mem.sh run-target "/ABS/PATH/TO/TARGET_PROJECT" --project target \
  --question "reconstruct this incident timeline, identify root cause, and produce a prevention checklist"
```

### 8) Handoff summary for teammates

```bash
bash Scripts/lakeside_mem.sh run-target "/ABS/PATH/TO/TARGET_PROJECT" --project target \
  --question "generate a handoff: what changed, why, evidence, open risks, and next executable command"
```

## Runtime & Output Contract

`ask` / `run-target` output includes:

- completion status (`LEARNING_COMPLETE` / `PARTIAL` / `INCOMPLETE`)
- section coverage report
- evidence stats
- mandatory `forced_next_input`
- 6-layer runtime metadata (`memory_runtime_layers`)

## MCP Usage (Local Install, No Store Required)

Run MCP server:

```bash
python3 Scripts/codex_mem_mcp.py --root . --project-default demo
```

Register locally:

```bash
codex mcp add lakeside-mem -- python3 /ABS/PATH/TO/lakeside-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/TO/lakeside-mem --project-default demo
```

## Benchmarks (Latest)

- prompt compaction saving: `52.86%`
- scenario max token saving: `99.84%`
- marketing benchmark saving: `99.84%`
- runtime pipeline: `4/4` stages passed

Run pipeline:

```bash
python3 Scripts/benchmark_runtime_pipeline.py \
  --root . \
  --out Documentation/benchmarks/runtime_pipeline_latest.json \
  --checkpoint Documentation/benchmarks/runtime_pipeline_checkpoint.json
```

## Keep It Simple

This is not magic. It is a practical runtime that helps you stop repeating context and start shipping with cleaner evidence.

## License

See repository license file.
