# dev-mem

`dev-mem` is a memory runtime for coding workflows.
It lets you run long tasks on a target repository with less repeated context, lower token waste, and more stable outputs.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-ready-0A7C66)

[Quick Start](#quick-start) • [One-Click Target Run](#one-click-target-run) • [CLI Guide](Documentation/CODEX_MEM.md) • [Architecture](Documentation/ARCHITECTURE.md) • [MCP Tools](Documentation/MCP_TOOLS.md) • [Benchmarks](BENCHMARKS.md)

Legacy compatibility:
- `Scripts/codex_mem.sh` remains available as an alias.

## What You Get

- One-click target-project learning/execution entry.
- Persistent project memory across sessions.
- Hybrid retrieval (lexical + structure + graph-lite + optional embeddings).
- Evidence-grounded outputs with file/symbol traceability.
- Stable output contract with forced next executable command.

## Quick Start

### 1) Prerequisites

- macOS/Linux shell
- Python 3.10+

### 2) Initialize local memory

```bash
bash Scripts/dev_mem.sh init --project demo
```

### 3) Ask with memory + repository grounding

```bash
bash Scripts/dev_mem.sh ask "map entrypoint and persistence chain" --project demo
```

## One-Click Target Run

Use this when you want to run `dev-mem` against another repository.

### Auto mode (natural language in)

```bash
bash /ABS/PATH/TO/dev-mem/Scripts/dev_mem.sh \
  run-target-auto "learn this project: north star, architecture, module map, entrypoint, main flow, persistence, ai generation, tests, risks"
```

### Explicit target root mode

```bash
bash /ABS/PATH/TO/dev-mem/Scripts/dev_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "learn this project: north star, architecture, module map, entrypoint, main flow, persistence, ai generation, tests, risks"
```

### Single-model executor per run

```bash
bash /ABS/PATH/TO/dev-mem/Scripts/dev_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --executor codex \
  --question "implement feature X with evidence and tests"
```

Executor options:
- `none`
- `codex`
- `claude`

## Typical Workflow

### 1) Capture lifecycle

```bash
bash Scripts/dev_mem.sh session-start s1 --project demo --title "Streaming refactor"
bash Scripts/dev_mem.sh prompt s1 "Map generation and persistence path" --project demo
bash Scripts/dev_mem.sh tool s1 shell "rg -n 'DatabaseBootstrapper|index.ts'" --project demo --title "Locate chain" --compact
bash Scripts/dev_mem.sh stop s1 --project demo --content "checkpoint"
bash Scripts/dev_mem.sh session-end s1 --project demo
```

### 2) Retrieve progressively

```bash
# Layer 1: compact search
bash Scripts/dev_mem.sh search "persistence bootstrap" --project demo --limit 20

# Layer 2: timeline around one item
bash Scripts/dev_mem.sh timeline E12 --project demo --before 5 --after 5

# Layer 3: full details
bash Scripts/dev_mem.sh get E12 O3 --project demo
```

### 3) Run fused ask

```bash
bash Scripts/dev_mem.sh ask "map entrypoint, ai generation path, persistence, and top risks" --project demo
```

## Output Contract

`ask` responses include:

- completion status (`LEARNING_COMPLETE` / `PARTIAL` / `INCOMPLETE`)
- section coverage report
- evidence stats
- mandatory `forced_next_input` command

## Runtime Modes

Read config:

```bash
bash Scripts/dev_mem.sh config-get
```

Set config:

```bash
bash Scripts/dev_mem.sh config-set --channel beta --viewer-refresh-sec 2 --beta-endless-mode on
```

## Local Viewer

Start:

```bash
bash Scripts/dev_mem.sh web --project-default demo --host 127.0.0.1 --port 37777
```

Open: `http://127.0.0.1:37777/`

## MCP Server

Run server:

```bash
python3 Scripts/codex_mem_mcp.py --root . --project-default demo
```

Register:

```bash
codex mcp add dev-mem -- python3 /ABS/PATH/TO/dev-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/TO/dev-mem --project-default demo
```

Key tools:

- retrieval: `mem_search`, `mem_nl_search`, `mem_timeline`, `mem_get_observations`, `mem_ask`
- lifecycle: `mem_session_start`, `mem_user_prompt_submit`, `mem_post_tool_use`, `mem_stop`, `mem_session_end`
- runtime: `mem_config_get`, `mem_config_set`

## Benchmarks

Run standardized runtime pipeline:

```bash
python3 Scripts/benchmark_runtime_pipeline.py \
  --root . \
  --out Documentation/benchmarks/runtime_pipeline_latest.json \
  --checkpoint Documentation/benchmarks/runtime_pipeline_checkpoint.json
```

Generate PMF dashboard:

```bash
python3 Scripts/build_pmf_dashboard.py --root . --out Documentation/benchmarks/PMF_DASHBOARD.md
```

## Project Structure

- `Scripts/codex_mem.py`: CLI engine and runtime entry.
- `Scripts/dev_mem.sh`: shell entrypoints including `run-target` and `run-target-auto`.
- `Scripts/memory_runtime/`: plan/retrieval/execution contracts.
- `Scripts/tests/`: unit and e2e tests.
- `Documentation/`: architecture, runtime rules, and guides.

## License

See repository license file.
