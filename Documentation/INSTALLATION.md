# Installation

## Requirements

- Python 3.10+
- SQLite with FTS5 support
- Optional: Codex CLI for MCP registration

## Install

No package manager is required for the current version.

```bash
git clone https://github.com/<YOUR_ORG_OR_USER>/codex-mem.git
cd codex-mem
```

Initialize memory store:

```bash
bash Scripts/codex_mem.sh init --project demo
```

## Optional: register MCP in Codex

```bash
codex mcp add codex-mem -- python3 /ABS/PATH/codex-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/codex-mem --project-default demo
```

## Verify

```bash
python3 Scripts/codex_mem_smoketest.py --root .
```
