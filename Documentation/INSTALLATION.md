# Installation

## Requirements

- Python 3.10+
- SQLite with FTS5 support
- Optional: Codex CLI for MCP registration

## Clone

```bash
git clone https://github.com/<YOUR_ORG_OR_USER>/codex-mem.git
cd codex-mem
```

## Initialize Local Memory Store

```bash
bash Scripts/codex_mem.sh init --project demo
```

## Optional: Start Local Web Viewer

```bash
bash Scripts/codex_mem.sh web --project-default demo --host 127.0.0.1 --port 37777
```

Open:
- `http://127.0.0.1:37777/`

## Optional: Register MCP in Codex

```bash
codex mcp add codex-mem -- python3 /ABS/PATH/codex-mem/Scripts/codex_mem_mcp.py --root /ABS/PATH/codex-mem --project-default demo
```

## Verify

```bash
python3 Scripts/codex_mem_smoketest.py --root .
```

Expected output includes:
- `"ok": true`
- non-zero search results
- verified MCP tools
- verified web endpoints
