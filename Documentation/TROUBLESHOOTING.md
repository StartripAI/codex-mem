# Troubleshooting

## Push fails with `Repository not found`

- Ensure the target repository exists.
- Ensure PAT has write permission to that repository.
- Verify remote URL:

```bash
git remote -v
```

## MCP server exits immediately

- Verify `Scripts/codex_mem.py` exists under `--root`.
- Run directly for diagnostics:

```bash
python3 Scripts/codex_mem_mcp.py --root .
```

## No search results

- Check events are written under the same `--project`.
- End session to generate summaries:

```bash
bash Scripts/codex_mem.sh session-end <session_id> --project <project>
```

## Permission error writing cache/pyc

Run under a user-writable directory and avoid protected system paths.

## SQLite lock errors

Avoid running many concurrent writers against the same `.codex_mem` database file.
