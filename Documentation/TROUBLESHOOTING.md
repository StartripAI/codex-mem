# Troubleshooting

## Push fails with `Repository not found`

- Verify the repository exists and remote URL is correct:

```bash
git remote -v
```

- Ensure PAT has write scope for the target repository.

## MCP server exits immediately

- Verify script path:

```bash
python3 Scripts/codex_mem_mcp.py --root .
```

- Ensure `Scripts/codex_mem.py` exists under `--root`.

## Web viewer does not open

- Start viewer explicitly:

```bash
bash Scripts/codex_mem.sh web --host 127.0.0.1 --port 37777
```

- Check health endpoint:

```bash
curl -s http://127.0.0.1:37777/api/health
```

- If port is occupied, use another port.

## No mem-search results

- Confirm lifecycle records exist for same `--project`.
- Ensure session was closed to generate summaries:

```bash
bash Scripts/codex_mem.sh session-end <session_id> --project <project>
```

- For private records, use `--include-private`.

## Private records appear unexpectedly

- Verify tags were passed via `--privacy-tag`, not only `--tag`.
- Check retrieval command does not include `--include-private`.

## Blocked sensitive writes still appear

- Use blocking privacy tags (`no_mem`, `block`, `skip`, `secret_block`) with `--privacy-tag`.
- Confirm command output reported `"skipped": true`.

## Config changes not taking effect

- Read current runtime config:

```bash
bash Scripts/codex_mem.sh config-get
```

- If channel is `stable`, auto endless compaction is disabled by design.

## SQLite lock errors

- Avoid running many concurrent writer processes against the same `.codex_mem` DB.
- Keep one writer flow per repo/session when possible.

## pycache/permission issues

- Run from a user-writable directory.
- Delete stale caches if needed:

```bash
find Scripts -name "__pycache__" -type d -prune -exec rm -rf {} +
```
