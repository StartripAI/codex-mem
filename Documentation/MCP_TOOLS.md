# MCP Tools Reference

`Scripts/codex_mem_mcp.py` exposes the following tools.

## Retrieval tools

### mem_search

Purpose: Layer-1 compact retrieval.

Input:
- `query` (required)
- `project` (optional)
- `session_id` (optional)
- `limit` (optional)
- `alpha` (optional)

### mem_timeline

Purpose: Layer-2 temporal context around selected ID.

Input:
- `id` (required, `E<ID>` or `O<ID>`)
- `before`, `after`, `snippet_chars` (optional)

### mem_get_observations

Purpose: Layer-3 full details for selected IDs.

Input:
- `ids` (required)
- `compact`, `snippet_chars` (optional)

### mem_ask

Purpose: fused memory + repository retrieval.

Input:
- `question` (required)
- `project`, `session_id`, `search_limit`, `detail_limit`, `code_top_k`, `code_module_limit`, `repo_index_dir`, `alpha`, `snippet_chars`, `prompt_only` (optional)

## Lifecycle tools

- `mem_session_start`
- `mem_user_prompt_submit`
- `mem_post_tool_use`
- `mem_stop`
- `mem_session_end`
- `mem_summarize_session`

## Error model

The server returns JSON-RPC errors for invalid schema, unknown tools, command failures, and internal errors.
