# MCP Tools Reference

`Scripts/codex_mem_mcp.py` exposes the following tools.

## Retrieval Tools

### `mem_search`

Purpose:
- Layer-1 compact retrieval by lexical + semantic score.

Input:
- `query` (required)
- `project` (optional)
- `session_id` (optional)
- `since`, `until` (optional ISO datetime bounds)
- `include_private` (optional bool)
- `limit` (optional int)
- `alpha` (optional float)

### `mem_nl_search`

Purpose:
- Natural language memory retrieval with lightweight time phrase and intent parsing.

Input:
- `query` (required)
- `project`, `session_id` (optional)
- `since`, `until` (optional override)
- `include_private` (optional bool)
- `limit`, `alpha`, `snippet_chars` (optional)

### `mem_timeline`

Purpose:
- Layer-2 temporal context around selected record ID.

Input:
- `id` (required, `E<ID>` or `O<ID>`)
- `before`, `after` (optional)
- `include_private` (optional bool)
- `snippet_chars` (optional)

### `mem_get_observations`

Purpose:
- Layer-3 detailed payload for selected IDs.

Input:
- `ids` (required array)
- `compact` (optional bool)
- `include_private` (optional bool)
- `snippet_chars` (optional)

### `mem_ask`

Purpose:
- fused memory + repository retrieval context.

Input:
- `question` (required)
- `project`, `session_id` (optional)
- `search_limit`, `detail_limit` (optional)
- `code_top_k`, `code_module_limit` (optional)
- `repo_index_dir` (optional)
- `alpha` (optional)
- `include_private` (optional bool)
- `snippet_chars` (optional)
- `prompt_only` (optional bool)

## Runtime Config Tools

### `mem_config_get`

Purpose:
- read runtime mode and viewer settings.

Input:
- none

### `mem_config_set`

Purpose:
- update runtime mode and viewer settings.

Input:
- `channel` (`stable` or `beta`)
- `viewer_refresh_sec` (1..60)
- `beta_endless_mode` (bool)

## Lifecycle Tools

### `mem_session_start`
- `session_id` required
- optional: `project`, `title`, `content`

### `mem_user_prompt_submit`
- `session_id`, `prompt` required
- optional: `project`, `title`

### `mem_post_tool_use`
- required: `session_id`, `tool_name`, `content`
- optional:
  - `project`, `title`, `file_path`, `exit_code`
  - `tags` (semantic tags)
  - `privacy_tags` (policy tags)
  - `compact`, `compact_chars`

### `mem_stop`
- `session_id` required
- optional: `project`, `title`, `content`

### `mem_session_end`
- `session_id` required
- optional: `project`, `title`, `content`, `skip_summary`

### `mem_summarize_session`
- `session_id` required

## Error Model

JSON-RPC errors are returned for:
- invalid input schema
- unknown tool
- `codex_mem.py` command execution failure
- internal server errors
