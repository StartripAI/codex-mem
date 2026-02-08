# Architecture

## System Overview

`codex-mem` is a local-first memory system for Codex with four planes:

1. **Capture plane**: lifecycle hooks persist session evidence
2. **Retrieval plane**: progressive search/timeline/detail retrieval
3. **Integration plane**: MCP tools + Skill workflow
4. **UX plane**: local web viewer with runtime mode controls

## Core Modules

- `Scripts/codex_mem.py`
  - DB schema and persistence
  - retrieval ranking and progressive disclosure logic
  - natural-language query parsing
  - runtime config and privacy policy handling
  - fused memory + code retrieval (`ask`)

- `Scripts/codex_mem_mcp.py`
  - MCP transport/protocol
  - tool schemas and validation
  - CLI bridge for `mem_*` tools

- `Scripts/codex_mem_web.py`
  - local HTTP API + SPA viewer
  - stream/session/search endpoints
  - stable/beta runtime configuration controls

- `Scripts/repo_knowledge.py`
  - repository indexing/query for fused ask context

## Data Flow

1. Lifecycle command writes event into SQLite.
2. Event text is indexed in FTS and vectorized locally.
3. `session-end` compiles session summary observations.
4. Retrieval path:
   - Layer 1 (`search` or `nl-search`)
   - Layer 2 (`timeline`)
   - Layer 3 (`get-observations`)
5. `ask` fuses memory shortlist with repository code context.
6. MCP/Web surfaces expose the same underlying engine.

## Storage Schema

Main tables:
- `meta`
- `sessions`
- `events`
- `observations`
- `events_fts`
- `observations_fts`

Runtime config in `meta`:
- `channel` (`stable`/`beta`)
- `viewer_refresh_sec`
- `beta_endless_mode`

Privacy metadata per record in `metadata_json`:
- visibility (`public`/`private`)
- policy tags
- redaction flag

## Ranking Model

- Lexical score: FTS5 BM25-derived normalization
- Semantic score: local deterministic hash vector cosine similarity
- Blended score: `alpha * lexical + (1-alpha) * semantic`

Natural-language queries additionally apply:
- lightweight time phrase parsing
- intent keyword post-filtering

## Compaction Model

Tool output compaction keeps:
- head snippet
- signal lines (errors/warnings/failures/etc.)
- tail snippet

Modes:
- `stable`: compaction only when requested
- `beta` + endless mode: auto-compaction for post-tool-use writes

## Privacy Model (Dual Tags)

- semantic tags (`--tag`): search taxonomy and retrieval hints
- privacy tags (`--privacy-tag`): policy controls
  - block write
  - private visibility
  - redaction

Default retrieval excludes private records unless explicitly requested.

## MCP Surface

Tool categories:
- retrieval: `mem_search`, `mem_nl_search`, `mem_timeline`, `mem_get_observations`, `mem_ask`
- runtime config: `mem_config_get`, `mem_config_set`
- lifecycle: session and tool hook operations
