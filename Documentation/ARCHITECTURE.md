# Architecture

## Overview

`codex-mem` is a local-first memory engine with progressive retrieval and MCP exposure.

Core modules:
- `codex_mem.py` (storage, retrieval, summary generation, fused ask)
- `codex_mem_mcp.py` (MCP transport + tool mapping)
- `repo_knowledge.py` (repository retrieval for fused ask)

## Data flow

1. Lifecycle hooks append events to SQLite.
2. Session end generates structured summary observations.
3. Retrieval uses FTS5 + local semantic vectors for blended ranking.
4. MCP tools expose retrieval/lifecycle operations to Codex.
5. `ask` merges memory context with repository chunks.

## Storage schema

Tables:
- `sessions`
- `events`
- `observations`
- `events_fts`
- `observations_fts`
- `meta`

## Ranking model

- Lexical: FTS5 BM25-derived score normalization
- Semantic: local deterministic vector similarity
- Blend: weighted lexical/semantic score (`alpha`)

## Compaction model

For heavy tool output (`--compact`):
- keep head segment
- extract signal lines (error/warning/fail/trace/etc.)
- keep tail segment
