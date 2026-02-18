# 6-Layer Memory Runtime

`dev-mem` is implemented as a developer memory runtime with strict layer contracts.

## Layer Contracts

1. `access_ux`
- Entrypoints: `run-target`, `run-target-auto`, MCP tools
- Output: normalized task input

2. `task_compiler_plan`
- Profile mapping + docs-first policy + 9-section checklist
- Output: deterministic execution plan

3. `memory_ingestion_structuring`
- Canonical event/observation/evidence objects with trace metadata

4. `storage_index`
- SQLite + FTS
- Optional dense vectors
- Local graph-lite relations (persisted in `graph_lite_edges`)

5. `retrieval_ranking`
- Hybrid order: lexical/FTS -> structure -> graph-lite -> optional dense -> rerank
- Output: evidence pack + coverage stats

6. `execution_critic_delivery`
- Single-model execution switch (`none|codex|claude`)
- Critic hooks: stuck/risk/coverage follow-up
- Output contract: `LEARNING_COMPLETE|PARTIAL|INCOMPLETE` + forced next command
- Coverage recovery loop: `coverage_retry_max` (onboarding)

## Single-Model Runtime Rule

Each run uses exactly one executor:
- `none`: retrieval-only
- `codex`: codex CLI execution
- `claude`: claude CLI execution

Dual-agent validation is internal QA tooling only (`Scripts/dev_cross_verify_plan.py`).

## PMF Metrics

- Coverage: `>=95%` on required sections
- Efficiency: `>=30%` gain on combined time+token
- Usability: one-click natural-language entry via `run-target-auto`
