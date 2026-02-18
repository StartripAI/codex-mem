# Callable Prompt Backend SOP (ZH)

## Goal

- Use `Lakeside-mem` as the only execution entrypoint for target-project deep onboarding.
- Produce a single-pass, high-completeness project understanding report grounded in file/symbol evidence.

## Entrypoint (Hard-Locked)

```bash
bash /ABS/PATH/TO/lakeside-mem/Scripts/lakeside_mem.sh run-target "<TARGET_ROOT_ABS>" --project "<PROJECT_SLUG>" --question "<TASK>"
```

## Root + Project Resolution

1. Resolve target root from absolute path in request text.
2. If missing, resolve from current workspace root.
3. If unresolved, output `TARGET_ROOT_REQUIRED`.
4. Derive project slug from target root basename.

## Read Order (Hard-Locked)

1. Documentation first.
2. Core code second.
3. Tests and risk surfaces third.

## MECE Output Sections (Required)

1. North star and boundaries.
2. Architecture and module map.
3. Entrypoint and main flow.
4. Persistence chain.
5. AI generation chain.
6. Test status and quality signals.
7. Key risks and priority.

## Evidence Gate

- Each section must include at least 3 evidence lines.
- Each evidence line must include:
  - absolute file path
  - key symbol (function/class/module/entry)
  - role/impact

## Gap Handling

- If any section lacks required evidence, continue retrieval in the same run.
- Do not finalize before all required sections satisfy evidence gate.

## Final Delivery

- MECE 7-section report
- P0/P1/P2 action list
- Next executable command
- Completion may include percentage/range when backed by section coverage and evidence counts
- For completion queries, prioritize: project purpose + completion estimate + covered/missing sections
- Use `INCOMPLETE` only when evidence is severely insufficient
- If partial coverage exists, default status should be `PARTIAL` (not `INCOMPLETE`)

## Benchmark Targets

- Coverage target: >= 90%
- Efficiency target: >= 30% gain on combined time + token cost
- Report results as-is whether outcome is better or worse than target
