# Prompt Playbook (English)

This playbook is for **short user prompts**. `codex-mem` handles routing and prompt shaping internally, so users do not need long template instructions.

## Default Behavior

`ask` now defaults to `--prompt-style compact` and runs:
1. prompt-to-profile mapping (`onboarding`, `daily_qa`, `bug_triage`, `implementation`)
2. memory + repo retrieval
3. onboarding coverage gate (`entrypoint`, `persistence`, `ai_generation`)
4. token budgeting + compact rendering

Useful controls:
- `--prompt-style {compact,legacy}`
- `--mapping-fallback {auto,off}`
- `--mapping-debug`

## Minimal User Inputs (Recommended)

Use short task statements. Typical examples:
- onboarding: `learn this repo architecture and risks`
- daily Q&A: `why did this flow fail yesterday`
- bug triage: `triage crash in startup path`
- implementation: `implement compact renderer with compatibility`

## Case 1: Cold Start (learn project, then standby)

```bash
python3 Scripts/codex_mem.py --root . ask \
  "learn this project: architecture, entrypoint, persistence, risks" \
  --project demo
```

Optional diagnostics:

```bash
python3 Scripts/codex_mem.py --root . ask \
  "learn this project: architecture, entrypoint, persistence, risks" \
  --project demo --mapping-debug
```

## Case 2: Daily Q&A (incremental retrieval)

```bash
python3 Scripts/codex_mem.py --root . ask "what changed in generation flow" --project demo
```

If you want strict local routing only:

```bash
python3 Scripts/codex_mem.py --root . ask \
  "what changed in generation flow" \
  --project demo --mapping-fallback off
```

## Case 3: Bug/Incident Triage

```bash
python3 Scripts/codex_mem.py --root . ask \
  "triage this regression and provide root cause path" \
  --project demo
```

## Case 4: Implementation Mode

```bash
python3 Scripts/codex_mem.py --root . ask \
  "implement this task with minimal compatibility risk" \
  --project demo
```

## Case 5: Legacy Prompt Comparison

Use this only for regression checks:

```bash
python3 Scripts/codex_mem.py --root . ask \
  "learn this repo architecture and top risks" \
  --project demo --prompt-style legacy
```

## Reading `ask` Output

Compact mode returns additional decision/quality fields:
- `mapping_decision`: route source, confidence, low-confidence flag
- `coverage_gate`: required/present/missing categories and pass/fail
- `prompt_plan`: budget allocations and selected evidence
- `prompt_metrics`: rendered prompt size and budget usage

Compatibility fields still exist:
- `suggested_prompt`
- `token_estimate`

## Rule of Thumb

- Prefer short prompts.
- Use `--mapping-debug` only when validating routing behavior.
- Keep `compact` as default for production usage.
- Use `legacy` only for A/B or regression checks.
