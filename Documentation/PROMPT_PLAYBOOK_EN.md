# Prompt Playbook (English, Strict SOP)

This playbook is for short user prompts, but with strict operating rules. `codex-mem` handles routing and prompt shaping internally; users should not send long template prompts.

## Scope

- Applies to `ask` workflows (`onboarding`, `daily_qa`, `bug_triage`, `implementation`).
- Defines mandatory first-read procedure for project learning.
- Enforces evidence-grounded output rules.

## Single North Star (Hard Requirement)

The only success criterion is:
- run a target repository through `codex-mem` automation
- reduce context/memory waste without losing evidence quality

No workflow is considered valid if it does not run through the `codex-mem` entrypoint with an explicit target root.

## Mandatory Execution Entrypoint (Hard Requirement)

Always execute via `codex_mem.sh run-target` with both:
- explicit `run-target "/ABS/PATH/TO/TARGET_PROJECT"`
- `--project <target_project_name>`

Canonical command:

```bash
bash /ABS/PATH/TO/codex-mem/Scripts/codex_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "learn this project: north star, architecture, module map, entrypoint, main flow, persistence, ai generation, risks."
```

Forbidden execution patterns:
- bypassing `codex_mem.sh run-target` in cross-repo onboarding flows
- running `run-target` without explicit target repository path
- running without `--project`
- returning advice without an executable next command

## Natural-Language Routing (Hard Requirement)

When the user gives natural language (not CLI args), routing must still output a runnable command.

Hard routing policy:
1. Extract target root from user text if an absolute path is present.
2. Else use current IDE/CLI workspace root if it is not the `codex-mem` repo.
3. Else output `TARGET_ROOT_REQUIRED` only.
4. Output format must be one line command only (no explanation).

## Callable Prompt Template (Hard Requirement)

Only callable prompts are allowed. Any prompt that cannot trigger `codex-mem` is invalid.

Canonical short prompt (ZH):

```text
通过 codex-mem run-target 执行目标项目深度首读并返回结果；自动识别目标项目根目录与项目名，无法识别时返回 TARGET_ROOT_REQUIRED。
```

Backend SOP (hard-locked):
- `Documentation/CALLABLE_PROMPT_SOP_ZH.md`

## Non-Negotiable Rules

1. Keep user prompts short and task-focused.
2. First-read must run in order:
   - top-down: goal -> architecture -> module boundaries -> main flow
   - bottom-up: entrypoint -> key modules -> key functions/data paths
3. On onboarding, coverage must include:
   - `entrypoint`
   - `persistence`
   - `ai_generation`
4. Every key conclusion must include evidence:
   - file path
   - function/class/symbol
   - command output summary (when commands were used)
5. Do not invent KPI or percentage targets.
   - Exception: completion percentage/range is allowed when derived from explicit section coverage + evidence counts.
   - Other numeric claims are allowed only when quoting existing benchmark files.

## Mandatory Response Contract (No Intro Text)

This is a hard requirement for project-learning and optimization runs.

### Forbidden Output

- Introductory/meta lines such as:
  - "Summary"
  - "Goal is..."
  - "A/B/C sections"
  - any restatement of the user request
- Generic advice without repository evidence
- Decorative headings before required content

### Required Output Order

Output must be returned in exactly this order:
1. Up to 10 factual lines, no title/header line.
2. Optimization actions grouped as `P0`, `P1`, `P2` (each item must include target change + risk).
3. Validation fields (verbatim keys must appear):
   - `mapping_decision`
   - `coverage_gate`
   - `prompt_plan`
   - `prompt_metrics`
   - `forced_next_input`
4. One fully executable next `ask` command.
   - Must use `codex_mem.sh run-target "/ABS/PATH/TO/TARGET_PROJECT"` format.

### Failure Contract

If `coverage_gate.pass` is `false`, return only:
1. `INCOMPLETE`
2. missing category list
3. one full `ask` command that includes "only fill missing categories"
   - Must use `codex_mem.sh run-target "/ABS/PATH/TO/TARGET_PROJECT"` format.

No additional explanation is allowed in failure mode.

## Default Ask Behavior

`ask` defaults to `--prompt-style compact` and runs:
1. prompt-to-profile mapping (`onboarding`, `daily_qa`, `bug_triage`, `implementation`)
2. memory + repo retrieval
3. onboarding coverage gate (`entrypoint`, `persistence`, `ai_generation`)
4. token budgeting + compact rendering

Useful controls:
- `--prompt-style {compact,legacy}`
- `--mapping-fallback {auto,off}`
- `--mapping-debug`

## First-Read SOP (Strict)

### Step 1: Run onboarding ask

```bash
bash /ABS/PATH/TO/codex-mem/Scripts/codex_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "learn this project: north star, architecture, module map, entrypoint, persistence, risks"
```

### Step 2: Validate routing and coverage (required for first-read)

```bash
bash /ABS/PATH/TO/codex-mem/Scripts/codex_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "learn this project: north star, architecture, module map, entrypoint, persistence, risks" \
  -- --mapping-debug
```

### Step 3: Produce first-read report in fixed sections

Required sections:
- north star and boundaries
- architecture + module map
- main flow (input -> processing -> persistence -> output)
- evidence table for `entrypoint` / `persistence` / `ai_generation`
- top risks + unknowns (explicitly marked)

### Step 4: Completion gate (pass/fail)

First-read is complete only if all are true:
- `coverage_gate.pass = true`
- no required section is missing
- no claim without evidence
- no invented numeric target

If any item fails, mark output as `INCOMPLETE` and list missing evidence.

## Minimal User Inputs (Recommended)

Use short task statements:
- onboarding: `learn this repo architecture and risks`
- daily Q&A: `why did this flow fail yesterday`
- bug triage: `triage crash in startup path`
- implementation: `implement compact renderer with compatibility`

## Standard Cases

### Case 1: Cold Start (learn project, then standby)

```bash
bash /ABS/PATH/TO/codex-mem/Scripts/codex_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "learn this project: architecture, entrypoint, persistence, risks"
```

### Case 2: Daily Q&A (incremental retrieval)

```bash
bash /ABS/PATH/TO/codex-mem/Scripts/codex_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "what changed in generation flow"
```

Strict local routing only:

```bash
bash /ABS/PATH/TO/codex-mem/Scripts/codex_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "what changed in generation flow" \
  -- --mapping-fallback off
```

### Case 3: Bug/Incident Triage

```bash
bash /ABS/PATH/TO/codex-mem/Scripts/codex_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "triage this regression and provide root cause path"
```

### Case 4: Implementation Mode

```bash
bash /ABS/PATH/TO/codex-mem/Scripts/codex_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "implement this task with minimal compatibility risk"
```

### Case 5: Legacy Prompt Comparison (regression only)

```bash
bash /ABS/PATH/TO/codex-mem/Scripts/codex_mem.sh \
  run-target "/ABS/PATH/TO/TARGET_PROJECT" \
  --project target \
  --question "learn this repo architecture and top risks" \
  -- --prompt-style legacy
```

## Reading `ask` Output

Compact mode returns:
- `mapping_decision`: route source, confidence, low-confidence flag
- `coverage_gate`: required/present/missing categories and pass/fail
- `prompt_plan`: budget allocations and selected evidence
- `prompt_metrics`: rendered prompt size and budget usage

Compatibility fields:
- `suggested_prompt`
- `token_estimate`

## Rule of Thumb

- Prefer short prompts.
- Keep `compact` as production default.
- Use `--mapping-debug` only for routing/coverage validation.
- Use `legacy` only for A/B or regression checks.
