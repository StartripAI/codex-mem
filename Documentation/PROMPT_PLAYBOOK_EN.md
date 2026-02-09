# Prompt Playbook (English)

Use these copy-ready prompts in any project. They enforce codex-mem's 3-layer retrieval and keep token usage controlled.

## Case 1: Cold Start (learn project, then wait for questions)

```text
In this project, run codex-mem onboarding in low-token mode, then switch to Q&A standby.

Constraints:
- Do not modify code
- Do not generate any report files
- Do not print exploration logs (no explored/list/search progress chatter)
- Stop if total retrieval token estimate exceeds 2500 and report what is missing

Workflow (strict order):
1) Confirm command entrypoint (codex_mem.sh or codex_mem.py).
2) mem-search on: north star, architecture, entrypoints, core flow, persistence/output, major risks (limit 8).
3) timeline for top 2 hits (before 2, after 2).
4) get-observations for at most 4 selected IDs.
5) Read only up to 3 key files if evidence is still insufficient.

Output only:
1. North star goal
2. Module map
3. Main flow (entry -> processing -> persistence/output)
4. Top 3 technical risks with evidence (absolute path + function/module)
5. What context is loaded vs intentionally not loaded

Then wait for my questions. Do not rebuild full context unless I explicitly say: "rebuild learning context".
```

## Case 2: Daily Q&A (incremental, no rebuild)

```text
Answer using existing codex-mem context in this project. Do incremental retrieval only.

Workflow:
1) mem-search "<my question>" (limit 6)
2) timeline for top 2 IDs (before 2, after 2)
3) get-observations for top 3 IDs
4) Read at most 2 files only if evidence is missing

Output:
- Direct answer
- Evidence (command + ID + path/function)
- Risks/uncertainties (max 3)
- Next step (max 3)

Do not rebuild full project learning.
```

## Case 3: Bug/Incident Triage

```text
Triage this issue in this project with codex-mem 3-layer retrieval first, then code validation.

Workflow:
1) mem-search bug symptoms and related components (limit 12)
2) timeline around top 3 IDs (before 4, after 4)
3) get-observations for critical IDs
4) Validate against current code paths

Output:
1. Probable root cause
2. Reproduction path
3. Minimal fix options (A/B)
4. Verification checklist
5. Evidence references (absolute path + function/module)

Do not implement changes unless I explicitly say "apply fix".
```

## Case 4: Migration / Big Upgrade Review

```text
In this project, reconstruct architecture changes after upgrade using codex-mem.

Workflow:
1) mem-search migration keywords (limit 15)
2) timeline on top 4 IDs (before 5, after 5)
3) get-observations on migration-critical IDs
4) Read only key architecture files for confirmation

Output:
- New architecture overview
- Differences vs previous architecture
- Migration/compatibility risks
- Priority validation checklist
- Evidence (ID + absolute path + function/module)

No file generation, no code changes.
```

## Case 5: Implementation Mode (after agreement)

```text
Implement now, but keep codex-mem evidence-first process.

Before coding:
1) mem-search -> timeline -> get-observations for current task
2) Confirm assumptions and affected modules

During coding:
- Minimal patch
- Keep interfaces stable unless explicitly required
- Preserve backwards compatibility where possible

After coding:
- Run tests/build checks
- Report what changed, what passed, and remaining risks
- Include file-level evidence
```

## Practical Rule of Thumb

- Cold start: savings are moderate because code reading dominates.
- Daily Q&A: savings are highest because retrieval can stay narrow and incremental.
- Deep forensics: savings are still high, but lower than daily Q&A due to wider Layer-3 fetch.
