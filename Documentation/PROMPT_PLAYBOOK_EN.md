# Prompt Playbook (English)

Use these copy-ready prompts in any project. They enforce codex-mem's **progressive disclosure retrieval** (Layer 1 shortlist → Layer 2 timeline → Layer 3 details) and **evidence-first gating** (only pull more context when the current evidence is insufficient).

## Case 1: Cold Start (learn project, then wait for questions)

```text
Cold start this project with codex-mem, then switch to Q&A standby.

Constraints:
- Do not modify code
- Do not generate any report files
- Do not print exploration logs (no explored/list/search progress chatter)

Workflow (strict order):
1) Confirm command entrypoint (codex_mem.sh or codex_mem.py).
2) Run `codex-mem ask` with:
   - question: "Learn this project: north star, architecture, module map, entrypoints, main flow, persistence/output, top risks."
   - `--code-top-k 10 --code-module-limit 6 --snippet-chars 1000`
   - `--search-limit 6 --detail-limit 3`
3) If the answer is still ambiguous, use the 3-layer memory tools in order:
   - Layer 1: `mem-search "<missing topic>" --limit 8`
   - Layer 2: `timeline <top-id> --before 2 --after 2`
   - Layer 3: `get-observations <id1> <id2> <id3>`
4) Read only the minimum number of files needed to confirm the top-level model (max 3).

Output only:
1. North star goal
2. Module map
3. Main flow (entry -> processing -> persistence/output)
4. Top 3 technical risks with evidence (path + function/module)
5. Assumptions I will carry forward (max 5)

Then wait for my questions. Do not rebuild full context unless I explicitly say: "rebuild learning context".
```

## Case 2: Daily Q&A (incremental, no rebuild)

```text
Answer using existing codex-mem context in this project. Do incremental retrieval only.

Workflow:
1) `codex-mem ask "<my question>" --search-limit 6 --detail-limit 3 --code-top-k 6 --code-module-limit 4`
2) If the question is time/sequence dependent (what happened before/after), run:
   - `timeline <top-id> --before 2 --after 2`
3) Read at most 2 files only if evidence is missing.

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
1) `codex-mem ask "<task>" --search-limit 8 --detail-limit 5 --code-top-k 12 --code-module-limit 6`
2) If needed: mem-search -> timeline -> get-observations for historical decisions
3) Confirm assumptions and affected modules

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

- Cold start: savings are moderate because code grounding still dominates.
- Daily Q&A: savings are highest because retrieval can stay narrow and incremental.
- Deep forensics: savings remain high, but drop when you intentionally pull many Layer-3 details.
