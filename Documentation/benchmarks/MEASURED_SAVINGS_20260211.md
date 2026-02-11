# Measured Startup + Context Savings (2026-02-11)

I built `codex-mem` because the real bottleneck in AI coding workflows is **context management**:

- new sessions start blank
- dumping everything into context doesn’t scale
- I want retrieval to be evidence-first and reproducible

This page is a snapshot of what I measured locally (scripts included in this repo).

## 1) Cold Start (Curated Onboarding Pack vs `ask`)

Some teams cold-start by pasting a hand-picked “onboarding pack” of full files (README + entrypoints + core flows).

This benchmark compares that baseline against `codex-mem ask` (which fuses repo grounding + progressive memory retrieval, and auto-seeds a minimal baseline when memory is empty).

**Result (context payload size):**
- Baseline pack (9 full files): **58,292** tokens (estimate)
- `ask` context: **3,086** tokens (estimate)
- **Context reduction:** **94.71%**

**Result (latency):**
- Cold `ask` (repo index missing): **421.824 ms**
- Warm `ask` (repo index up-to-date): **163.386 ms**

Source:
- `Documentation/benchmarks/onboarding_pack_codex_mem_rich_20260211.json`

Reproduce:
```bash
python3 Scripts/benchmark_onboarding_pack.py \
  --target-root . \
  --label "codex-mem (this repo)" \
  --pack README.md \
  --pack Documentation/ARCHITECTURE.md \
  --pack Scripts/codex_mem.py \
  --pack Scripts/repo_knowledge.py \
  --pack Scripts/codex_mem_web.py \
  --pack Scripts/codex_mem_mcp.py \
  --pack Documentation/INSTALLATION.md \
  --pack Documentation/MCP_TOOLS.md \
  --pack Skills/codex-mem/SKILL.md \
  --out Documentation/benchmarks/onboarding_pack_codex_mem_rich_20260211.json
```

## 2) Cold Start (Repo Grounding Prompt vs Full Indexable Corpus)

For very large repos, a common “naive baseline” is loading a huge amount of code/docs into context.

This benchmark measures `repo_knowledge.py prompt` (the code-grounding index used by `codex-mem ask`) against the full indexable corpus.

**Result (context payload size):**
- Full indexable corpus: **113,902** tokens (estimate)
- Onboarding prompt (top-k=10, module-limit=6): **2,557** tokens (estimate)
- **Context reduction:** **97.76%**

**Result (latency):**
- Index build (one-time per repo): **337.441 ms**
- Onboarding prompt generation (per question): **83.041 ms**

Source:
- `Documentation/benchmarks/repo_onboarding_codex_mem_20260211.json`

Reproduce:
```bash
python3 Scripts/benchmark_repo_onboarding.py \
  --target-root . \
  --label "codex-mem (this repo)" \
  --index-dir .codex_knowledge_bench \
  --all-files on \
  --ignore-dir .codex_mem \
  --top-k 10 \
  --module-limit 6 \
  --snippet-chars 1000 \
  --out Documentation/benchmarks/repo_onboarding_codex_mem_20260211.json
```

## 3) Warm Daily Workflow (Memory Only)

This measures `codex-mem`’s **3-layer progressive disclosure retrieval** on a simulated “daily work” dataset (lots of historical memory).

**Result (context payload size):**
- Naive “load full history”: **379,275** tokens (estimate)
- Progressive load (Layer 1 + Layer 2 + Layer 3): **596** tokens (estimate)
- **Context reduction:** **99.84%**

**Result (time-to-first-context):**
- Layer-1 search median: **61.508 ms**
- Full-history load median: **81.534 ms**
- **Speedup:** **1.326x** (Layer-1 is faster vs full-history load)

Source:
- `Documentation/benchmarks/marketing_claims_20260211.json`

Reproduce:
```bash
python3 Scripts/benchmark_marketing_claim.py \
  --root . \
  --out Documentation/benchmarks/marketing_claims_20260211.json
```

## Notes

- Token estimates use a simple `chars/4` heuristic (not true BPE).
- Results vary by repo size, language mix, and how aggressively you choose to pull Layer-3 details.
- The goal is **correctness-first grounding** with a **controlled context payload**, not “never read code”.

