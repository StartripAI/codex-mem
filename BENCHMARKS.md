# Benchmarks

Track retrieval quality and efficiency across versions.

## Metrics

- Token estimate (search vs mem-search)
- Retrieval latency (CLI call time)
- Recall proxy (expected term hit rank)
- Overlap/Jaccard between retrieval modes

## Standard Query Set

Stored at:
- `Documentation/benchmarks/queries.json`

## Run Benchmark

```bash
python Scripts/compare_search_modes.py --root . --project demo-recording
```

Marketing metrics benchmark (token + startup):

```bash
python3 Scripts/benchmark_marketing_claim.py --root . --out Documentation/benchmarks/marketing_claims_latest.json
```

## Latest Result Template

| Version | Query Count | Avg Token (search) | Avg Token (mem-search) | Avg Hit Rank (search) | Avg Hit Rank (mem-search) | Avg Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| v0.2.0 | - | - | - | - | - | - |

## Marketing Metrics Snapshot (2026-02-11)

- Source: `Documentation/benchmarks/marketing_claims_20260211.json`
- Dataset scale: 30 sessions / 1230 events / 120 observations
- Token saving: `99.84%` (`379275` -> `596`)
- Startup to first context (Layer-1 search): `61.508 ms` median
- Startup speedup vs full-history load: `1.326x`

## Notes

- Always use sanitized demo data
- Keep query set stable for comparability

## Repo Onboarding Snapshot (2026-02-11)

- Source: `Documentation/benchmarks/repo_onboarding_codex_mem_20260211.json`
- Indexable corpus: 56 files / 274 chunks (~113,902 tokens estimated)
- Onboarding prompt (top-k=10, module-limit=6): ~2,557 tokens estimated
- Context reduction: `97.76%`
- Index build time (one-time): `~337 ms`
- Prompt generation (per question): `~83 ms`

## Onboarding Pack Snapshot (2026-02-11)

This benchmarks `codex-mem ask` against a curated onboarding pack (a set of full files a human might paste).

- Source: `Documentation/benchmarks/onboarding_pack_codex_mem_rich_20260211.json`
- Baseline pack (9 files): ~58,292 tokens estimated
- `ask` context: ~3,086 tokens estimated
- Context reduction: `94.71%`
- Cold `ask` (index missing): `~422 ms`
- Warm `ask` (index up-to-date): `~163 ms`


## Scenario Savings Snapshot (2026-02-11)

Source file:
- `Documentation/benchmarks/scenario_savings_20260211.json`

| Scenario | Dataset Size (events/obs) | Token Saving | Startup Median (Layer-1) | Startup Speedup vs Full Load |
|---|---:|---:|---:|---:|
| Cold start (lean) | 14 / 8 | 61.70% | 55.956 ms | 0.973x |
| Cold start (deeper context) | 39 / 12 | 72.62% | 56.312 ms | 0.992x |
| Daily Q&A (standard) | 1230 / 120 | 99.84% | 61.937 ms | 1.348x |
| Daily Q&A (deep retrieval) | 1230 / 120 | 99.69% | 65.155 ms | 1.314x |
| Incident forensics (wide detail pull) | 1230 / 120 | 88.97% | 67.062 ms | 1.236x |

Interpretation:
- 99%+ savings is realistic in warm daily workflows.
- Cold start savings are lower because initial understanding still requires code reading.
- Forensics savings remain high, but drop when you intentionally pull many Layer-3 details.
