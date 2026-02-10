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

## Marketing Metrics Snapshot (2026-02-10)

- Source: `Documentation/benchmarks/marketing_claims_20260210.json`
- Dataset scale: 30 sessions / 1230 events / 120 observations
- Token saving: `99.84%` (`379275` -> `596`)
- Startup to first context (Layer-1 search): `66.571 ms` median
- Startup speedup vs full-history load: `1.34x`

## Notes

- Always use sanitized demo data
- Keep query set stable for comparability

## Repo Onboarding Snapshot (2026-02-10)

- Source: `Documentation/benchmarks/repo_onboarding_codex_mem_20260210.json`
- Indexable corpus: 48 files / 230 chunks (~95,398 tokens estimated)
- Onboarding prompt (top-k=10, module-limit=6): ~2,723 tokens estimated
- Context reduction: `97.15%`
- Index build time (one-time): `~226 ms`
- Prompt generation (per question): `~74 ms`


## Scenario Savings Snapshot (2026-02-10)

Source file:
- `Documentation/benchmarks/scenario_savings_20260210.json`

| Scenario | Dataset Size (events/obs) | Token Saving | Startup Median (Layer-1) | Startup Speedup vs Full Load |
|---|---:|---:|---:|---:|
| Cold start (lean) | 14 / 8 | 63.98% | 62.151 ms | 1.023x |
| Cold start (deeper context) | 39 / 12 | 72.90% | 64.979 ms | 0.939x |
| Daily Q&A (standard) | 1230 / 120 | 99.84% | 61.555 ms | 1.324x |
| Daily Q&A (deep retrieval) | 1230 / 120 | 99.70% | 63.335 ms | 1.321x |
| Incident forensics (wide detail pull) | 1230 / 120 | 88.97% | 70.821 ms | 1.191x |

Interpretation:
- 99%+ savings is realistic in warm daily workflows.
- Cold start savings are lower because initial understanding still requires code reading.
- Forensics savings remain high, but drop when you intentionally pull many Layer-3 details.
