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

## Marketing Metrics Snapshot (2026-02-08)

- Source: `Documentation/benchmarks/marketing_claims_20260208.json`
- Dataset scale: 30 sessions / 1230 events / 120 observations
- Token saving: `99.84%` (`379275` -> `596`)
- Startup to first context (Layer-1 search): `59.548 ms` median
- Startup speedup vs full-history load: `1.32x`

## Notes

- Always use sanitized demo data
- Keep query set stable for comparability
