# Failure Case Library

This document captures retrieval failures and mitigation patterns.

## Case 1 - False Positive from Broad Keyword

- Symptom: query returns unrelated events due to generic token overlap.
- Trigger: short keyword-only query.
- Mitigation:
  - use `mem-search` with intent phrase
  - add session/time bounds
  - add expected term checks in benchmark

## Case 2 - False Negative on Private Records

- Symptom: expected record missing in default search.
- Trigger: record tagged with `--privacy-tag private`.
- Mitigation:
  - run with `--include-private` for authorized review
  - verify privacy policy was intended

## Case 3 - Timeline Context Drift

- Symptom: timeline around anchor misses critical neighboring event.
- Trigger: dense sessions with many similarly-timed events.
- Mitigation:
  - increase `--before/--after`
  - use `get-observations` on multiple anchors

## Case 4 - Over-Compaction in Beta Mode

- Symptom: compacted output loses detail needed for debugging.
- Trigger: `beta` + `beta_endless_mode=on`.
- Mitigation:
  - switch to stable or disable endless mode
  - replay command with explicit non-compact logging
