#!/usr/bin/env python3
"""
Generate a scenario matrix for marketing-facing "savings" claims.

Why this script exists:
- `benchmark_marketing_claim.py` benchmarks one scenario at a time.
- README/BENCHMARKS want a reproducible table across common use cases.

This script runs a fixed set of cases (cold start / daily / forensics) and writes
one JSON file you can reference from docs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from typing import Any, Dict, List

from benchmark_marketing_claim import Scenario, run_benchmark


def now_utc_date() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark scenario savings matrix for codex-mem")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--out",
        default="Documentation/benchmarks/scenario_savings_latest.json",
        help="Output JSON file path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.root).resolve()
    script = root / "Scripts" / "codex_mem.py"
    if not script.exists():
        raise FileNotFoundError(f"codex_mem.py not found at {script}")

    # Keep these aligned with the README copy so it's always reproducible.
    cases: List[Dict[str, Any]] = [
        {
            "case": "cold_start_lean",
            "project": "cold-start-lean",
            "scenario": Scenario(
                name="scenario_matrix_v1",
                sessions=2,
                tool_events_per_session=4,
                noise_chars=65,
                chunks_per_event=13,
                stage1_limit=8,
                details_limit=3,
                timeline_before=1,
                timeline_after=1,
                startup_samples=3,
                seed=31,
            ),
        },
        {
            "case": "cold_start_deeper_context",
            "project": "cold-start-deep",
            "scenario": Scenario(
                name="scenario_matrix_v1",
                sessions=3,
                tool_events_per_session=10,
                noise_chars=65,
                chunks_per_event=13,
                stage1_limit=12,
                details_limit=8,
                timeline_before=3,
                timeline_after=3,
                startup_samples=3,
                seed=41,
            ),
        },
        {
            "case": "daily_qa_standard",
            "project": "daily-standard",
            "scenario": Scenario(
                name="scenario_matrix_v1",
                sessions=30,
                tool_events_per_session=38,
                noise_chars=65,
                chunks_per_event=13,
                stage1_limit=20,
                details_limit=8,
                timeline_before=4,
                timeline_after=4,
                startup_samples=5,
                seed=19,
            ),
        },
        {
            "case": "daily_qa_deep_retrieval",
            "project": "daily-deep",
            "scenario": Scenario(
                name="scenario_matrix_v1",
                sessions=30,
                tool_events_per_session=38,
                noise_chars=65,
                chunks_per_event=13,
                stage1_limit=30,
                details_limit=20,
                timeline_before=8,
                timeline_after=8,
                startup_samples=5,
                seed=23,
            ),
        },
        {
            "case": "incident_forensics_wide_detail_pull",
            "project": "incident-forensics",
            "scenario": Scenario(
                name="scenario_matrix_v1",
                sessions=30,
                tool_events_per_session=38,
                noise_chars=65,
                chunks_per_event=13,
                stage1_limit=120,
                details_limit=120,
                timeline_before=20,
                timeline_after=20,
                startup_samples=5,
                seed=29,
            ),
        },
    ]

    out_cases: List[Dict[str, Any]] = []
    query_text = None
    for item in cases:
        case_name = str(item["case"])
        project = str(item["project"])
        scenario: Scenario = item["scenario"]
        result = run_benchmark(root=root, script=script, scenario=scenario, project=project)
        query_text = query_text or str(result.get("query", {}).get("text") or "")
        out_cases.append({"case": case_name, **result})

    payload = {
        "date": now_utc_date(),
        "benchmark_script": "Scripts/benchmark_marketing_claim.py",
        "query": query_text or "",
        "cases": out_cases,
        "notes": [
            "Each case generates its own synthetic memory dataset via codex-mem lifecycle hooks for reproducibility.",
            "Token counts are estimates using the same chars/4 heuristic codex-mem uses in CLI outputs.",
            "Startup is local CLI time-to-first-context (Layer-1 search median) vs full-history load median.",
        ],
    }

    out_path = pathlib.Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"ok": True, "out": str(out_path), "case_count": len(out_cases)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

