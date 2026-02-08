#!/usr/bin/env python3
"""
Benchmark marketing-facing metrics for codex-mem.

Primary outputs:
- token saving percentage (naive full load vs progressive retrieval)
- startup time (time-to-first-context via Layer-1 search)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sqlite3
import string
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Scenario:
    name: str
    sessions: int
    tool_events_per_session: int
    noise_chars: int
    chunks_per_event: int
    stage1_limit: int
    details_limit: int
    timeline_before: int
    timeline_after: int
    startup_samples: int
    seed: int


DEFAULT_SCENARIO = Scenario(
    name="marketing_default",
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
)


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def run_cli(
    *,
    script: pathlib.Path,
    root: pathlib.Path,
    index_dir: str,
    args: List[str],
) -> Tuple[Dict[str, Any], float]:
    cmd = [sys.executable, str(script), "--root", str(root), "--index-dir", index_dir, *args]
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )
    text = proc.stdout.strip()
    return (json.loads(text) if text else {}), elapsed_ms


def build_tool_output(
    rng: random.Random,
    *,
    session_idx: int,
    event_idx: int,
    noise_chars: int,
    chunks_per_event: int,
) -> str:
    topics = [
        "stream orchestrator",
        "privacy redaction",
        "memory retrieval",
        "timeline context",
        "token budget",
        "search ranking",
        "session summary",
        "delta handling",
        "cache invalidation",
    ]
    parts: List[str] = []
    for chunk_idx in range(chunks_per_event):
        topic = topics[(session_idx + event_idx + chunk_idx) % len(topics)]
        noise = "".join(rng.choices(string.ascii_lowercase + " ", k=noise_chars)).strip()
        parts.append(f"{topic} :: task {session_idx}-{event_idx}-{chunk_idx} :: {noise}")
    return " ".join(parts)


def median(values: List[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def run_benchmark(
    *,
    root: pathlib.Path,
    script: pathlib.Path,
    scenario: Scenario,
    project: str,
) -> Dict[str, Any]:
    rng = random.Random(scenario.seed)
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="codex_mem_marketing_"))
    index_dir = str(temp_root / ".codex_mem")

    run_cli(script=script, root=root, index_dir=index_dir, args=["init", "--project", project])

    for s in range(scenario.sessions):
        session_id = f"M{s:03d}"
        run_cli(
            script=script,
            root=root,
            index_dir=index_dir,
            args=["session-start", session_id, "--project", project, "--title", f"Marketing Session {session_id}"],
        )
        run_cli(
            script=script,
            root=root,
            index_dir=index_dir,
            args=[
                "user-prompt-submit",
                session_id,
                "refactor stream memory retrieval and privacy controls",
                "--project",
                project,
            ],
        )
        for i in range(scenario.tool_events_per_session):
            content = build_tool_output(
                rng,
                session_idx=s,
                event_idx=i,
                noise_chars=scenario.noise_chars,
                chunks_per_event=scenario.chunks_per_event,
            )
            run_cli(
                script=script,
                root=root,
                index_dir=index_dir,
                args=[
                    "post-tool-use",
                    session_id,
                    "shell",
                    content,
                    "--project",
                    project,
                    "--title",
                    f"tool {i}",
                ],
            )
        run_cli(
            script=script,
            root=root,
            index_dir=index_dir,
            args=["session-end", session_id, "--project", project, "--content", "done"],
        )

    db = sqlite3.connect(str(pathlib.Path(index_dir) / "codex_mem.sqlite3"))
    db.row_factory = sqlite3.Row
    event_rows = db.execute("SELECT id, content FROM events").fetchall()
    obs_rows = db.execute("SELECT id, body FROM observations").fetchall()

    naive_tokens = sum(estimate_tokens(str(row["content"])) for row in event_rows) + sum(
        estimate_tokens(str(row["body"])) for row in obs_rows
    )

    query = "how did we handle memory retrieval privacy and stream orchestrator issues"
    layer1_payload, _ = run_cli(
        script=script,
        root=root,
        index_dir=index_dir,
        args=["search", query, "--project", project, "--limit", str(scenario.stage1_limit)],
    )
    layer1 = list(layer1_payload.get("results", []))
    layer1_tokens = int(layer1_payload.get("token_estimate_total", 0))

    selected_ids = [item["id"] for item in layer1[: scenario.details_limit]]
    if selected_ids:
        details_payload, _ = run_cli(
            script=script,
            root=root,
            index_dir=index_dir,
            args=["get-observations", *selected_ids],
        )
        layer3_tokens = int(details_payload.get("token_estimate_total", 0))
    else:
        layer3_tokens = 0

    timeline_tokens = 0
    timeline_anchor = None
    if layer1:
        timeline_anchor = str(layer1[0]["id"])
        timeline_payload, _ = run_cli(
            script=script,
            root=root,
            index_dir=index_dir,
            args=[
                "timeline",
                timeline_anchor,
                "--before",
                str(scenario.timeline_before),
                "--after",
                str(scenario.timeline_after),
            ],
        )
        before = list(timeline_payload.get("before", []))
        after = list(timeline_payload.get("after", []))
        timeline_tokens = sum(int(item.get("token_estimate", 0)) for item in before + after)

    progressive_tokens = layer1_tokens + timeline_tokens + layer3_tokens
    token_saving_ratio = 0.0 if naive_tokens <= 0 else (1.0 - progressive_tokens / naive_tokens)

    all_ids = [f"E{int(row['id'])}" for row in event_rows] + [f"O{int(row['id'])}" for row in obs_rows]

    # Warmup for stable timing.
    run_cli(
        script=script,
        root=root,
        index_dir=index_dir,
        args=["search", query, "--project", project, "--limit", str(scenario.stage1_limit)],
    )
    run_cli(
        script=script,
        root=root,
        index_dir=index_dir,
        args=["get-observations", *all_ids[: min(len(all_ids), 200)]],
    )

    stage1_times: List[float] = []
    full_times: List[float] = []
    for _ in range(scenario.startup_samples):
        _, elapsed = run_cli(
            script=script,
            root=root,
            index_dir=index_dir,
            args=["search", query, "--project", project, "--limit", str(scenario.stage1_limit)],
        )
        stage1_times.append(elapsed)
    for _ in range(scenario.startup_samples):
        _, elapsed = run_cli(
            script=script,
            root=root,
            index_dir=index_dir,
            args=["get-observations", *all_ids],
        )
        full_times.append(elapsed)

    stage1_median = median(stage1_times)
    full_median = median(full_times)
    startup_speedup = 0.0 if stage1_median <= 0 else (full_median / stage1_median)

    return {
        "scenario": {
            "name": scenario.name,
            "sessions": scenario.sessions,
            "tool_events_per_session": scenario.tool_events_per_session,
            "noise_chars": scenario.noise_chars,
            "chunks_per_event": scenario.chunks_per_event,
            "stage1_limit": scenario.stage1_limit,
            "details_limit": scenario.details_limit,
            "timeline_before": scenario.timeline_before,
            "timeline_after": scenario.timeline_after,
            "startup_samples": scenario.startup_samples,
            "seed": scenario.seed,
        },
        "dataset": {
            "project": project,
            "events": len(event_rows),
            "observations": len(obs_rows),
            "total_items": len(all_ids),
        },
        "token": {
            "naive_full_load": naive_tokens,
            "progressive_load": progressive_tokens,
            "layer1_tokens": layer1_tokens,
            "layer2_timeline_tokens": timeline_tokens,
            "layer3_detail_tokens": layer3_tokens,
            "saving_ratio": round(token_saving_ratio, 4),
            "saving_percent": round(token_saving_ratio * 100.0, 2),
        },
        "startup_ms": {
            "stage1_time_to_first_context_samples": [round(v, 3) for v in stage1_times],
            "full_history_load_samples": [round(v, 3) for v in full_times],
            "stage1_median": round(stage1_median, 3),
            "full_history_median": round(full_median, 3),
            "speedup_x": round(startup_speedup, 3),
        },
        "query": {
            "text": query,
            "timeline_anchor": timeline_anchor,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark codex-mem marketing claims")
    parser.add_argument("--root", default=".")
    parser.add_argument("--project", default="marketing-bench")
    parser.add_argument("--scenario-name", default=DEFAULT_SCENARIO.name)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SCENARIO.sessions)
    parser.add_argument("--events-per-session", type=int, default=DEFAULT_SCENARIO.tool_events_per_session)
    parser.add_argument("--noise-chars", type=int, default=DEFAULT_SCENARIO.noise_chars)
    parser.add_argument("--chunks-per-event", type=int, default=DEFAULT_SCENARIO.chunks_per_event)
    parser.add_argument("--stage1-limit", type=int, default=DEFAULT_SCENARIO.stage1_limit)
    parser.add_argument("--details-limit", type=int, default=DEFAULT_SCENARIO.details_limit)
    parser.add_argument("--timeline-before", type=int, default=DEFAULT_SCENARIO.timeline_before)
    parser.add_argument("--timeline-after", type=int, default=DEFAULT_SCENARIO.timeline_after)
    parser.add_argument("--startup-samples", type=int, default=DEFAULT_SCENARIO.startup_samples)
    parser.add_argument("--seed", type=int, default=DEFAULT_SCENARIO.seed)
    parser.add_argument(
        "--out",
        default="Documentation/benchmarks/marketing_claims_latest.json",
        help="Output JSON file path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.root).resolve()
    script = root / "Scripts" / "codex_mem.py"
    if not script.exists():
        raise FileNotFoundError(f"codex_mem.py not found at {script}")

    scenario = Scenario(
        name=str(args.scenario_name),
        sessions=max(1, int(args.sessions)),
        tool_events_per_session=max(1, int(args.events_per_session)),
        noise_chars=max(10, int(args.noise_chars)),
        chunks_per_event=max(2, int(args.chunks_per_event)),
        stage1_limit=max(1, int(args.stage1_limit)),
        details_limit=max(1, int(args.details_limit)),
        timeline_before=max(0, int(args.timeline_before)),
        timeline_after=max(0, int(args.timeline_after)),
        startup_samples=max(1, int(args.startup_samples)),
        seed=int(args.seed),
    )

    result = run_benchmark(root=root, script=script, scenario=scenario, project=str(args.project))
    out_path = pathlib.Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "out": str(out_path),
                "token_saving_percent": result["token"]["saving_percent"],
                "stage1_median_ms": result["startup_ms"]["stage1_median"],
                "startup_speedup_x": result["startup_ms"]["speedup_x"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
