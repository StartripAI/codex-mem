#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def run_json(cmd: List[str], cwd: pathlib.Path) -> Tuple[Dict[str, Any], float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}\n{proc.stdout}")
    payload = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return payload, elapsed_ms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ask prompt compaction: compact vs legacy.")
    p.add_argument("--root", default=".")
    p.add_argument(
        "--question",
        default="Learn this project: project goal, architecture, module map, entrypoints, main flow, persistence, top risks",
    )
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--out", default="-")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.root).resolve()
    script = root / "Scripts" / "codex_mem.py"

    compact_times: List[float] = []
    legacy_times: List[float] = []
    compact_tokens: List[int] = []
    legacy_tokens: List[int] = []
    compact_cov: List[int] = []
    legacy_cov: List[int] = []

    for _ in range(max(1, int(args.runs))):
        cmd_base = [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "ask",
            str(args.question),
            "--search-limit",
            "6",
            "--detail-limit",
            "3",
            "--code-top-k",
            "10",
            "--code-module-limit",
            "6",
            "--snippet-chars",
            "1000",
        ]

        compact_payload, compact_ms = run_json(cmd_base + ["--prompt-style", "compact"], root)
        legacy_payload, legacy_ms = run_json(cmd_base + ["--prompt-style", "legacy"], root)

        compact_prompt = str(compact_payload.get("suggested_prompt", ""))
        legacy_prompt = str(legacy_payload.get("suggested_prompt", ""))

        compact_tokens.append(estimate_tokens(compact_prompt))
        legacy_tokens.append(estimate_tokens(legacy_prompt))
        compact_times.append(compact_ms)
        legacy_times.append(legacy_ms)

        compact_cov.append(1 if bool((compact_payload.get("coverage_gate") or {}).get("pass")) else 0)
        legacy_cov.append(1 if bool((legacy_payload.get("coverage_gate") or {}).get("pass")) else 0)

    avg_compact_tokens = statistics.mean(compact_tokens)
    avg_legacy_tokens = statistics.mean(legacy_tokens)
    saving_ratio = 0.0 if avg_legacy_tokens <= 0 else (1.0 - avg_compact_tokens / avg_legacy_tokens)

    out = {
        "benchmark": "prompt_compaction_v1",
        "question": args.question,
        "runs": max(1, int(args.runs)),
        "compact": {
            "tokens_est_samples": compact_tokens,
            "tokens_est_avg": round(avg_compact_tokens, 2),
            "time_ms_samples": [round(v, 3) for v in compact_times],
            "time_ms_avg": round(statistics.mean(compact_times), 3),
            "coverage_pass_rate": round(sum(compact_cov) / len(compact_cov), 4),
        },
        "legacy": {
            "tokens_est_samples": legacy_tokens,
            "tokens_est_avg": round(avg_legacy_tokens, 2),
            "time_ms_samples": [round(v, 3) for v in legacy_times],
            "time_ms_avg": round(statistics.mean(legacy_times), 3),
            "coverage_pass_rate": round(sum(legacy_cov) / len(legacy_cov), 4),
        },
        "savings": {
            "token_saving_ratio_vs_legacy": round(saving_ratio, 4),
            "token_saving_percent_vs_legacy": round(saving_ratio * 100.0, 2),
        },
    }

    out_text = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    if str(args.out).strip() and str(args.out).strip() != "-":
        out_path = pathlib.Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text, encoding="utf-8")
    else:
        sys.stdout.write(out_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
