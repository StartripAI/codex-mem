#!/usr/bin/env python3
"""
Benchmark repo_knowledge onboarding context size + latency for a target repository.

This is intended for *cold-start* project understanding:
- build a local repo_knowledge index (one-time)
- generate an onboarding prompt (top-K chunks)
- compare against a naive "load everything" baseline (entire chunk corpus)

Outputs are aggregate-only (no code snippets) so you can publish results safely.
Token estimates use a simple chars/4 heuristic to match codex-mem's memory token estimator.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, Tuple


def now_utc_date() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def run_cmd(cmd: list[str], *, cwd: pathlib.Path) -> Tuple[str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  cwd: {cwd}\n"
            f"  exit: {proc.returncode}\n"
            f"  stderr:\n{proc.stderr}\n"
            f"  stdout:\n{proc.stdout}\n"
        )
    return proc.stdout, elapsed_ms


def safe_index_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    # Keep only aggregate fields safe for publishing.
    keep = {
        "ok",
        "index_version",
        "file_count",
        "chunk_count",
        "module_count",
        "avg_chunk_tokens",
        "embedding_provider",
        "vector_dim",
    }
    out: Dict[str, Any] = {}
    for key in keep:
        if key in raw:
            out[key] = raw[key]
    return out


def db_aggregate_tokens(db_path: pathlib.Path) -> Dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        # chunks.text is the largest corpus; treat it as the naive "load everything" baseline.
        total_chars = int(conn.execute("SELECT COALESCE(SUM(LENGTH(text)), 0) AS n FROM chunks").fetchone()["n"])
        file_count = int(conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"])
        chunk_count = int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])
    finally:
        conn.close()
    return {
        "files": file_count,
        "chunks": chunk_count,
        "full_context_chars": total_chars,
        "full_context_tokens_est": max(0, math.ceil(total_chars / 4)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark repo_knowledge onboarding on a target repo")
    parser.add_argument(
        "--target-root",
        required=True,
        help="Target repository root path to index (not the codex-mem repo).",
    )
    parser.add_argument(
        "--label",
        default="target-repo",
        help="Human-friendly label for publishing results (avoid absolute paths).",
    )
    parser.add_argument(
        "--index-dir",
        default=".codex_knowledge",
        help="Index directory under target repo root.",
    )
    parser.add_argument(
        "--all-files",
        choices=["on", "off"],
        default="off",
        help="Index all files via filesystem walk (on) vs only git-tracked files (off).",
    )
    parser.add_argument(
        "--ignore-dir",
        action="append",
        default=[],
        help="Additional directory name(s) to ignore during indexing. Repeatable.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top chunks included in onboarding prompt.",
    )
    parser.add_argument(
        "--module-limit",
        type=int,
        default=4,
        help="Module recall limit for onboarding prompt.",
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=1000,
        help="Max chars per chunk snippet in prompt.",
    )
    parser.add_argument(
        "--question",
        default="Learn this project: project goal, architecture, modules, entrypoints, main flow, persistence/output, top risks.",
        help="Onboarding question used for prompt generation.",
    )
    parser.add_argument(
        "--cleanup",
        choices=["on", "off"],
        default="off",
        help="Delete the generated index directory after the benchmark.",
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Output JSON path (default: stdout).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_root = pathlib.Path(args.target_root).resolve()
    label = str(args.label).strip() or "target-repo"
    index_dir = str(args.index_dir).strip() or ".codex_knowledge"

    script_dir = pathlib.Path(__file__).resolve().parent
    repo_knowledge = script_dir / "repo_knowledge.py"
    if not repo_knowledge.exists():
        raise FileNotFoundError(f"Missing repo_knowledge.py next to this script: {repo_knowledge}")

    # 1) Build index (one-time cost)
    index_cmd = [
        sys.executable,
        str(repo_knowledge),
        "--root",
        str(target_root),
        "--index-dir",
        index_dir,
        "index",
        "--embedding-provider",
        "local",
    ]
    if str(args.all_files).lower() == "on":
        index_cmd.append("--all-files")
    for ignored in args.ignore_dir or []:
        if ignored and str(ignored).strip():
            index_cmd.extend(["--ignore-dir", str(ignored).strip()])
    index_out, index_ms = run_cmd(index_cmd, cwd=target_root)
    index_payload = json.loads(index_out) if index_out.strip() else {}

    db_path = (target_root / index_dir / "repo_knowledge.sqlite3").resolve()
    db_agg_start = time.perf_counter()
    db_agg = db_aggregate_tokens(db_path)
    full_scan_ms = (time.perf_counter() - db_agg_start) * 1000.0

    # 2) Generate an onboarding prompt (what you would actually inject as context)
    prompt_cmd = [
        sys.executable,
        str(repo_knowledge),
        "--root",
        str(target_root),
        "--index-dir",
        index_dir,
        "prompt",
        str(args.question),
        "--top-k",
        str(max(1, int(args.top_k))),
        "--module-limit",
        str(max(1, int(args.module_limit))),
        "--snippet-chars",
        str(max(100, int(args.snippet_chars))),
    ]
    prompt_text, prompt_ms = run_cmd(prompt_cmd, cwd=target_root)
    prompt_tokens = estimate_tokens(prompt_text)
    prompt_chars = len(prompt_text)

    full_tokens = int(db_agg.get("full_context_tokens_est", 0))
    saving_ratio = 0.0 if full_tokens <= 0 else (1.0 - (prompt_tokens / full_tokens))
    speedup_x = 0.0 if prompt_ms <= 0 else (full_scan_ms / prompt_ms)

    out = {
        "date": now_utc_date(),
        "benchmark": "repo_knowledge_onboarding_v1",
        "target": {
            "label": label,
            "index_dir": index_dir,
            "all_files": bool(str(args.all_files).lower() == "on"),
            "extra_ignored_dirs": [str(v) for v in (args.ignore_dir or []) if str(v).strip()],
        },
        "index": {
            "time_ms": round(index_ms, 3),
            "summary": safe_index_summary(index_payload),
        },
        "baseline_full_context": {
            "files": db_agg.get("files", 0),
            "chunks": db_agg.get("chunks", 0),
            "chars": db_agg.get("full_context_chars", 0),
            "tokens_est": full_tokens,
            "scan_time_ms": round(full_scan_ms, 3),
        },
        "onboarding_prompt": {
            "question": str(args.question),
            "top_k": int(args.top_k),
            "module_limit": int(args.module_limit),
            "snippet_chars": int(args.snippet_chars),
            "chars": prompt_chars,
            "tokens_est": prompt_tokens,
            "time_ms": round(prompt_ms, 3),
        },
        "savings": {
            "context_saving_ratio": round(saving_ratio, 4),
            "context_saving_percent": round(saving_ratio * 100.0, 2),
            "context_speedup_x": round(speedup_x, 3),
        },
        "notes": [
            "Token estimates use chars/4 heuristic (same as codex-mem memory token estimator).",
            "Baseline is the full chunk corpus in repo_knowledge index (indexable code/docs only).",
            "Onboarding prompt is what repo_knowledge prints via `prompt` (top-K chunks + module recall).",
            "Indexing time is a one-time cost per repo; prompt generation is the per-question cost.",
        ],
    }

    if str(args.out).strip() and str(args.out).strip() != "-":
        out_path = pathlib.Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "out": str(out_path)}, ensure_ascii=False))
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))

    if str(args.cleanup).lower() == "on":
        try:
            shutil.rmtree(target_root / index_dir)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
