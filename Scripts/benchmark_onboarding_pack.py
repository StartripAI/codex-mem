#!/usr/bin/env python3
"""
Benchmark codex-mem cold-start onboarding against a curated "onboarding pack" baseline.

Why this exists:
- Some teams cold-start by pasting a hand-picked set of core files (README + entrypoints + main flows).
- codex-mem cold-start should match that *correctness* while injecting a much smaller, structured context.

This script is publish-safe by default: it outputs aggregate token/time numbers only (no code snippets).
Token estimates use a simple chars/4 heuristic (same estimator used in codex-mem CLI JSON).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple


def now_utc_date() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def run_json(cmd: List[str], *, cwd: pathlib.Path) -> Tuple[Dict[str, Any], float]:
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
        )
    payload = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return payload, elapsed_ms


def safe_index_refresh(raw: Any) -> Dict[str, Any]:
    """
    Strip potentially sensitive fields (absolute paths, git hashes) from index_refresh payloads.

    We keep only publish-safe timing + reason signals.
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in ("refreshed", "reason", "index_time_ms", "index_exit_code"):
        if key in raw:
            out[key] = raw[key]
    return out


def read_files_token_sum(target_root: pathlib.Path, rel_paths: List[str]) -> Dict[str, Any]:
    total_tokens = 0
    per_file: List[Dict[str, Any]] = []
    seen_resolved: set[str] = set()
    for rel in rel_paths:
        p = (target_root / rel).resolve()
        if not p.exists() or not p.is_file():
            per_file.append({"path": rel, "present": False})
            continue
        # Avoid double-counting the same file on case-insensitive filesystems.
        key = str(p)
        if key.lower() in seen_resolved:
            per_file.append({"path": rel, "present": True, "deduped": True})
            continue
        seen_resolved.add(key.lower())
        txt = p.read_text(encoding="utf-8", errors="replace")
        tok = estimate_tokens(txt)
        total_tokens += tok
        per_file.append({"path": rel, "present": True, "chars": len(txt), "tokens_est": tok})
    return {"files": per_file, "tokens_est_total": total_tokens}


def default_pack_for_repo(target_root: pathlib.Path) -> List[str]:
    """
    A conservative, cross-project onboarding pack:
    - README + architecture docs if present
    - likely entrypoints (App.swift/main.swift/index.ts)
    - likely core flows (generation/logic/service)

    This is intentionally heuristic and file-name driven so it's reproducible without model calls.
    """
    candidates: List[str] = []
    for rel in (
        "README.md",
        "readme.md",
        "Documentation/ARCHITECTURE.md",
        "Documentation/Architecture.md",
        "Documentation/README.md",
        "docs/ARCHITECTURE.md",
        "docs/README.md",
    ):
        if (target_root / rel).exists():
            candidates.append(rel)

    # Entry points and common main files.
    for rel in (
        "App/App.swift",
        "App/main.swift",
        "main.swift",
        "src/index.ts",
        "src/main.ts",
        "Backend/src/index.ts",
        "backend/src/index.ts",
        "index.ts",
    ):
        if (target_root / rel).exists():
            candidates.append(rel)

    # Common "core flow" files by name patterns.
    patterns = (
        "*Generation*.swift",
        "*Orchestrator*.swift",
        "*Bootstrapper*.swift",
        "*Database*.swift",
        "*AI*Service*.swift",
        "src/routes/*.ts",
        "Backend/src/routes/*.ts",
    )
    for pat in patterns:
        for p in target_root.glob(pat):
            if p.is_file():
                candidates.append(str(p.relative_to(target_root)))

    # Dedupe preserving order.
    out: List[str] = []
    seen = set()
    for rel in candidates:
        rel = str(rel).replace("\\", "/")
        # Dedup by normalized path string first.
        if rel in seen:
            continue
        seen.add(rel)
        out.append(rel)
    return out[:20]


def dedupe_pack_paths(target_root: pathlib.Path, pack: List[str]) -> List[str]:
    out: List[str] = []
    seen_resolved: set[str] = set()
    for rel in pack:
        rel = str(rel).replace("\\", "/").strip()
        if not rel:
            continue
        p = (target_root / rel).resolve()
        if p.exists():
            key = str(p).lower()
            if key in seen_resolved:
                continue
            seen_resolved.add(key)
            try:
                rel = str(p.relative_to(target_root)).replace("\\", "/")
            except Exception:
                pass
        else:
            # Keep missing paths as-is (they won't contribute tokens).
            if rel in out:
                continue
        out.append(rel)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark codex-mem cold-start onboarding vs a curated file pack.")
    parser.add_argument("--target-root", default=".", help="Target repo root to benchmark.")
    parser.add_argument("--label", default="target-repo", help="Human-friendly label (avoid absolute paths).")
    parser.add_argument(
        "--pack",
        action="append",
        default=[],
        help="Relative file path to include in baseline pack. Repeatable. If omitted, a heuristic pack is used.",
    )
    parser.add_argument(
        "--question",
        default="Learn this project: north star, architecture, module map, entrypoints, main flow, persistence/output, top risks.",
        help="Onboarding question for codex-mem ask.",
    )
    parser.add_argument("--search-limit", type=int, default=6)
    parser.add_argument("--detail-limit", type=int, default=3)
    parser.add_argument("--code-top-k", type=int, default=10)
    parser.add_argument("--code-module-limit", type=int, default=6)
    parser.add_argument("--snippet-chars", type=int, default=1000)
    parser.add_argument("--repo-index-dir", default=".codex_knowledge_bench", help="Repo index dir for the benchmark.")
    parser.add_argument("--out", default="-", help="Output JSON path (default: stdout).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_root = pathlib.Path(args.target_root).resolve()
    label = str(args.label).strip() or "target-repo"

    script_dir = pathlib.Path(__file__).resolve().parent
    codex_mem = script_dir / "codex_mem.py"
    if not codex_mem.exists():
        raise FileNotFoundError(f"codex_mem.py not found next to this script: {codex_mem}")

    pack = [str(p).strip() for p in (args.pack or []) if str(p).strip()]
    if not pack:
        pack = default_pack_for_repo(target_root)
    pack = dedupe_pack_paths(target_root, pack)

    baseline = read_files_token_sum(target_root, pack)
    baseline_tokens = int(baseline.get("tokens_est_total", 0))

    # Ensure codex-mem memory DB is local to the target repo but isolated for benchmarking.
    mem_index_dir = ".codex_mem_bench"

    # `ask` is the cold-start entrypoint: it fuses repo grounding with memory retrieval (if any),
    # and auto-seeds a minimal baseline when the memory DB is empty.
    ask_cmd = [
        sys.executable,
        str(codex_mem),
        "--root",
        str(target_root),
        "--index-dir",
        mem_index_dir,
        "ask",
        str(args.question),
        "--search-limit",
        str(max(1, int(args.search_limit))),
        "--detail-limit",
        str(max(1, int(args.detail_limit))),
        "--code-top-k",
        str(max(1, int(args.code_top_k))),
        "--code-module-limit",
        str(max(1, int(args.code_module_limit))),
        "--snippet-chars",
        str(max(200, int(args.snippet_chars))),
        "--repo-index-dir",
        str(args.repo_index_dir),
    ]
    # Force a cold run by deleting the repo index dir if it exists.
    repo_index_path = (target_root / str(args.repo_index_dir)).resolve()
    if repo_index_path.exists():
        shutil.rmtree(repo_index_path, ignore_errors=True)

    ask_payload_cold, ask_ms_cold = run_json(ask_cmd, cwd=target_root)
    ask_tokens_cold = int(ask_payload_cold.get("token_estimate", {}).get("total", 0) or 0)
    index_refresh_cold = safe_index_refresh(ask_payload_cold.get("repo_context", {}).get("index_refresh", {}))

    ask_payload_warm, ask_ms_warm = run_json(ask_cmd, cwd=target_root)
    ask_tokens_warm = int(ask_payload_warm.get("token_estimate", {}).get("total", 0) or 0)
    index_refresh_warm = safe_index_refresh(ask_payload_warm.get("repo_context", {}).get("index_refresh", {}))

    saving_ratio = 0.0 if baseline_tokens <= 0 else (1.0 - (ask_tokens_warm / baseline_tokens))

    out = {
        "date": now_utc_date(),
        "benchmark": "onboarding_pack_v1",
        "target": {"label": label, "root_redacted": True},
        "question": str(args.question),
        "baseline_pack": {
            "file_count": len(pack),
            "tokens_est_total": baseline_tokens,
            # Publish-safe: paths only (relative), no file contents.
            "paths": pack,
        },
        "ask": {
            "cold": {
                "tokens_est_total": ask_tokens_cold,
                "time_ms": round(ask_ms_cold, 3),
                "index_refresh": index_refresh_cold,
            },
            "warm": {
                "tokens_est_total": ask_tokens_warm,
                "time_ms": round(ask_ms_warm, 3),
                "index_refresh": index_refresh_warm,
            },
        },
        "savings": {
            "context_saving_ratio": round(saving_ratio, 4),
            "context_saving_percent": round(saving_ratio * 100.0, 2),
        },
        "notes": [
            "Token estimates use chars/4 heuristic (same estimator used in codex-mem CLI JSON).",
            "Baseline pack is a curated set of full files a human might paste for onboarding.",
            "`ask` output size includes both memory retrieval (if any) and repo grounding context.",
        ],
    }

    out_str = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    if str(args.out).strip() and str(args.out).strip() != "-":
        out_path = pathlib.Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_str, encoding="utf-8")
    else:
        sys.stdout.write(out_str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
