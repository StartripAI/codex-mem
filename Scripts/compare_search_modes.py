#!/usr/bin/env python3
"""
Compare `search` and `nl-search` retrieval behavior on a query set.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Tuple


def run_cli(root: pathlib.Path, index_dir: str, args: List[str]) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(root / "Scripts" / "codex_mem.py"),
        "--root",
        str(root),
        "--index-dir",
        index_dir,
        *args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI failed: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")
    txt = proc.stdout.strip()
    return json.loads(txt) if txt else {}


def jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def hit_expected(results: List[Dict[str, Any]], terms: List[str]) -> int:
    if not terms:
        return 0
    tset = [t.lower() for t in terms]
    for idx, item in enumerate(results):
        hay = f"{item.get('title','')} {item.get('kind','')}".lower()
        if any(t in hay for t in tset):
            return idx + 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare search vs mem-search retrieval quality")
    parser.add_argument("--root", default=".")
    parser.add_argument("--index-dir", default=".codex_mem")
    parser.add_argument("--project", default="default")
    parser.add_argument("--queries", default="Documentation/benchmarks/queries.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", default="dist/benchmarks/search_vs_mem-search.json")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    queries_path = pathlib.Path(args.queries)
    if not queries_path.is_absolute():
        queries_path = root / queries_path
    data = json.loads(queries_path.read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []
    overlaps: List[float] = []
    token_search: List[int] = []
    token_nl: List[int] = []
    rank_search: List[int] = []
    rank_nl: List[int] = []

    for item in data:
        query = str(item.get("query", "")).strip()
        expected_terms = [str(t) for t in item.get("expected_terms", [])]
        if not query:
            continue

        base = run_cli(root, args.index_dir, ["search", query, "--project", args.project, "--limit", str(args.limit)])
        nl = run_cli(root, args.index_dir, ["nl-search", query, "--project", args.project, "--limit", str(args.limit)])

        base_results = list(base.get("results", []))
        nl_results = list(nl.get("results", []))
        base_ids = [str(r.get("id")) for r in base_results]
        nl_ids = [str(r.get("id")) for r in nl_results]

        ov = jaccard(base_ids, nl_ids)
        overlaps.append(ov)
        token_search.append(int(base.get("token_estimate_total", 0)))
        token_nl.append(int(nl.get("token_estimate_total", 0)))

        rs = hit_expected(base_results, expected_terms)
        rn = hit_expected(nl_results, expected_terms)
        if rs:
            rank_search.append(rs)
        if rn:
            rank_nl.append(rn)

        rows.append(
            {
                "query": query,
                "expected_terms": expected_terms,
                "search_result_count": len(base_results),
                "nl_result_count": len(nl_results),
                "search_token_estimate": int(base.get("token_estimate_total", 0)),
                "nl_token_estimate": int(nl.get("token_estimate_total", 0)),
                "topk_jaccard": round(ov, 4),
                "expected_hit_rank_search": rs,
                "expected_hit_rank_nl": rn,
            }
        )

    summary = {
        "query_count": len(rows),
        "avg_jaccard": round(statistics.mean(overlaps), 4) if overlaps else 0.0,
        "avg_token_search": round(statistics.mean(token_search), 2) if token_search else 0.0,
        "avg_token_nl_search": round(statistics.mean(token_nl), 2) if token_nl else 0.0,
        "avg_expected_hit_rank_search": round(statistics.mean(rank_search), 2) if rank_search else 0.0,
        "avg_expected_hit_rank_nl": round(statistics.mean(rank_nl), 2) if rank_nl else 0.0,
    }

    payload = {
        "ok": True,
        "project": args.project,
        "index_dir": str(root / args.index_dir),
        "queries_file": str(queries_path),
        "summary": summary,
        "rows": rows,
    }

    out_path = pathlib.Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"ok": True, "out": str(out_path), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
