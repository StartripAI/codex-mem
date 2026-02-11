#!/usr/bin/env python3
"""
Generate social copy packs for X, Reddit, and Product Hunt.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List


def extract_highlights(release_notes: str, version: str, limit: int = 5) -> List[str]:
    marker = f"## {version}"
    pos = release_notes.find(marker)
    if pos < 0:
        return []
    block = release_notes[pos:]
    next_pos = block.find("\n## ", len(marker))
    if next_pos > 0:
        block = block[:next_pos]

    bullets = re.findall(r"^-\s+(.+)$", block, flags=re.MULTILINE)
    clean = [b.strip() for b in bullets if b.strip()]
    return clean[:limit]


def load_json_if_exists(path: pathlib.Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def extract_metrics(onboarding_pack: Dict[str, Any], warm_daily: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    try:
        metrics["onboarding_context_reduction_percent"] = float(
            onboarding_pack.get("savings", {}).get("context_saving_percent", 0.0)
        )
        metrics["onboarding_cold_ms"] = float(onboarding_pack.get("ask", {}).get("cold", {}).get("time_ms", 0.0))
        metrics["onboarding_warm_ms"] = float(onboarding_pack.get("ask", {}).get("warm", {}).get("time_ms", 0.0))
    except Exception:
        pass

    try:
        metrics["warm_memory_context_reduction_percent"] = float(warm_daily.get("token", {}).get("saving_percent", 0.0))
        metrics["warm_memory_stage1_median_ms"] = float(warm_daily.get("startup_ms", {}).get("stage1_median", 0.0))
    except Exception:
        pass

    return metrics


def format_primary_hook(version: str, metrics: Dict[str, Any]) -> str:
    cold = metrics.get("onboarding_cold_ms")
    warm = metrics.get("onboarding_warm_ms")
    red = metrics.get("onboarding_context_reduction_percent")
    if isinstance(cold, (int, float)) and isinstance(warm, (int, float)) and isinstance(red, (int, float)) and red > 0:
        return (
            f"codex-mem {version}: ~{int(round(cold))}ms cold project grounding / "
            f"~{int(round(warm))}ms warm, {red:.2f}% smaller onboarding context."
        )
    return f"codex-mem {version} is live: Codex-native persistent memory + progressive retrieval."


def build_x_copy(version: str, highlights: List[str], url: str, metrics: Dict[str, Any]) -> str:
    top = highlights[:3]
    lines = [format_primary_hook(version, metrics)]
    for item in top:
        lines.append(f"- {item}")
    lines.append("Built for Codex: local memory, progressive retrieval, and launch-ready asset workflow.")
    lines.append(url)
    return "\n".join(lines)


def build_reddit_copy(version: str, highlights: List[str], url: str, metrics: Dict[str, Any]) -> str:
    hook = format_primary_hook(version, metrics)
    warm_red = metrics.get("warm_memory_context_reduction_percent")
    warm_ms = metrics.get("warm_memory_stage1_median_ms")
    warm_line = ""
    if isinstance(warm_red, (int, float)) and warm_red > 0:
        warm_line = f"- Warm daily memory (simulated): {warm_red:.2f}% context reduction"
        if isinstance(warm_ms, (int, float)) and warm_ms > 0:
            warm_line += f", ~{int(round(warm_ms))}ms Layer-1 search median"

    lines = [
        f"Title: [Showcase] {hook}",
        "",
        "Primary benchmark hook (reproducible scripts included):",
        "- Cold start onboarding: curated pack vs `ask`",
    ]
    if warm_line:
        lines.append(warm_line)
    lines += [
        "",
        "What it does:",
        "- Local-first memory and progressive retrieval (search -> timeline -> full detail)",
        "- MCP + Skill integration for Codex workflows",
        "- Local web viewer with stable/beta mode and privacy controls",
        "- Launch kit scaffolding for GIF recording and PRD-style screenshot copy",
        "",
        "Highlights in this version:",
    ]
    lines.extend([f"- {h}" for h in highlights[:6]])
    lines.extend(["", f"Repo: {url}"])
    return "\n".join(lines)


def build_ph_copy(version: str, highlights: List[str], url: str, metrics: Dict[str, Any]) -> str:
    tagline = format_primary_hook(version, metrics).replace(f"codex-mem {version}: ", "")
    one_liner = (
        "codex-mem helps Codex keep context across sessions, retrieve only relevant history, "
        "and package your product story with reusable GIF/screenshot assets."
    )
    feature_lines = "\n".join([f"- {h}" for h in highlights[:6]])
    return (
        f"Tagline:\n{tagline}\n\n"
        f"Description:\n{one_liner}\n\n"
        f"What\'s new in {version}:\n{feature_lines}\n\n"
        f"Link:\n{url}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate social copy pack for release")
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", required=True, help="Release version label, e.g. v0.2.0")
    parser.add_argument("--repo-url", default="https://github.com/<YOUR_ORG_OR_USER>/codex-mem")
    parser.add_argument("--release-notes", default="RELEASE_NOTES.md")
    parser.add_argument(
        "--metrics-onboarding-pack",
        default="Documentation/benchmarks/onboarding_pack_codex_mem_rich_20260211.json",
        help="Path to onboarding-pack benchmark JSON used for the primary hook.",
    )
    parser.add_argument(
        "--metrics-warm-daily",
        default="Documentation/benchmarks/marketing_claims_20260211.json",
        help="Path to warm-daily benchmark JSON used for the primary hook.",
    )
    parser.add_argument("--out-dir", default="dist/social")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    notes_path = root / args.release_notes
    if not notes_path.exists():
        raise FileNotFoundError(f"Release notes not found: {notes_path}")

    notes = notes_path.read_text(encoding="utf-8")
    highlights = extract_highlights(notes, args.version)
    if not highlights:
        highlights = ["Release notes section not found; add highlights manually."]

    onboarding_metrics = load_json_if_exists((root / args.metrics_onboarding_pack).resolve())
    warm_metrics = load_json_if_exists((root / args.metrics_warm_daily).resolve())
    metrics = extract_metrics(onboarding_metrics, warm_metrics)

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    x_text = build_x_copy(args.version, highlights, args.repo_url, metrics)
    reddit_text = build_reddit_copy(args.version, highlights, args.repo_url, metrics)
    ph_text = build_ph_copy(args.version, highlights, args.repo_url, metrics)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"social_pack_{args.version}_{ts}".replace("/", "-")

    (out_dir / f"{prefix}_x.txt").write_text(x_text + "\n", encoding="utf-8")
    (out_dir / f"{prefix}_reddit.txt").write_text(reddit_text + "\n", encoding="utf-8")
    (out_dir / f"{prefix}_producthunt.txt").write_text(ph_text + "\n", encoding="utf-8")
    (out_dir / f"{prefix}_meta.json").write_text(
        json.dumps(
            {
                "ok": True,
                "version": args.version,
                "repo_url": args.repo_url,
                "highlights": highlights,
                "generated_at": ts,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "out_dir": str(out_dir),
                "version": args.version,
                "files": sorted([p.name for p in out_dir.glob(f"{prefix}*")]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
