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
from typing import List


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


def build_x_copy(version: str, highlights: List[str], url: str) -> str:
    top = highlights[:3]
    lines = [f"codex-mem {version} is live."]
    for item in top:
        lines.append(f"- {item}")
    lines.append("Built for Codex: local memory, progressive retrieval, and launch-ready asset workflow.")
    lines.append(url)
    return "\n".join(lines)


def build_reddit_copy(version: str, highlights: List[str], url: str) -> str:
    lines = [
        f"Title: [Showcase] codex-mem {version}: Codex-native persistent memory + launch asset toolkit",
        "",
        "What it does:",
        "- Local-first memory and progressive retrieval (search -> timeline -> full detail)",
        "- MCP + Skill integration for Codex workflows",
        "- Local web viewer with stable/beta mode and privacy controls",
        "- Launch kit for GIF recording and PRD-style screenshot copy",
        "",
        "Highlights in this version:",
    ]
    lines.extend([f"- {h}" for h in highlights[:6]])
    lines.extend(["", f"Repo: {url}"])
    return "\n".join(lines)


def build_ph_copy(version: str, highlights: List[str], url: str) -> str:
    tagline = "Persistent memory for Codex, with progressive retrieval and launch-ready workflows"
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

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    x_text = build_x_copy(args.version, highlights, args.repo_url)
    reddit_text = build_reddit_copy(args.version, highlights, args.repo_url)
    ph_text = build_ph_copy(args.version, highlights, args.repo_url)

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
