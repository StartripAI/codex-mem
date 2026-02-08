#!/usr/bin/env python3
"""
Validate launch marketing assets and README media references.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List

try:
    from PIL import Image
except Exception:  # noqa: BLE001
    Image = None  # type: ignore[assignment]

GIF_NAME_RE = re.compile(r"^gif_\d{2}_[a-z0-9\-]+_v\d+\.gif$")
README_GIF_RE = re.compile(r"!\[[^\]]*\]\(([^)]+\.gif(?:\?[^)]*)?)\)", re.IGNORECASE)


@dataclass
class Finding:
    level: str
    message: str


def gif_duration_seconds(path: pathlib.Path) -> float:
    if Image is None:
        return -1.0
    with Image.open(path) as img:
        total_ms = 0
        frames = getattr(img, "n_frames", 1)
        for idx in range(frames):
            img.seek(idx)
            total_ms += int(img.info.get("duration", 0))
        return max(0.0, total_ms / 1000.0)


def validate_assets(root: pathlib.Path, max_gif_mb: float, min_duration: float, max_duration: float) -> List[Finding]:
    findings: List[Finding] = []
    gif_dir = root / "Assets" / "LaunchKit" / "gif" / "export"
    poster_dir = root / "Assets" / "LaunchKit" / "gif" / "posters"
    screenshot_dir = root / "Assets" / "LaunchKit" / "screenshots" / "final"

    if not gif_dir.exists():
        findings.append(Finding("error", f"GIF export dir missing: {gif_dir}"))
        return findings

    gifs = sorted(p for p in gif_dir.glob("*.gif") if p.is_file())
    if not gifs:
        findings.append(Finding("error", "No GIF files found in Assets/LaunchKit/gif/export"))

    for gif in gifs:
        if not GIF_NAME_RE.match(gif.name):
            findings.append(Finding("error", f"Invalid GIF name format: {gif.name}"))
        size_mb = gif.stat().st_size / (1024 * 1024)
        if size_mb > max_gif_mb:
            findings.append(Finding("error", f"GIF too large ({size_mb:.2f}MB): {gif.name}"))

        duration = gif_duration_seconds(gif)
        if duration < 0:
            findings.append(Finding("warn", f"Duration skipped (Pillow unavailable): {gif.name}"))
        elif duration < min_duration or duration > max_duration:
            findings.append(
                Finding(
                    "error",
                    f"GIF duration out of range ({duration:.2f}s not in {min_duration}-{max_duration}s): {gif.name}",
                )
            )

        poster = poster_dir / f"{gif.stem}.png"
        if not poster.exists():
            findings.append(Finding("error", f"Missing poster for GIF: {poster}"))

    screenshots = sorted(p for p in screenshot_dir.glob("*.png") if p.is_file())
    if len(screenshots) < 3:
        findings.append(Finding("error", "Need at least 3 final screenshots in Assets/LaunchKit/screenshots/final"))

    return findings


def validate_readme_links(root: pathlib.Path) -> List[Finding]:
    findings: List[Finding] = []
    readme = root / "README.md"
    if not readme.exists():
        return [Finding("error", "README.md not found")]

    content = readme.read_text(encoding="utf-8")
    gif_links = README_GIF_RE.findall(content)
    if not gif_links:
        findings.append(Finding("error", "README has no GIF links"))
        return findings

    for link in gif_links:
        path_text = link.split("?", 1)[0].strip()
        if path_text.startswith("http://") or path_text.startswith("https://"):
            findings.append(Finding("error", f"README GIF link must be local file, found remote URL: {path_text}"))
            continue
        target = (root / path_text).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            findings.append(Finding("error", f"README GIF path escapes repo root: {path_text}"))
            continue
        if not target.exists():
            findings.append(Finding("error", f"README GIF file missing: {path_text}"))

    if "## Comparison Table" not in content:
        findings.append(Finding("error", "README is missing '## Comparison Table' section"))

    if "[Release Notes](RELEASE_NOTES.md)" not in content:
        findings.append(Finding("error", "README is missing release notes entry link"))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate codex-mem launch assets and README GIF references")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--max-gif-mb", type=float, default=8.0)
    parser.add_argument("--min-duration", type=float, default=0.2)
    parser.add_argument("--max-duration", type=float, default=25.0)
    parser.add_argument("--check-readme", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    findings: List[Finding] = []
    findings.extend(validate_assets(root, args.max_gif_mb, args.min_duration, args.max_duration))
    if args.check_readme:
        findings.extend(validate_readme_links(root))

    has_error = False
    has_warn = False
    for item in findings:
        level = item.level.upper()
        print(f"[{level}] {item.message}")
        if item.level == "error":
            has_error = True
        if item.level == "warn":
            has_warn = True

    if not findings:
        print("[OK] Asset validation passed with no findings")

    if has_error:
        return 1
    if has_warn and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
