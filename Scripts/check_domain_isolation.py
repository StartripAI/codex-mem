#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from typing import Iterable, List, Tuple


def _parts(*values: str) -> str:
    return "".join(values)


BANNED_PATTERNS = (
    _parts("h", "openote"),
    r"hope\s*note",
    _parts("copilot", "invoice"),
    _parts("readme", "first", r"\.md"),
    _parts("sota", "_plan", r"\.md"),
)

SKIP_DIR_PARTS = {
    ".git",
    "__pycache__",
    ".codex_mem",
    ".codex_knowledge",
    ".codex_mem_bench",
    ".codex_knowledge_bench",
}


def list_tracked_files(root: pathlib.Path) -> List[pathlib.Path]:
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        files = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            files.append((root / line).resolve())
        return files
    except Exception:
        out: List[pathlib.Path] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIR_PARTS for part in path.parts):
                continue
            out.append(path.resolve())
        return out


def should_skip(path: pathlib.Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIR_PARTS:
            return True
    if path.name == "check_domain_isolation.py":
        return True
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".sqlite3", ".pack", ".idx", ".rev", ".pyc"}:
        return True
    return False


def scan_file(path: pathlib.Path, patterns: Iterable[re.Pattern[str]]) -> List[Tuple[int, str, str]]:
    hits: List[Tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern in patterns:
            if pattern.search(line):
                hits.append((lineno, pattern.pattern, line.strip()))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Check repository for banned cross-domain strings.")
    parser.add_argument("--root", default=".", help="Repository root")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in BANNED_PATTERNS]

    failures: List[Tuple[pathlib.Path, int, str, str]] = []
    for path in list_tracked_files(root):
        if should_skip(path):
            continue
        for lineno, pattern, line in scan_file(path, compiled):
            failures.append((path, lineno, pattern, line))

    if failures:
        print("Domain isolation check failed. Banned patterns found:")
        for path, lineno, pattern, line in failures:
            rel = path.relative_to(root)
            print(f"- {rel}:{lineno} pattern={pattern} line={line}")
        return 1

    print("Domain isolation check passed: no banned patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
