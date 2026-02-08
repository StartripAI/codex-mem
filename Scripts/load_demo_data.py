#!/usr/bin/env python3
"""
Load sanitized demo data into codex-mem for recording and screenshots.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any, Dict, List


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
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")
    text = proc.stdout.strip()
    if not text:
        return {}
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load sanitized demo dataset into codex-mem")
    parser.add_argument("--root", default=".")
    parser.add_argument("--index-dir", default=".codex_mem")
    parser.add_argument(
        "--data",
        default="Assets/LaunchKit/demo-data/session_demo.json",
        help="Path to demo JSON file (relative to repo root by default)",
    )
    parser.add_argument("--reset", action="store_true", help="Delete existing index directory before load")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    data_path = pathlib.Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Demo data file not found: {data_path}")

    index_dir = args.index_dir
    if args.reset:
        idx_path = root / index_dir
        if idx_path.exists():
            import shutil

            shutil.rmtree(idx_path)

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    project = str(payload.get("project", "demo-recording"))
    session_id = str(payload.get("session_id", "demo-session-001"))
    title = str(payload.get("title", "Demo Session"))

    run_cli(root, index_dir, ["init", "--project", project])
    run_cli(root, index_dir, ["session-start", session_id, "--project", project, "--title", title])

    for prompt in payload.get("prompts", []):
        run_cli(root, index_dir, ["user-prompt-submit", session_id, str(prompt), "--project", project])

    for tool in payload.get("tools", []):
        cmd = [
            "post-tool-use",
            session_id,
            str(tool.get("tool_name", "shell")),
            str(tool.get("content", "")),
            "--project",
            project,
            "--title",
            str(tool.get("title", "Tool event")),
        ]
        for tag in tool.get("tags", []) or []:
            cmd.extend(["--tag", str(tag)])
        for ptag in tool.get("privacy_tags", []) or []:
            cmd.extend(["--privacy-tag", str(ptag)])
        run_cli(root, index_dir, cmd)

    run_cli(
        root,
        index_dir,
        [
            "session-end",
            session_id,
            "--project",
            project,
            "--content",
            str(payload.get("end_note", "Demo data loaded")),
        ],
    )

    print(
        json.dumps(
            {
                "ok": True,
                "project": project,
                "session_id": session_id,
                "index_dir": str(root / index_dir),
                "data_file": str(data_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
