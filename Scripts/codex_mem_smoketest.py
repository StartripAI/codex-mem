#!/usr/bin/env python3
"""
Simulated end-to-end smoke test for codex_mem CLI + MCP server.

Usage:
  python3 Scripts/codex_mem_smoketest.py --root /path/to/repo
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Dict, Mapping


def run_cli(root: pathlib.Path, index_dir: pathlib.Path, args: list[str]) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(root / "Scripts" / "codex_mem.py"),
        "--root",
        str(root),
        "--index-dir",
        str(index_dir),
        *args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI failed: {cmd}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")
    out = proc.stdout.strip()
    if not out:
        return {}
    return json.loads(out)


def write_mcp(stdin, payload: Mapping[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    stdin.write(body)
    stdin.flush()


def read_mcp(stdout) -> Dict[str, Any]:
    headers: Dict[str, str] = {}
    while True:
        line = stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed stream unexpectedly")
        if line in (b"\r\n", b"\n"):
            break
        text = line.decode("utf-8").strip()
        key, value = text.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        raise RuntimeError(f"Invalid MCP response length: {headers}")
    raw = stdout.read(length)
    return json.loads(raw.decode("utf-8"))


def mcp_call(proc: subprocess.Popen[bytes], req_id: int, method: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    assert proc.stdin is not None
    assert proc.stdout is not None
    payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": dict(params)}
    write_mcp(proc.stdin, payload)
    resp = read_mcp(proc.stdout)
    if "error" in resp:
        raise RuntimeError(f"MCP {method} failed: {json.dumps(resp, ensure_ascii=False)}")
    return resp


def mcp_tool(proc: subprocess.Popen[bytes], req_id: int, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    resp = mcp_call(proc, req_id, "tools/call", {"name": name, "arguments": dict(arguments)})
    result = resp.get("result", {})
    content = result.get("content", [])
    if not content:
        return {}
    text = content[0].get("text", "")
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def run_smoke(root: pathlib.Path) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="codex_mem_smoke_") as tmp:
        index_dir = pathlib.Path(tmp) / ".codex_mem"

        # CLI flow
        run_cli(root, index_dir, ["init", "--project", "smoketest"])
        run_cli(root, index_dir, ["session-start", "s1", "--project", "smoketest", "--title", "Smoke Session"])
        run_cli(root, index_dir, ["user-prompt-submit", "s1", "分析 Home streaming 主流程", "--project", "smoketest"])
        run_cli(
            root,
            index_dir,
            [
                "post-tool-use",
                "s1",
                "shell",
                "grep -n HomeStreamOrchestrator HomeViewModel.swift",
                "--project",
                "smoketest",
                "--title",
                "定位 orchestrator",
                "--compact",
                "--compact-chars",
                "600",
            ],
        )
        end_payload = run_cli(root, index_dir, ["session-end", "s1", "--project", "smoketest"])
        search_payload = run_cli(
            root,
            index_dir,
            ["search", "orchestrator streaming", "--project", "smoketest", "--limit", "5"],
        )
        if not search_payload.get("results"):
            raise RuntimeError("CLI search returned empty results")

        # MCP flow
        mcp_cmd = [
            sys.executable,
            str(root / "Scripts" / "codex_mem_mcp.py"),
            "--root",
            str(root),
            "--index-dir",
            str(index_dir),
            "--project-default",
            "smoketest",
        ]
        proc = subprocess.Popen(
            mcp_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            init_resp = mcp_call(proc, 1, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
            _ = init_resp.get("result", {})
            tools_resp = mcp_call(proc, 2, "tools/list", {})
            tools = tools_resp.get("result", {}).get("tools", [])
            tool_names = {tool.get("name") for tool in tools}
            required = {"mem_search", "mem_timeline", "mem_get_observations", "mem_ask"}
            if not required.issubset(tool_names):
                raise RuntimeError(f"Missing required MCP tools: {required - tool_names}")

            mcp_search = mcp_tool(proc, 3, "mem_search", {"query": "orchestrator", "limit": 5})
            results = mcp_search.get("results", [])
            if not results:
                raise RuntimeError("MCP mem_search returned empty results")
            first_id = str(results[0]["id"])
            mcp_timeline = mcp_tool(proc, 4, "mem_timeline", {"id": first_id, "before": 2, "after": 2})
            _ = mcp_timeline.get("anchor")

            mcp_get = mcp_tool(proc, 5, "mem_get_observations", {"ids": [first_id]})
            if int(mcp_get.get("count", 0)) < 1:
                raise RuntimeError("MCP mem_get_observations returned no items")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

        token_est = search_payload.get("token_estimate_total", 0)
        return {
            "ok": True,
            "summary_event_count": end_payload.get("summary", {}).get("event_count", 0),
            "search_result_count": len(search_payload.get("results", [])),
            "search_token_estimate_total": token_est,
            "mcp_tools_verified": sorted(required),
            "first_result_id": first_id,
            "index_dir": str(index_dir),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root path")
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    payload = run_smoke(root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
