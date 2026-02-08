#!/usr/bin/env python3
"""
Comprehensive simulated smoke test for codex_mem CLI + MCP + Web viewer.

Usage:
  python3 Scripts/codex_mem_smoketest.py --root /path/to/repo
"""

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
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


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def http_get_json(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def http_post_json(url: str, payload: Mapping[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
    raw = json.dumps(dict(payload)).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = resp.read().decode("utf-8")
    return json.loads(out)


def wait_web_ready(base_url: str, timeout_sec: float = 8.0) -> None:
    deadline = time.time() + timeout_sec
    last_err = ""
    while time.time() < deadline:
        try:
            payload = http_get_json(f"{base_url}/api/health", timeout=1.5)
            if payload.get("ok"):
                return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_err = str(exc)
        time.sleep(0.15)
    raise RuntimeError(f"Web viewer not ready: {last_err}")


def run_smoke(root: pathlib.Path) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="codex_mem_smoke_") as tmp:
        index_dir = pathlib.Path(tmp) / ".codex_mem"
        project = "smoketest"

        # CLI flow + runtime config + dual-tag privacy
        run_cli(root, index_dir, ["init", "--project", project])
        cfg_set = run_cli(
            root,
            index_dir,
            ["config-set", "--channel", "beta", "--viewer-refresh-sec", "2", "--beta-endless-mode", "on"],
        )
        cfg = run_cli(root, index_dir, ["config-get"])
        if cfg.get("config", {}).get("channel") != "beta":
            raise RuntimeError("config-set/channel verification failed")

        run_cli(root, index_dir, ["session-start", "s1", "--project", project, "--title", "Smoke Session"])
        run_cli(
            root,
            index_dir,
            ["user-prompt-submit", "s1", "Track bugfixes and stream architecture", "--project", project],
        )
        run_cli(
            root,
            index_dir,
            [
                "post-tool-use",
                "s1",
                "shell",
                "Fixed bug in orchestrator parser and added tests.",
                "--project",
                project,
                "--title",
                "Bugfix patch",
                "--tag",
                "bugfix",
                "--compact",
                "--compact-chars",
                "500",
            ],
        )
        private_payload = run_cli(
            root,
            index_dir,
            [
                "post-tool-use",
                "s1",
                "shell",
                "credential value should be hidden from default retrieval",
                "--project",
                project,
                "--title",
                "Sensitive output",
                "--privacy-tag",
                "private",
                "--privacy-tag",
                "redact",
            ],
        )
        blocked_payload = run_cli(
            root,
            index_dir,
            [
                "post-tool-use",
                "s1",
                "shell",
                "cat ~/.ssh/id_rsa",
                "--project",
                project,
                "--title",
                "Should be blocked",
                "--privacy-tag",
                "no_mem",
            ],
        )
        end_payload = run_cli(root, index_dir, ["session-end", "s1", "--project", project])

        if not blocked_payload.get("skipped"):
            raise RuntimeError("blocked privacy tag did not skip write")
        if private_payload.get("privacy", {}).get("visibility") != "private":
            raise RuntimeError("private privacy tag not applied")

        public_search = run_cli(
            root,
            index_dir,
            ["search", "Sensitive output", "--project", project, "--limit", "20"],
        )
        private_search = run_cli(
            root,
            index_dir,
            ["search", "Sensitive output", "--project", project, "--limit", "20", "--include-private"],
        )
        if public_search.get("results"):
            raise RuntimeError("private content leaked into default search")
        if not private_search.get("results"):
            raise RuntimeError("private search should return private content with include-private")

        nl_search_payload = run_cli(
            root,
            index_dir,
            ["nl-search", "what bugs were fixed", "--project", project, "--limit", "10"],
        )
        if not nl_search_payload.get("results"):
            raise RuntimeError("nl-search returned empty results")

        # Web viewer API flow
        port = reserve_port()
        web_cmd = [
            sys.executable,
            str(root / "Scripts" / "codex_mem_web.py"),
            "--root",
            str(root),
            "--index-dir",
            str(index_dir),
            "--project-default",
            project,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        web_proc = subprocess.Popen(
            web_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            wait_web_ready(base_url)
            api_cfg = http_get_json(f"{base_url}/api/config")
            if api_cfg.get("config", {}).get("channel") != "beta":
                raise RuntimeError("web api config mismatch")
            _ = http_post_json(
                f"{base_url}/api/config",
                {"channel": "stable", "viewer_refresh_sec": 4, "beta_endless_mode": False},
            )
            stream_public = http_get_json(
                f"{base_url}/api/stream?project={urllib.parse.quote(project)}&limit=30&include_private=0"
            )
            stream_private = http_get_json(
                f"{base_url}/api/stream?project={urllib.parse.quote(project)}&limit=30&include_private=1"
            )
            sessions_payload = http_get_json(
                f"{base_url}/api/sessions?project={urllib.parse.quote(project)}&limit=10"
            )
            web_nl = http_get_json(
                f"{base_url}/api/nl-search?q={urllib.parse.quote('what bugs were fixed')}&project={urllib.parse.quote(project)}"
            )
            if len(stream_private.get("items", [])) < len(stream_public.get("items", [])):
                raise RuntimeError("web private stream should include at least public stream items")
            if not sessions_payload.get("items"):
                raise RuntimeError("web sessions endpoint returned empty result")
            if not web_nl.get("results"):
                raise RuntimeError("web nl-search returned empty result")
        finally:
            web_proc.terminate()
            try:
                web_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                web_proc.kill()

        # MCP flow
        mcp_cmd = [
            sys.executable,
            str(root / "Scripts" / "codex_mem_mcp.py"),
            "--root",
            str(root),
            "--index-dir",
            str(index_dir),
            "--project-default",
            project,
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
            required = {
                "mem_search",
                "mem_nl_search",
                "mem_timeline",
                "mem_get_observations",
                "mem_ask",
                "mem_config_get",
                "mem_config_set",
            }
            if not required.issubset(tool_names):
                raise RuntimeError(f"Missing required MCP tools: {required - tool_names}")

            mcp_cfg = mcp_tool(proc, 3, "mem_config_get", {})
            if not mcp_cfg.get("config"):
                raise RuntimeError("MCP mem_config_get returned empty config")
            _ = mcp_tool(proc, 4, "mem_config_set", {"channel": "beta", "beta_endless_mode": True})
            mcp_search = mcp_tool(proc, 5, "mem_nl_search", {"query": "what bugs were fixed", "limit": 5})
            results = mcp_search.get("results", [])
            if not results:
                raise RuntimeError("MCP mem_nl_search returned empty results")
            first_id = str(results[0]["id"])
            mcp_timeline = mcp_tool(proc, 6, "mem_timeline", {"id": first_id, "before": 2, "after": 2})
            _ = mcp_timeline.get("anchor")
            mcp_get = mcp_tool(proc, 7, "mem_get_observations", {"ids": [first_id]})
            if int(mcp_get.get("count", 0)) < 1:
                raise RuntimeError("MCP mem_get_observations returned no items")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

        token_est = nl_search_payload.get("token_estimate_total", 0)
        return {
            "ok": True,
            "config_after_set": cfg_set.get("config", {}),
            "summary_event_count": end_payload.get("summary", {}).get("event_count", 0),
            "nl_search_result_count": len(nl_search_payload.get("results", [])),
            "nl_search_token_estimate_total": token_est,
            "private_search_result_count": len(private_search.get("results", [])),
            "public_search_result_count": len(public_search.get("results", [])),
            "web_stream_public_count": len(stream_public.get("items", [])),
            "web_stream_private_count": len(stream_private.get("items", [])),
            "web_sessions_count": len(sessions_payload.get("items", [])),
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
