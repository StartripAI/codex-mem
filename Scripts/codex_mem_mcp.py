#!/usr/bin/env python3
"""
MCP server for codex_mem.

Provides Codex-accessible tools:
- mem_search
- mem_nl_search
- mem_timeline
- mem_get_observations
- mem_ask
- mem_config_get
- mem_config_set
- mem_session_start
- mem_user_prompt_submit
- mem_post_tool_use
- mem_stop
- mem_session_end
- mem_summarize_session
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Mapping, Sequence


SERVER_NAME = "codex-mem-mcp"
SERVER_VERSION = "0.2.0"
PROTOCOL_VERSION = "2024-11-05"


class MCPError(Exception):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def read_message(stream) -> Mapping[str, Any] | None:
    """
    Read one MCP JSON-RPC message framed with Content-Length.
    """
    headers: Dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        raw = line.decode("utf-8", errors="replace").strip()
        if ":" not in raw:
            # Fallback for newline-delimited JSON in local manual tests.
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                continue
        key, value = raw.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    length = headers.get("content-length")
    if not length:
        raise MCPError(-32700, "Missing Content-Length header")
    try:
        size = int(length)
    except ValueError as exc:
        raise MCPError(-32700, f"Invalid Content-Length: {length}") from exc
    body = stream.read(size)
    if len(body) != size:
        raise MCPError(-32700, "Incomplete message body")
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise MCPError(-32700, f"Invalid JSON body: {exc}") from exc


def write_message(stream, payload: Mapping[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    stream.write(header)
    stream.write(body)
    stream.flush()


def make_tool(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    required: Sequence[str],
) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": dict(properties),
            "required": list(required),
            "additionalProperties": False,
        },
    }


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    make_tool(
        "mem_search",
        "Stage 1 progressive retrieval: compact index entries by lexical+semantic score.",
        {
            "query": {"type": "string"},
            "project": {"type": "string"},
            "session_id": {"type": "string"},
            "since": {"type": "string"},
            "until": {"type": "string"},
            "include_private": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            "alpha": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        ["query"],
    ),
    make_tool(
        "mem_nl_search",
        "Natural-language project history search (time phrases + intent hints).",
        {
            "query": {"type": "string"},
            "project": {"type": "string"},
            "session_id": {"type": "string"},
            "since": {"type": "string"},
            "until": {"type": "string"},
            "include_private": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            "alpha": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "snippet_chars": {"type": "integer", "minimum": 32, "maximum": 8000},
        },
        ["query"],
    ),
    make_tool(
        "mem_timeline",
        "Stage 2 retrieval: timeline neighborhood for E<ID> or O<ID>.",
        {
            "id": {"type": "string"},
            "before": {"type": "integer", "minimum": 0, "maximum": 200},
            "after": {"type": "integer", "minimum": 0, "maximum": 200},
            "include_private": {"type": "boolean"},
            "snippet_chars": {"type": "integer", "minimum": 32, "maximum": 4000},
        },
        ["id"],
    ),
    make_tool(
        "mem_get_observations",
        "Stage 3 retrieval: fetch full details for selected IDs.",
        {
            "ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "compact": {"type": "boolean"},
            "include_private": {"type": "boolean"},
            "snippet_chars": {"type": "integer", "minimum": 32, "maximum": 4000},
        },
        ["ids"],
    ),
    make_tool(
        "mem_ask",
        "Fused retrieval: memory progressive retrieval + repo_knowledge query.",
        {
            "question": {"type": "string"},
            "project": {"type": "string"},
            "session_id": {"type": "string"},
            "search_limit": {"type": "integer", "minimum": 1, "maximum": 200},
            "detail_limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "code_top_k": {"type": "integer", "minimum": 1, "maximum": 100},
            "code_module_limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "repo_index_dir": {"type": "string"},
            "alpha": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "include_private": {"type": "boolean"},
            "snippet_chars": {"type": "integer", "minimum": 32, "maximum": 8000},
            "prompt_only": {"type": "boolean"},
        },
        ["question"],
    ),
    make_tool(
        "mem_config_get",
        "Read runtime configuration (channel/view refresh/endless mode).",
        {},
        [],
    ),
    make_tool(
        "mem_config_set",
        "Update runtime configuration (stable/beta, refresh interval, beta endless mode).",
        {
            "channel": {"type": "string", "enum": ["stable", "beta"]},
            "viewer_refresh_sec": {"type": "integer", "minimum": 1, "maximum": 60},
            "beta_endless_mode": {"type": "boolean"},
        },
        [],
    ),
    make_tool(
        "mem_session_start",
        "Lifecycle hook: SessionStart.",
        {
            "session_id": {"type": "string"},
            "project": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
        },
        ["session_id"],
    ),
    make_tool(
        "mem_user_prompt_submit",
        "Lifecycle hook: UserPromptSubmit.",
        {
            "session_id": {"type": "string"},
            "prompt": {"type": "string"},
            "project": {"type": "string"},
            "title": {"type": "string"},
        },
        ["session_id", "prompt"],
    ),
    make_tool(
        "mem_post_tool_use",
        "Lifecycle hook: PostToolUse.",
        {
            "session_id": {"type": "string"},
            "tool_name": {"type": "string"},
            "content": {"type": "string"},
            "project": {"type": "string"},
            "title": {"type": "string"},
            "file_path": {"type": "string"},
            "exit_code": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "privacy_tags": {"type": "array", "items": {"type": "string"}},
            "compact": {"type": "boolean"},
            "compact_chars": {"type": "integer", "minimum": 128, "maximum": 20000},
        },
        ["session_id", "tool_name", "content"],
    ),
    make_tool(
        "mem_stop",
        "Lifecycle hook: Stop.",
        {
            "session_id": {"type": "string"},
            "project": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
        },
        ["session_id"],
    ),
    make_tool(
        "mem_session_end",
        "Lifecycle hook: SessionEnd (optionally generate structured summary).",
        {
            "session_id": {"type": "string"},
            "project": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "skip_summary": {"type": "boolean"},
        },
        ["session_id"],
    ),
    make_tool(
        "mem_summarize_session",
        "Regenerate structured session summary observations.",
        {
            "session_id": {"type": "string"},
        },
        ["session_id"],
    ),
]


class CodexMemMCPServer:
    def __init__(
        self,
        *,
        root: pathlib.Path,
        index_dir: str,
        project_default: str,
        python_bin: str,
    ) -> None:
        self.root = root
        self.index_dir = index_dir
        self.project_default = project_default
        self.python_bin = python_bin
        self.codex_mem_script = self.root / "Scripts" / "codex_mem.py"
        if not self.codex_mem_script.exists():
            raise RuntimeError(f"Missing script: {self.codex_mem_script}")

    def run_cli(self, args: Sequence[str], expect_json: bool = True) -> Any:
        cmd = [
            self.python_bin,
            str(self.codex_mem_script),
            "--root",
            str(self.root),
            "--index-dir",
            self.index_dir,
            *list(args),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(self.root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise MCPError(
                -32000,
                "codex_mem command failed",
                {"cmd": cmd, "stderr": proc.stderr.strip(), "stdout": proc.stdout.strip()},
            )
        out = proc.stdout.strip()
        if not expect_json:
            return out
        if not out:
            return {}
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {"raw": out}

    @staticmethod
    def text_content(payload: Any) -> Dict[str, Any]:
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
        return {"content": [{"type": "text", "text": text}]}

    @staticmethod
    def get_arg(arguments: Mapping[str, Any], key: str, default: Any = None) -> Any:
        val = arguments.get(key, default)
        return val

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        project = str(self.get_arg(arguments, "project", self.project_default))
        if name == "mem_search":
            query = str(self.get_arg(arguments, "query", "")).strip()
            if not query:
                raise MCPError(-32602, "`query` is required")
            cmd = ["search", query, "--project", project]
            if arguments.get("session_id"):
                cmd.extend(["--session-id", str(arguments["session_id"])])
            if arguments.get("since"):
                cmd.extend(["--since", str(arguments["since"])])
            if arguments.get("until"):
                cmd.extend(["--until", str(arguments["until"])])
            if bool(arguments.get("include_private")):
                cmd.append("--include-private")
            if arguments.get("limit") is not None:
                cmd.extend(["--limit", str(int(arguments["limit"]))])
            if arguments.get("alpha") is not None:
                cmd.extend(["--alpha", str(float(arguments["alpha"]))])
            return self.text_content(self.run_cli(cmd))

        if name == "mem_nl_search":
            query = str(self.get_arg(arguments, "query", "")).strip()
            if not query:
                raise MCPError(-32602, "`query` is required")
            cmd = ["nl-search", query, "--project", project]
            if arguments.get("session_id"):
                cmd.extend(["--session-id", str(arguments["session_id"])])
            if arguments.get("since"):
                cmd.extend(["--since", str(arguments["since"])])
            if arguments.get("until"):
                cmd.extend(["--until", str(arguments["until"])])
            if bool(arguments.get("include_private")):
                cmd.append("--include-private")
            if arguments.get("limit") is not None:
                cmd.extend(["--limit", str(int(arguments["limit"]))])
            if arguments.get("alpha") is not None:
                cmd.extend(["--alpha", str(float(arguments["alpha"]))])
            if arguments.get("snippet_chars") is not None:
                cmd.extend(["--snippet-chars", str(int(arguments["snippet_chars"]))])
            return self.text_content(self.run_cli(cmd))

        if name == "mem_timeline":
            ident = str(self.get_arg(arguments, "id", "")).strip()
            if not ident:
                raise MCPError(-32602, "`id` is required")
            cmd = ["timeline", ident]
            if arguments.get("before") is not None:
                cmd.extend(["--before", str(int(arguments["before"]))])
            if arguments.get("after") is not None:
                cmd.extend(["--after", str(int(arguments["after"]))])
            if bool(arguments.get("include_private")):
                cmd.append("--include-private")
            if arguments.get("snippet_chars") is not None:
                cmd.extend(["--snippet-chars", str(int(arguments["snippet_chars"]))])
            return self.text_content(self.run_cli(cmd))

        if name == "mem_get_observations":
            ids = arguments.get("ids")
            if not isinstance(ids, list) or not ids:
                raise MCPError(-32602, "`ids` must be a non-empty array")
            cmd = ["get-observations", *[str(v) for v in ids]]
            if bool(arguments.get("compact")):
                cmd.append("--compact")
            if bool(arguments.get("include_private")):
                cmd.append("--include-private")
            if arguments.get("snippet_chars") is not None:
                cmd.extend(["--snippet-chars", str(int(arguments["snippet_chars"]))])
            return self.text_content(self.run_cli(cmd))

        if name == "mem_ask":
            question = str(self.get_arg(arguments, "question", "")).strip()
            if not question:
                raise MCPError(-32602, "`question` is required")
            cmd = ["ask", question, "--project", project]
            if arguments.get("session_id"):
                cmd.extend(["--session-id", str(arguments["session_id"])])
            if arguments.get("search_limit") is not None:
                cmd.extend(["--search-limit", str(int(arguments["search_limit"]))])
            if arguments.get("detail_limit") is not None:
                cmd.extend(["--detail-limit", str(int(arguments["detail_limit"]))])
            if arguments.get("code_top_k") is not None:
                cmd.extend(["--code-top-k", str(int(arguments["code_top_k"]))])
            if arguments.get("code_module_limit") is not None:
                cmd.extend(["--code-module-limit", str(int(arguments["code_module_limit"]))])
            if arguments.get("repo_index_dir"):
                cmd.extend(["--repo-index-dir", str(arguments["repo_index_dir"])])
            if arguments.get("alpha") is not None:
                cmd.extend(["--alpha", str(float(arguments["alpha"]))])
            if bool(arguments.get("include_private")):
                cmd.append("--include-private")
            if arguments.get("snippet_chars") is not None:
                cmd.extend(["--snippet-chars", str(int(arguments["snippet_chars"]))])
            if bool(arguments.get("prompt_only")):
                cmd.append("--prompt-only")
                return self.text_content(self.run_cli(cmd, expect_json=False))
            return self.text_content(self.run_cli(cmd))

        if name == "mem_config_get":
            return self.text_content(self.run_cli(["config-get"]))

        if name == "mem_config_set":
            cmd = ["config-set"]
            if arguments.get("channel"):
                cmd.extend(["--channel", str(arguments["channel"])])
            if arguments.get("viewer_refresh_sec") is not None:
                cmd.extend(["--viewer-refresh-sec", str(int(arguments["viewer_refresh_sec"]))])
            if arguments.get("beta_endless_mode") is not None:
                enabled = bool(arguments["beta_endless_mode"])
                cmd.extend(["--beta-endless-mode", "on" if enabled else "off"])
            return self.text_content(self.run_cli(cmd))

        if name == "mem_session_start":
            sid = str(self.get_arg(arguments, "session_id", "")).strip()
            if not sid:
                raise MCPError(-32602, "`session_id` is required")
            cmd = ["session-start", sid, "--project", project]
            if arguments.get("title"):
                cmd.extend(["--title", str(arguments["title"])])
            if arguments.get("content"):
                cmd.extend(["--content", str(arguments["content"])])
            return self.text_content(self.run_cli(cmd))

        if name == "mem_user_prompt_submit":
            sid = str(self.get_arg(arguments, "session_id", "")).strip()
            prompt = str(self.get_arg(arguments, "prompt", "")).strip()
            if not sid or not prompt:
                raise MCPError(-32602, "`session_id` and `prompt` are required")
            cmd = ["user-prompt-submit", sid, prompt, "--project", project]
            if arguments.get("title"):
                cmd.extend(["--title", str(arguments["title"])])
            return self.text_content(self.run_cli(cmd))

        if name == "mem_post_tool_use":
            sid = str(self.get_arg(arguments, "session_id", "")).strip()
            tool_name = str(self.get_arg(arguments, "tool_name", "")).strip()
            content = str(self.get_arg(arguments, "content", ""))
            if not sid or not tool_name:
                raise MCPError(-32602, "`session_id`, `tool_name`, `content` are required")
            cmd = ["post-tool-use", sid, tool_name, content, "--project", project]
            if arguments.get("title"):
                cmd.extend(["--title", str(arguments["title"])])
            if arguments.get("file_path"):
                cmd.extend(["--file-path", str(arguments["file_path"])])
            if arguments.get("exit_code") is not None:
                cmd.extend(["--exit-code", str(int(arguments["exit_code"]))])
            tags = arguments.get("tags")
            if isinstance(tags, list):
                for tag in tags:
                    cmd.extend(["--tag", str(tag)])
            privacy_tags = arguments.get("privacy_tags")
            if isinstance(privacy_tags, list):
                for tag in privacy_tags:
                    cmd.extend(["--privacy-tag", str(tag)])
            if bool(arguments.get("compact")):
                cmd.append("--compact")
            if arguments.get("compact_chars") is not None:
                cmd.extend(["--compact-chars", str(int(arguments["compact_chars"]))])
            return self.text_content(self.run_cli(cmd))

        if name == "mem_stop":
            sid = str(self.get_arg(arguments, "session_id", "")).strip()
            if not sid:
                raise MCPError(-32602, "`session_id` is required")
            cmd = ["stop", sid, "--project", project]
            if arguments.get("title"):
                cmd.extend(["--title", str(arguments["title"])])
            if arguments.get("content"):
                cmd.extend(["--content", str(arguments["content"])])
            return self.text_content(self.run_cli(cmd))

        if name == "mem_session_end":
            sid = str(self.get_arg(arguments, "session_id", "")).strip()
            if not sid:
                raise MCPError(-32602, "`session_id` is required")
            cmd = ["session-end", sid, "--project", project]
            if arguments.get("title"):
                cmd.extend(["--title", str(arguments["title"])])
            if arguments.get("content"):
                cmd.extend(["--content", str(arguments["content"])])
            if bool(arguments.get("skip_summary")):
                cmd.append("--skip-summary")
            return self.text_content(self.run_cli(cmd))

        if name == "mem_summarize_session":
            sid = str(self.get_arg(arguments, "session_id", "")).strip()
            if not sid:
                raise MCPError(-32602, "`session_id` is required")
            cmd = ["summarize-session", sid]
            return self.text_content(self.run_cli(cmd))

        raise MCPError(-32601, f"Unknown tool: {name}")

    def handle_request(self, request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params", {})

        # Notifications do not require responses.
        is_notification = req_id is None

        if method == "notifications/initialized":
            return None

        if method == "initialize":
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
            if is_notification:
                return None
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        if method == "ping":
            if is_notification:
                return None
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            result = {"tools": TOOL_SCHEMAS}
            if is_notification:
                return None
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        if method == "tools/call":
            if not isinstance(params, Mapping):
                raise MCPError(-32602, "Invalid params for tools/call")
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str) or not name:
                raise MCPError(-32602, "`name` is required")
            if not isinstance(arguments, Mapping):
                raise MCPError(-32602, "`arguments` must be an object")
            result = self.call_tool(name, arguments)
            if is_notification:
                return None
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        raise MCPError(-32601, f"Method not found: {method}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCP server for codex_mem.py")
    parser.add_argument("--root", default=".", help="Repository root containing Scripts/codex_mem.py")
    parser.add_argument("--index-dir", default=".codex_mem")
    parser.add_argument("--project-default", default="default")
    parser.add_argument("--python-bin", default=sys.executable or "python3")
    args = parser.parse_args(argv)

    server = CodexMemMCPServer(
        root=pathlib.Path(args.root).resolve(),
        index_dir=args.index_dir,
        project_default=args.project_default,
        python_bin=args.python_bin,
    )

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        try:
            req = read_message(stdin)
            if req is None:
                break
            if not isinstance(req, Mapping):
                continue
            try:
                resp = server.handle_request(req)
                if resp is not None:
                    write_message(stdout, resp)
            except MCPError as err:
                if req.get("id") is None:
                    continue
                payload: Dict[str, Any] = {
                    "jsonrpc": "2.0",
                    "id": req.get("id"),
                    "error": {"code": err.code, "message": err.message},
                }
                if err.data is not None:
                    payload["error"]["data"] = err.data
                write_message(stdout, payload)
            except Exception as err:  # noqa: BLE001
                if req.get("id") is None:
                    continue
                payload = {
                    "jsonrpc": "2.0",
                    "id": req.get("id"),
                    "error": {
                        "code": -32603,
                        "message": "Internal error",
                        "data": {
                            "error": str(err),
                            "traceback": traceback.format_exc(limit=8),
                        },
                    },
                }
                write_message(stdout, payload)
        except MCPError as err:
            # Protocol-level parse issue with no request id context.
            write_message(
                stdout,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": err.code,
                        "message": err.message,
                        "data": err.data,
                    },
                },
            )
        except Exception:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
