#!/usr/bin/env python3
"""
Local web viewer for codex-mem.

Provides:
- Real-time memory stream
- Session summaries
- Natural-language search endpoint
- Stable/Beta config switching
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from codex_mem import (  # noqa: E402
    DEFAULT_SNIPPET_CHARS,
    DEFAULT_VECTOR_DIM,
    blended_search,
    fetch_meta,
    filter_results_by_intent,
    get_runtime_config,
    open_db,
    parse_natural_query,
    set_runtime_config,
)


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def html_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>codex-mem viewer</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --primary: #0a7c66;
      --line: #dbe3ef;
    }
    body {
      margin: 0;
      padding: 18px;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(1200px 500px at 20% -20%, #d7f7ee, transparent),
                  radial-gradient(900px 450px at 90% -30%, #e4eafe, transparent),
                  var(--bg);
    }
    .grid {
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    }
    h1, h2 {
      margin: 0 0 10px;
      font-weight: 700;
      letter-spacing: 0.1px;
    }
    .muted { color: var(--muted); font-size: 13px; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
    input, select, button {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 14px;
      background: #fff;
    }
    button {
      cursor: pointer;
      background: var(--primary);
      border-color: var(--primary);
      color: #fff;
      font-weight: 600;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 8px 6px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    code {
      background: #f1f5f9;
      padding: 2px 4px;
      border-radius: 6px;
      font-size: 12px;
    }
    .pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: #f8fafc;
    }
    .private { color: #9a3412; border-color: #f5c9ab; background: #fff7ed; }
    .public { color: #14532d; border-color: #bbf7d0; background: #f0fdf4; }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="panel">
    <h1>codex-mem local viewer</h1>
    <div class="muted">Real-time memory stream, session summaries, natural-language mem-search, and stable/beta runtime switch.</div>
    <div class="row">
      <label>Project <input id="project" value="default" /></label>
      <label>Channel
        <select id="channel">
          <option value="stable">stable</option>
          <option value="beta">beta</option>
        </select>
      </label>
      <label><input type="checkbox" id="endlessMode" /> beta endless mode</label>
      <label>Refresh (sec) <input id="refreshSec" type="number" min="1" max="60" value="3" style="width:72px" /></label>
      <button id="saveConfig">Save Config</button>
      <button id="reload">Reload</button>
      <span id="status" class="muted"></span>
    </div>
    <div class="row">
      <input id="nlQuery" style="min-width:420px;flex:1" placeholder="e.g. What bugs were fixed last week?" />
      <button id="runQuery">Run mem-search</button>
      <label><input type="checkbox" id="includePrivate" /> include private</label>
    </div>
  </div>

  <div class="grid" style="margin-top:14px;">
    <div class="panel">
      <h2>Memory Stream</h2>
      <table id="streamTable">
        <thead>
          <tr><th>Time</th><th>Session</th><th>Kind</th><th>Title</th><th>Visibility</th></tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="panel">
      <h2>Session Summaries</h2>
      <div id="sessions"></div>
    </div>
  </div>

  <div class="panel" style="margin-top:14px;">
    <h2>mem-search Results</h2>
    <div id="queryInfo" class="muted"></div>
    <table id="searchTable">
      <thead>
        <tr><th>ID</th><th>Type</th><th>Title</th><th>Score</th><th>Session</th><th>Time</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

  <script>
    const qs = (id) => document.getElementById(id);
    let refreshTimer = null;

    async function getJSON(url, options) {
      const res = await fetch(url, options);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return res.json();
    }

    function renderStream(items) {
      const tbody = qs('streamTable').querySelector('tbody');
      tbody.innerHTML = '';
      for (const item of items) {
        const tr = document.createElement('tr');
        const vis = item.visibility || 'public';
        tr.innerHTML = `<td>${item.created_at}</td>
                        <td><code>${item.session_id}</code></td>
                        <td>${item.event_kind}</td>
                        <td>${item.title}</td>
                        <td><span class="pill ${vis}">${vis}</span></td>`;
        tbody.appendChild(tr);
      }
    }

    function renderSessions(items) {
      const root = qs('sessions');
      root.innerHTML = '';
      for (const s of items) {
        const block = document.createElement('div');
        block.style.border = '1px solid var(--line)';
        block.style.borderRadius = '10px';
        block.style.padding = '10px';
        block.style.marginBottom = '8px';
        const summary = s.summary || {};
        const completed = (summary.completed_work || []).slice(0, 3).map(x => `<li>${x}</li>`).join('');
        block.innerHTML = `
          <div><strong>${s.title}</strong> <span class="muted">(${s.session_id})</span></div>
          <div class="muted">${s.started_at} → ${s.ended_at || 'active'}</div>
          <div style="margin-top:6px">Completed:</div>
          <ul>${completed || '<li class="muted">No summary yet</li>'}</ul>
        `;
        root.appendChild(block);
      }
    }

    function renderSearch(payload) {
      qs('queryInfo').textContent = `Interpreted query: ${JSON.stringify(payload.interpreted || {})}`;
      const tbody = qs('searchTable').querySelector('tbody');
      tbody.innerHTML = '';
      for (const r of payload.results || []) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td><code>${r.id}</code></td>
                        <td>${r.item_type}</td>
                        <td>${r.title}</td>
                        <td>${r.score}</td>
                        <td>${r.session_id}</td>
                        <td>${r.created_at}</td>`;
        tbody.appendChild(tr);
      }
    }

    async function loadConfig() {
      const payload = await getJSON('/api/config');
      const cfg = payload.config || {};
      qs('channel').value = cfg.channel || 'stable';
      qs('refreshSec').value = cfg.viewer_refresh_sec || 3;
      qs('endlessMode').checked = !!cfg.beta_endless_mode;
      return cfg;
    }

    async function saveConfig() {
      const body = {
        channel: qs('channel').value,
        viewer_refresh_sec: Number(qs('refreshSec').value || 3),
        beta_endless_mode: !!qs('endlessMode').checked
      };
      await getJSON('/api/config', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(body)
      });
      qs('status').textContent = 'Config saved';
      await reloadAll();
    }

    async function reloadAll() {
      const project = encodeURIComponent(qs('project').value || 'default');
      const includePrivate = qs('includePrivate').checked ? '1' : '0';
      const [stream, sessions, cfg] = await Promise.all([
        getJSON(`/api/stream?project=${project}&limit=50&include_private=${includePrivate}`),
        getJSON(`/api/sessions?project=${project}&limit=20`),
        getJSON('/api/config')
      ]);
      renderStream(stream.items || []);
      renderSessions(sessions.items || []);
      const sec = (cfg.config || {}).viewer_refresh_sec || 3;
      if (refreshTimer) clearInterval(refreshTimer);
      refreshTimer = setInterval(() => reloadAll().catch(console.error), Number(sec) * 1000);
      qs('status').textContent = `Updated at ${new Date().toLocaleTimeString()}`;
    }

    async function runNLSearch() {
      const q = encodeURIComponent(qs('nlQuery').value || '');
      if (!q) return;
      const project = encodeURIComponent(qs('project').value || 'default');
      const includePrivate = qs('includePrivate').checked ? '1' : '0';
      const payload = await getJSON(`/api/nl-search?q=${q}&project=${project}&limit=20&include_private=${includePrivate}`);
      renderSearch(payload);
    }

    qs('saveConfig').addEventListener('click', () => saveConfig().catch(err => alert(err.message)));
    qs('reload').addEventListener('click', () => reloadAll().catch(err => alert(err.message)));
    qs('runQuery').addEventListener('click', () => runNLSearch().catch(err => alert(err.message)));

    loadConfig()
      .then(() => reloadAll())
      .catch(err => { qs('status').textContent = err.message; });
  </script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    server: "ViewerServer"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    def _json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/":
                self._html(html_page())
                return
            if path == "/api/health":
                self._json({"ok": True, "service": "codex-mem-viewer"})
                return
            if path == "/api/config":
                conn = self.server.open_conn()
                try:
                    cfg = get_runtime_config(conn)
                finally:
                    conn.close()
                self._json({"ok": True, "config": cfg})
                return
            if path == "/api/stream":
                project = (query.get("project") or [self.server.project_default])[0]
                limit = max(1, min(200, int((query.get("limit") or ["50"])[0])))
                include_private = parse_bool((query.get("include_private") or ["0"])[0], False)
                self._json({"ok": True, "items": self.server.list_stream(project, limit, include_private)})
                return
            if path == "/api/sessions":
                project = (query.get("project") or [self.server.project_default])[0]
                limit = max(1, min(200, int((query.get("limit") or ["20"])[0])))
                self._json({"ok": True, "items": self.server.list_sessions(project, limit)})
                return
            if path == "/api/nl-search":
                q = (query.get("q") or [""])[0].strip()
                if not q:
                    self._json({"ok": False, "error": "missing q"}, status=400)
                    return
                project = (query.get("project") or [self.server.project_default])[0]
                limit = max(1, min(200, int((query.get("limit") or ["20"])[0])))
                include_private = parse_bool((query.get("include_private") or ["0"])[0], False)
                payload = self.server.nl_search(project=project, query=q, limit=limit, include_private=include_private)
                self._json(payload)
                return
            self._json({"ok": False, "error": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001
            self._json({"ok": False, "error": str(exc)}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/config":
            self._json({"ok": False, "error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("invalid body")
            conn = self.server.open_conn()
            try:
                cfg = set_runtime_config(
                    conn,
                    channel=body.get("channel"),
                    viewer_refresh_sec=body.get("viewer_refresh_sec"),
                    beta_endless_mode=body.get("beta_endless_mode"),
                )
            finally:
                conn.close()
            self._json({"ok": True, "config": cfg})
        except Exception as exc:  # noqa: BLE001
            self._json({"ok": False, "error": str(exc)}, status=400)


class ViewerServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: Tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        root: pathlib.Path,
        index_dir: str,
        project_default: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.root = root
        self.index_dir = index_dir
        self.project_default = project_default

    def open_conn(self):
        return open_db(self.root, self.index_dir)

    def list_stream(self, project: str, limit: int, include_private: bool) -> List[Dict[str, Any]]:
        conn = self.open_conn()
        try:
            where = ["project = ?"]
            params: List[Any] = [project]
            if not include_private:
                where.append("COALESCE(json_extract(metadata_json, '$.privacy.visibility'), 'public') != 'private'")
            sql = (
                "SELECT id, session_id, event_kind, title, content, metadata_json, created_at "
                "FROM events WHERE "
                + " AND ".join(where)
                + " ORDER BY created_at DESC, id DESC LIMIT ?"
            )
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                privacy = meta.get("privacy") or {}
                out.append(
                    {
                        "id": int(row["id"]),
                        "session_id": str(row["session_id"]),
                        "event_kind": str(row["event_kind"]),
                        "title": str(row["title"]),
                        "snippet": str(row["content"])[:220],
                        "visibility": str(privacy.get("visibility", "public")),
                        "created_at": str(row["created_at"]),
                    }
                )
            return out
        finally:
            conn.close()

    def list_sessions(self, project: str, limit: int) -> List[Dict[str, Any]]:
        conn = self.open_conn()
        try:
            rows = conn.execute(
                """
                SELECT session_id, project, title, started_at, ended_at, status, summary_json
                FROM sessions
                WHERE project = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (project, limit),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                summary = json.loads(row["summary_json"]) if row["summary_json"] else {}
                out.append(
                    {
                        "session_id": str(row["session_id"]),
                        "project": str(row["project"]),
                        "title": str(row["title"]),
                        "started_at": str(row["started_at"]),
                        "ended_at": str(row["ended_at"]) if row["ended_at"] else None,
                        "status": str(row["status"]),
                        "summary": summary,
                    }
                )
            return out
        finally:
            conn.close()

    def nl_search(self, *, project: str, query: str, limit: int, include_private: bool) -> Dict[str, Any]:
        conn = self.open_conn()
        try:
            meta = fetch_meta(conn)
            vector_dim = int(meta.get("vector_dim", str(DEFAULT_VECTOR_DIM)))
            parsed = parse_natural_query(query)
            raw_results = blended_search(
                conn,
                query=str(parsed["normalized_query"]),
                project=project,
                session_id=None,
                since=parsed["since"],
                until=parsed["until"],
                include_private=include_private,
                limit=max(10, limit * 2),
                vector_dim=vector_dim,
                alpha=0.7,
            )
            filtered = filter_results_by_intent(
                conn,
                raw_results,
                intent_keywords=parsed["intent_keywords"],
                snippet_chars=DEFAULT_SNIPPET_CHARS,
                include_private=include_private,
            )
            results = filtered[:limit] if filtered else raw_results[:limit]
            return {
                "ok": True,
                "stage": "nl-search",
                "query": query,
                "interpreted": parsed,
                "results": [
                    {
                        "id": item.item_id,
                        "item_type": item.item_type,
                        "kind": item.kind,
                        "title": item.title,
                        "session_id": item.session_id,
                        "created_at": item.created_at,
                        "score": round(item.score, 4),
                    }
                    for item in results
                ],
            }
        finally:
            conn.close()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="codex-mem local web viewer")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--index-dir", default=".codex_mem")
    parser.add_argument("--project-default", default="default")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=37777)
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root).resolve()
    server = ViewerServer(
        (args.host, args.port),
        ViewerHandler,
        root=root,
        index_dir=args.index_dir,
        project_default=args.project_default,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "url": f"http://{args.host}:{args.port}",
                "root": str(root),
                "index_dir": args.index_dir,
                "project_default": args.project_default,
            },
            ensure_ascii=False,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
