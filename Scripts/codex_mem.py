#!/usr/bin/env python3
"""
Codex-specific persistent memory + progressive retrieval CLI.

Key capabilities:
- Local-first durable memory (SQLite + FTS5 + lightweight semantic vectors)
- Five lifecycle hook commands:
  session-start, user-prompt-submit, post-tool-use, stop, session-end
- Three-stage retrieval:
  search (compact index) -> timeline (temporal context) -> get-observations (full detail)
- Memory + repository retrieval fusion via `ask` (integrates with repo_knowledge.py)
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import sqlite3
import struct
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


INDEX_VERSION = "1"
DEFAULT_INDEX_DIR = ".codex_mem"
DEFAULT_DB_NAME = "codex_mem.sqlite3"
DEFAULT_VECTOR_DIM = 256
DEFAULT_PROJECT = "default"
DEFAULT_SNIPPET_CHARS = 240
DEFAULT_TOOL_COMPACT_CHARS = 4000

BLOCKED_MEMORY_TAGS = {"no_mem", "private", "sensitive", "secret"}

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,}|[0-9]+|[\u4e00-\u9fff]+")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclasses.dataclass
class SearchResult:
    item_id: str
    item_type: str
    project: str
    session_id: str
    kind: str
    title: str
    created_at: str
    lexical: float
    semantic: float
    score: float
    token_estimate: int


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for token in TOKEN_RE.findall(text):
        if token.isdigit():
            tokens.append(token)
            continue
        if IDENTIFIER_RE.match(token):
            pieces = [p for p in re.split(r"[_\-]+", token) if p]
            for piece in pieces:
                camel_parts = [p for p in CAMEL_BOUNDARY_RE.split(piece) if p]
                if not camel_parts:
                    continue
                lowered = [part.lower() for part in camel_parts]
                tokens.extend(lowered)
                if len(lowered) > 1:
                    tokens.append("".join(lowered))
            continue
        tokens.append(token.lower())
    return tokens


def vectorize_text(text: str, dim: int) -> List[float]:
    tf = collections.Counter(tokenize(text))
    vec = [0.0] * dim
    for token, count in tf.items():
        base = 1.0 + math.log(count)
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign * base
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def pack_vector(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes, dim: int) -> List[float]:
    if not blob:
        return [0.0] * dim
    expected = struct.calcsize(f"<{dim}f")
    if len(blob) != expected:
        return [0.0] * dim
    return list(struct.unpack(f"<{dim}f", blob))


def normalize_scores(raw: Mapping[str, float]) -> Dict[str, float]:
    if not raw:
        return {}
    values = list(raw.values())
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-9:
        return {k: 1.0 for k in raw}
    return {k: (v - lo) / (hi - lo) for k, v in raw.items()}


def open_db(root: pathlib.Path, index_dir: str) -> sqlite3.Connection:
    base = root / index_dir
    base.mkdir(parents=True, exist_ok=True)
    db_path = base / DEFAULT_DB_NAME
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            title TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            status TEXT NOT NULL,
            summary_json TEXT,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            project TEXT NOT NULL,
            event_kind TEXT NOT NULL,
            role TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_name TEXT,
            file_path TEXT,
            tags_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            vector BLOB,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            project TEXT NOT NULL,
            observation_type TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            source_event_ids_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            vector BLOB,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_session_time
            ON events(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_events_project_time
            ON events(project, created_at);
        CREATE INDEX IF NOT EXISTS idx_obs_session_time
            ON observations(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_obs_project_time
            ON observations(project, created_at);

        CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
            title,
            content,
            tags,
            tokenize='unicode61'
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
            title,
            body,
            tags,
            tokenize='unicode61'
        );
        """
    )
    upsert_meta(conn, "index_version", INDEX_VERSION)
    upsert_meta(conn, "vector_dim", str(DEFAULT_VECTOR_DIM))
    conn.commit()


def upsert_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, value),
    )


def fetch_meta(conn: sqlite3.Connection) -> Dict[str, str]:
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def ensure_session(
    conn: sqlite3.Connection,
    session_id: str,
    project: str,
    title: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    existing = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if existing:
        return
    conn.execute(
        """
        INSERT INTO sessions(session_id, project, title, started_at, status, summary_json, metadata_json)
        VALUES(?, ?, ?, ?, 'active', NULL, ?)
        """,
        (
            session_id,
            project,
            title or f"session-{session_id}",
            now_iso(),
            json.dumps(metadata or {}, ensure_ascii=False),
        ),
    )


def insert_event(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    project: str,
    event_kind: str,
    role: str,
    title: str,
    content: str,
    tool_name: str | None,
    file_path: str | None,
    tags: Sequence[str],
    metadata: Mapping[str, object] | None,
    created_at: str | None = None,
) -> int:
    meta = fetch_meta(conn)
    dim = int(meta.get("vector_dim", str(DEFAULT_VECTOR_DIM)))
    vector_text = " ".join(
        [
            title or "",
            content or "",
            " ".join(tags),
            tool_name or "",
            file_path or "",
            event_kind or "",
        ]
    )
    vector_blob = pack_vector(vectorize_text(vector_text, dim))
    ts = created_at or now_iso()
    tags_clean = [t.strip().lower() for t in tags if t and t.strip()]
    row = conn.execute(
        """
        INSERT INTO events(
            session_id, project, event_kind, role, title, content,
            tool_name, file_path, tags_json, metadata_json, created_at, vector
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            project,
            event_kind,
            role,
            title.strip() or event_kind,
            content.strip(),
            tool_name,
            file_path,
            json.dumps(tags_clean, ensure_ascii=False),
            json.dumps(metadata or {}, ensure_ascii=False),
            ts,
            vector_blob,
        ),
    )
    event_id = int(row.lastrowid)
    conn.execute(
        "INSERT INTO events_fts(rowid, title, content, tags) VALUES(?, ?, ?, ?)",
        (event_id, title.strip() or event_kind, content.strip(), " ".join(tags_clean)),
    )
    return event_id


def insert_observation(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    project: str,
    observation_type: str,
    title: str,
    body: str,
    source_event_ids: Sequence[int],
    metadata: Mapping[str, object] | None = None,
    created_at: str | None = None,
) -> int:
    meta = fetch_meta(conn)
    dim = int(meta.get("vector_dim", str(DEFAULT_VECTOR_DIM)))
    vector_blob = pack_vector(vectorize_text(f"{title}\n{body}\n{observation_type}", dim))
    ts = created_at or now_iso()
    source_ids = [int(v) for v in source_event_ids]
    row = conn.execute(
        """
        INSERT INTO observations(
            session_id, project, observation_type, title, body,
            source_event_ids_json, metadata_json, created_at, vector
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            project,
            observation_type,
            title.strip() or observation_type,
            body.strip(),
            json.dumps(source_ids, ensure_ascii=False),
            json.dumps(metadata or {}, ensure_ascii=False),
            ts,
            vector_blob,
        ),
    )
    obs_id = int(row.lastrowid)
    tags = [observation_type, project]
    conn.execute(
        "INSERT INTO observations_fts(rowid, title, body, tags) VALUES(?, ?, ?, ?)",
        (obs_id, title.strip() or observation_type, body.strip(), " ".join(tags)),
    )
    return obs_id


def build_fts_query(query: str) -> str:
    toks = [tok for tok in tokenize(query) if tok]
    if not toks:
        return ""
    uniq: List[str] = []
    seen = set()
    for tok in toks:
        if tok in seen:
            continue
        seen.add(tok)
        uniq.append(tok)
    if not uniq:
        return ""
    fragments = [f"{tok}*" for tok in uniq[:12]]
    return " OR ".join(fragments)


def db_supports_fts5(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='events_fts'"
    ).fetchone()
    return bool(row)


def search_events(
    conn: sqlite3.Connection,
    *,
    query: str,
    project: str | None,
    session_id: str | None,
    limit: int,
) -> List[sqlite3.Row]:
    if not db_supports_fts5(conn):
        return []
    match_query = build_fts_query(query)
    if not match_query:
        return []
    clauses = ["events_fts MATCH ?"]
    params: List[object] = [match_query]
    if project:
        clauses.append("e.project = ?")
        params.append(project)
    if session_id:
        clauses.append("e.session_id = ?")
        params.append(session_id)
    where_sql = " AND ".join(clauses)
    sql = f"""
        SELECT
            e.id,
            e.project,
            e.session_id,
            e.event_kind,
            e.title,
            e.created_at,
            e.vector,
            bm25(events_fts) AS bm25
        FROM events_fts
        JOIN events e ON e.id = events_fts.rowid
        WHERE {where_sql}
        ORDER BY bm25(events_fts)
        LIMIT ?
    """
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def search_observations(
    conn: sqlite3.Connection,
    *,
    query: str,
    project: str | None,
    session_id: str | None,
    limit: int,
) -> List[sqlite3.Row]:
    if not db_supports_fts5(conn):
        return []
    match_query = build_fts_query(query)
    if not match_query:
        return []
    clauses = ["observations_fts MATCH ?"]
    params: List[object] = [match_query]
    if project:
        clauses.append("o.project = ?")
        params.append(project)
    if session_id:
        clauses.append("o.session_id = ?")
        params.append(session_id)
    where_sql = " AND ".join(clauses)
    sql = f"""
        SELECT
            o.id,
            o.project,
            o.session_id,
            o.observation_type,
            o.title,
            o.created_at,
            o.vector,
            bm25(observations_fts) AS bm25
        FROM observations_fts
        JOIN observations o ON o.id = observations_fts.rowid
        WHERE {where_sql}
        ORDER BY bm25(observations_fts)
        LIMIT ?
    """
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def blended_search(
    conn: sqlite3.Connection,
    *,
    query: str,
    project: str | None,
    session_id: str | None,
    limit: int,
    vector_dim: int,
    alpha: float,
) -> List[SearchResult]:
    event_rows = search_events(
        conn,
        query=query,
        project=project,
        session_id=session_id,
        limit=max(10, limit * 3),
    )
    obs_rows = search_observations(
        conn,
        query=query,
        project=project,
        session_id=session_id,
        limit=max(10, limit * 3),
    )

    q_vec = vectorize_text(query, vector_dim)
    raw_lexical: Dict[str, float] = {}
    raw_semantic: Dict[str, float] = {}
    stash: Dict[str, SearchResult] = {}

    for row in event_rows:
        key = f"E{int(row['id'])}"
        bm = float(row["bm25"]) if row["bm25"] is not None else 0.0
        lexical = 1.0 / (1.0 + abs(bm))
        sem = max(0.0, cosine_sim(q_vec, unpack_vector(row["vector"], vector_dim)))
        raw_lexical[key] = lexical
        raw_semantic[key] = sem
        title = str(row["title"] or "").strip()
        item = SearchResult(
            item_id=key,
            item_type="event",
            project=str(row["project"]),
            session_id=str(row["session_id"]),
            kind=str(row["event_kind"]),
            title=title,
            created_at=str(row["created_at"]),
            lexical=0.0,
            semantic=0.0,
            score=0.0,
            token_estimate=estimate_tokens(f"{key} {title} {row['event_kind']}"),
        )
        stash[key] = item

    for row in obs_rows:
        key = f"O{int(row['id'])}"
        bm = float(row["bm25"]) if row["bm25"] is not None else 0.0
        lexical = 1.0 / (1.0 + abs(bm))
        sem = max(0.0, cosine_sim(q_vec, unpack_vector(row["vector"], vector_dim)))
        raw_lexical[key] = lexical
        raw_semantic[key] = sem
        title = str(row["title"] or "").strip()
        item = SearchResult(
            item_id=key,
            item_type="observation",
            project=str(row["project"]),
            session_id=str(row["session_id"]),
            kind=str(row["observation_type"]),
            title=title,
            created_at=str(row["created_at"]),
            lexical=0.0,
            semantic=0.0,
            score=0.0,
            token_estimate=estimate_tokens(f"{key} {title} {row['observation_type']}"),
        )
        stash[key] = item

    if not stash:
        return []

    lex_norm = normalize_scores(raw_lexical)
    sem_norm = normalize_scores(raw_semantic)
    out: List[SearchResult] = []
    for key, item in stash.items():
        lx = lex_norm.get(key, 0.0)
        sm = sem_norm.get(key, 0.0)
        score = alpha * lx + (1.0 - alpha) * sm
        item.lexical = lx
        item.semantic = sm
        item.score = score
        out.append(item)
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:limit]


def parse_item_id(token: str) -> Tuple[str, int]:
    value = token.strip()
    if not value:
        raise ValueError("empty id")
    prefix = value[0].upper()
    if prefix in {"E", "O"} and value[1:].isdigit():
        return ("event" if prefix == "E" else "observation", int(value[1:]))
    if value.isdigit():
        return ("event", int(value))
    raise ValueError(f"unsupported id format: {token}")


def trim_snippet(text: str, limit: int) -> str:
    txt = text.strip()
    if len(txt) <= limit:
        return txt
    return txt[:limit].rstrip() + "...<trimmed>"


def compact_tool_output(text: str, max_chars: int) -> Tuple[str, Dict[str, int]]:
    """
    Deterministic lightweight compaction used for high-volume tool outputs.
    Keeps head+tail and distinct signal lines to reduce token usage.
    """
    raw = text.strip()
    raw_chars = len(raw)
    if raw_chars <= max_chars:
        return raw, {"raw_chars": raw_chars, "final_chars": raw_chars, "compacted": 0}

    lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
    signal: List[str] = []
    seen = set()
    for ln in lines:
        lower = ln.lower()
        if any(k in lower for k in ("error", "warning", "fail", "trace", "exception", "todo", "fix", "done")):
            if ln not in seen:
                signal.append(ln)
                seen.add(ln)
        if len(signal) >= 20:
            break
    head = raw[: max(256, max_chars // 3)]
    tail = raw[-max(256, max_chars // 4) :]
    sections = [
        "[compacted tool output]",
        "head:",
        head,
        "",
        "signal:",
        "\n".join(signal[:20]) if signal else "<none>",
        "",
        "tail:",
        tail,
    ]
    merged = "\n".join(sections)
    if len(merged) > max_chars:
        merged = merged[:max_chars].rstrip() + "\n...<trimmed>"
    final_chars = len(merged)
    return merged, {"raw_chars": raw_chars, "final_chars": final_chars, "compacted": 1}


def timeline_for_event(
    conn: sqlite3.Connection,
    *,
    event_id: int,
    before: int,
    after: int,
    snippet_chars: int,
) -> Dict[str, object]:
    anchor = conn.execute(
        """
        SELECT id, session_id, project, event_kind, title, content, tool_name, file_path, created_at
        FROM events WHERE id = ?
        """,
        (event_id,),
    ).fetchone()
    if not anchor:
        raise ValueError(f"event not found: E{event_id}")

    session_id = str(anchor["session_id"])
    created_at = str(anchor["created_at"])
    before_rows = conn.execute(
        """
        SELECT id, event_kind, title, content, created_at
        FROM events
        WHERE session_id = ? AND created_at < ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (session_id, created_at, max(0, before)),
    ).fetchall()
    after_rows = conn.execute(
        """
        SELECT id, event_kind, title, content, created_at
        FROM events
        WHERE session_id = ? AND created_at > ?
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (session_id, created_at, max(0, after)),
    ).fetchall()
    return {
        "anchor": {
            "id": f"E{int(anchor['id'])}",
            "session_id": session_id,
            "project": str(anchor["project"]),
            "kind": str(anchor["event_kind"]),
            "title": str(anchor["title"]),
            "content": str(anchor["content"]),
            "tool_name": anchor["tool_name"],
            "file_path": anchor["file_path"],
            "created_at": created_at,
        },
        "before": [
            {
                "id": f"E{int(row['id'])}",
                "kind": str(row["event_kind"]),
                "title": str(row["title"]),
                "snippet": trim_snippet(str(row["content"]), snippet_chars),
                "created_at": str(row["created_at"]),
                "token_estimate": estimate_tokens(
                    f"{row['title']} {trim_snippet(str(row['content']), snippet_chars)}"
                ),
            }
            for row in before_rows
        ],
        "after": [
            {
                "id": f"E{int(row['id'])}",
                "kind": str(row["event_kind"]),
                "title": str(row["title"]),
                "snippet": trim_snippet(str(row["content"]), snippet_chars),
                "created_at": str(row["created_at"]),
                "token_estimate": estimate_tokens(
                    f"{row['title']} {trim_snippet(str(row['content']), snippet_chars)}"
                ),
            }
            for row in after_rows
        ],
    }


def timeline_for_observation(
    conn: sqlite3.Connection,
    *,
    obs_id: int,
    before: int,
    after: int,
    snippet_chars: int,
) -> Dict[str, object]:
    obs = conn.execute(
        """
        SELECT id, session_id, project, observation_type, title, body, source_event_ids_json, created_at
        FROM observations WHERE id = ?
        """,
        (obs_id,),
    ).fetchone()
    if not obs:
        raise ValueError(f"observation not found: O{obs_id}")
    source_ids = json.loads(obs["source_event_ids_json"]) if obs["source_event_ids_json"] else []
    timeline = {
        "anchor": {
            "id": f"O{int(obs['id'])}",
            "session_id": str(obs["session_id"]),
            "project": str(obs["project"]),
            "kind": str(obs["observation_type"]),
            "title": str(obs["title"]),
            "body": str(obs["body"]),
            "source_event_ids": [f"E{int(v)}" for v in source_ids],
            "created_at": str(obs["created_at"]),
        },
        "before": [],
        "after": [],
    }
    if source_ids:
        seed = int(source_ids[0])
        linked = timeline_for_event(
            conn,
            event_id=seed,
            before=before,
            after=after,
            snippet_chars=snippet_chars,
        )
        timeline["before"] = linked["before"]
        timeline["after"] = linked["after"]
    return timeline


def get_item_detail(conn: sqlite3.Connection, item_type: str, item_id: int, snippet_chars: int) -> Dict[str, object]:
    if item_type == "event":
        row = conn.execute(
            """
            SELECT id, session_id, project, event_kind, role, title, content, tool_name, file_path,
                   tags_json, metadata_json, created_at
            FROM events WHERE id = ?
            """,
            (item_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"event not found: E{item_id}")
        content = str(row["content"])
        return {
            "id": f"E{int(row['id'])}",
            "item_type": "event",
            "project": str(row["project"]),
            "session_id": str(row["session_id"]),
            "kind": str(row["event_kind"]),
            "role": str(row["role"]),
            "title": str(row["title"]),
            "content": content,
            "snippet": trim_snippet(content, snippet_chars),
            "tool_name": row["tool_name"],
            "file_path": row["file_path"],
            "tags": json.loads(row["tags_json"]) if row["tags_json"] else [],
            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            "created_at": str(row["created_at"]),
            "token_estimate": estimate_tokens(content),
        }

    row = conn.execute(
        """
        SELECT id, session_id, project, observation_type, title, body, source_event_ids_json,
               metadata_json, created_at
        FROM observations WHERE id = ?
        """,
        (item_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"observation not found: O{item_id}")
    body = str(row["body"])
    source_ids = json.loads(row["source_event_ids_json"]) if row["source_event_ids_json"] else []
    return {
        "id": f"O{int(row['id'])}",
        "item_type": "observation",
        "project": str(row["project"]),
        "session_id": str(row["session_id"]),
        "kind": str(row["observation_type"]),
        "title": str(row["title"]),
        "content": body,
        "snippet": trim_snippet(body, snippet_chars),
        "source_event_ids": [f"E{int(v)}" for v in source_ids],
        "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        "created_at": str(row["created_at"]),
        "token_estimate": estimate_tokens(body),
    }


def summarize_session(conn: sqlite3.Connection, session_id: str) -> Dict[str, object]:
    rows = conn.execute(
        """
        SELECT id, event_kind, role, title, content, tool_name, created_at
        FROM events
        WHERE session_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (session_id,),
    ).fetchall()
    session = conn.execute(
        "SELECT project, title, started_at, ended_at, metadata_json FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if not session:
        raise ValueError(f"session not found: {session_id}")

    project = str(session["project"])
    investigation: List[str] = []
    learnings: List[str] = []
    completed: List[str] = []
    next_steps: List[str] = []
    source_ids = [int(row["id"]) for row in rows]

    for row in rows:
        title = str(row["title"] or "").strip()
        content = str(row["content"] or "").strip()
        kind = str(row["event_kind"])
        tool = str(row["tool_name"] or "").strip().lower()
        joined = f"{title}\n{content}".lower()
        if kind == "post_tool_use" or tool:
            if title:
                investigation.append(title)
            elif content:
                investigation.append(trim_snippet(content, 120))
        if any(k in joined for k in ("learn", "结论", "发现", "原因", "root cause", "insight")):
            learnings.append(title or trim_snippet(content, 120))
        if any(k in joined for k in ("done", "fixed", "implemented", "完成", "修复", "已做")):
            completed.append(title or trim_snippet(content, 120))
        if any(k in joined for k in ("todo", "next", "follow-up", "下一步", "待办")):
            next_steps.append(title or trim_snippet(content, 120))

    if not investigation and rows:
        investigation = [str(row["title"]) for row in rows if str(row["title"]).strip()][:5]
    if not learnings:
        learnings = [trim_snippet(str(row["content"]), 120) for row in rows if str(row["content"]).strip()][:3]
    if not completed:
        completed = [item for item in investigation[:3]]
    if not next_steps:
        next_steps = ["继续下一轮任务并先验证回归风险。"]

    def uniq(seq: Iterable[str], cap: int) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in seq:
            s = item.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= cap:
                break
        return out

    summary = {
        "session_id": session_id,
        "project": project,
        "title": str(session["title"]),
        "started_at": str(session["started_at"]),
        "ended_at": str(session["ended_at"] or now_iso()),
        "investigation": uniq(investigation, 8),
        "learnings": uniq(learnings, 8),
        "completed_work": uniq(completed, 8),
        "next_steps": uniq(next_steps, 8),
        "event_count": len(rows),
    }

    conn.execute(
        "UPDATE sessions SET summary_json = ? WHERE session_id = ?",
        (json.dumps(summary, ensure_ascii=False), session_id),
    )

    # Rebuild auto-generated observations for this session.
    stale_rows = conn.execute(
        """
        SELECT id FROM observations
        WHERE session_id = ? AND json_extract(metadata_json, '$.auto_generated') = 1
        """,
        (session_id,),
    ).fetchall()
    stale_ids = [int(row["id"]) for row in stale_rows]
    if stale_ids:
        conn.executemany("DELETE FROM observations WHERE id = ?", [(oid,) for oid in stale_ids])
        conn.executemany("DELETE FROM observations_fts WHERE rowid = ?", [(oid,) for oid in stale_ids])

    for obs_type, key in (
        ("investigation", "investigation"),
        ("learning", "learnings"),
        ("completed_work", "completed_work"),
        ("next_step", "next_steps"),
    ):
        body = "\n".join(f"- {line}" for line in summary[key])
        if not body.strip():
            continue
        insert_observation(
            conn,
            session_id=session_id,
            project=project,
            observation_type=obs_type,
            title=f"{obs_type.replace('_', ' ').title()} summary ({session_id})",
            body=body,
            source_event_ids=source_ids[:30],
            metadata={"auto_generated": 1},
        )
    return summary


def run_repo_query(
    root: pathlib.Path,
    *,
    question: str,
    top_k: int,
    module_limit: int,
    snippet_chars: int,
    index_dir: str = ".codex_knowledge",
) -> Dict[str, object]:
    script = root / "Scripts" / "repo_knowledge.py"
    if not script.exists():
        return {"warning": "repo_knowledge.py not found"}
    cmd = [
        sys.executable,
        str(script),
        "--root",
        str(root),
        "--index-dir",
        index_dir,
        "query",
        question,
        "--json",
        "--top-k",
        str(top_k),
        "--module-limit",
        str(module_limit),
        "--snippet-chars",
        str(snippet_chars),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {"warning": f"failed to run repo_knowledge.py: {exc}"}
    if proc.returncode != 0:
        return {
            "warning": "repo_knowledge query failed",
            "exit_code": proc.returncode,
            "stderr": trim_snippet(proc.stderr, 400),
        }
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"warning": "invalid json from repo_knowledge query"}


def cmd_init(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    upsert_meta(conn, "project_default", args.project)
    conn.commit()
    print(
        json.dumps(
            {
                "ok": True,
                "db": str(root / args.index_dir / DEFAULT_DB_NAME),
                "project_default": args.project,
                "index_version": INDEX_VERSION,
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_session_start(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    ensure_session(
        conn,
        session_id=args.session_id,
        project=args.project,
        title=args.title,
        metadata={"hook": "SessionStart"},
    )
    insert_event(
        conn,
        session_id=args.session_id,
        project=args.project,
        event_kind="session_start",
        role="system",
        title=args.title or f"SessionStart {args.session_id}",
        content=args.content or "",
        tool_name=None,
        file_path=None,
        tags=["session", "start"],
        metadata={"hook": "SessionStart"},
    )
    conn.commit()
    print(json.dumps({"ok": True, "session_id": args.session_id, "hook": "SessionStart"}, ensure_ascii=False))
    return 0


def cmd_user_prompt_submit(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    ensure_session(conn, args.session_id, args.project)
    event_id = insert_event(
        conn,
        session_id=args.session_id,
        project=args.project,
        event_kind="user_prompt_submit",
        role="user",
        title=args.title or trim_snippet(args.prompt, 80),
        content=args.prompt,
        tool_name=None,
        file_path=None,
        tags=["prompt", "user"],
        metadata={"hook": "UserPromptSubmit"},
    )
    conn.commit()
    print(json.dumps({"ok": True, "event_id": f"E{event_id}", "hook": "UserPromptSubmit"}, ensure_ascii=False))
    return 0


def cmd_post_tool_use(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    ensure_session(conn, args.session_id, args.project)
    tags = ["tool", args.tool_name.strip().lower()]
    if args.tag:
        tags.extend(args.tag)
    tags_clean = [t.strip().lower() for t in tags if t and t.strip()]
    if any(tag in BLOCKED_MEMORY_TAGS for tag in tags_clean):
        print(
            json.dumps(
                {
                    "ok": True,
                    "hook": "PostToolUse",
                    "skipped": True,
                    "reason": "blocked_by_tag",
                    "blocked_tags": [t for t in tags_clean if t in BLOCKED_MEMORY_TAGS],
                },
                ensure_ascii=False,
            )
        )
        return 0

    content = args.content
    compact_meta: Dict[str, int] = {"raw_chars": len(content), "final_chars": len(content), "compacted": 0}
    if args.compact:
        content, compact_meta = compact_tool_output(content, args.compact_chars)

    meta: Dict[str, object] = {"hook": "PostToolUse", "exit_code": args.exit_code, "compaction": compact_meta}
    event_id = insert_event(
        conn,
        session_id=args.session_id,
        project=args.project,
        event_kind="post_tool_use",
        role="tool",
        title=args.title or f"Tool {args.tool_name}",
        content=content,
        tool_name=args.tool_name,
        file_path=args.file_path,
        tags=tags_clean,
        metadata=meta,
    )
    conn.commit()
    payload = {"ok": True, "event_id": f"E{event_id}", "hook": "PostToolUse", "compaction": compact_meta}
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    ensure_session(conn, args.session_id, args.project)
    event_id = insert_event(
        conn,
        session_id=args.session_id,
        project=args.project,
        event_kind="stop",
        role="assistant",
        title=args.title or "Stop",
        content=args.content or "",
        tool_name=None,
        file_path=None,
        tags=["stop"],
        metadata={"hook": "Stop"},
    )
    conn.commit()
    print(json.dumps({"ok": True, "event_id": f"E{event_id}", "hook": "Stop"}, ensure_ascii=False))
    return 0


def cmd_session_end(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    ensure_session(conn, args.session_id, args.project)
    insert_event(
        conn,
        session_id=args.session_id,
        project=args.project,
        event_kind="session_end",
        role="system",
        title=args.title or f"SessionEnd {args.session_id}",
        content=args.content or "",
        tool_name=None,
        file_path=None,
        tags=["session", "end"],
        metadata={"hook": "SessionEnd"},
    )
    conn.execute(
        "UPDATE sessions SET ended_at = ?, status = 'ended' WHERE session_id = ?",
        (now_iso(), args.session_id),
    )
    summary = None
    if not args.skip_summary:
        summary = summarize_session(conn, args.session_id)
    conn.commit()
    payload: Dict[str, object] = {"ok": True, "session_id": args.session_id, "hook": "SessionEnd"}
    if summary:
        payload["summary"] = summary
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    ensure_session(conn, args.session_id, args.project)
    metadata: Dict[str, object] = {}
    if args.metadata_json:
        metadata = json.loads(args.metadata_json)
    event_id = insert_event(
        conn,
        session_id=args.session_id,
        project=args.project,
        event_kind=args.event_kind,
        role=args.role,
        title=args.title,
        content=args.content,
        tool_name=args.tool_name,
        file_path=args.file_path,
        tags=args.tag or [],
        metadata=metadata,
    )
    conn.commit()
    print(json.dumps({"ok": True, "event_id": f"E{event_id}"}, ensure_ascii=False))
    return 0


def cmd_summarize_session(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    summary = summarize_session(conn, args.session_id)
    conn.commit()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    meta = fetch_meta(conn)
    vector_dim = int(meta.get("vector_dim", str(DEFAULT_VECTOR_DIM)))
    results = blended_search(
        conn,
        query=args.query,
        project=args.project,
        session_id=args.session_id,
        limit=args.limit,
        vector_dim=vector_dim,
        alpha=args.alpha,
    )
    payload = {
        "query": args.query,
        "stage": "search",
        "results": [
            {
                "id": item.item_id,
                "item_type": item.item_type,
                "project": item.project,
                "session_id": item.session_id,
                "kind": item.kind,
                "title": item.title,
                "created_at": item.created_at,
                "score": round(item.score, 4),
                "lexical": round(item.lexical, 4),
                "semantic": round(item.semantic, 4),
                "token_estimate": item.token_estimate,
            }
            for item in results
        ],
        "token_estimate_total": sum(item.token_estimate for item in results),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_timeline(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    item_type, item_id = parse_item_id(args.id)
    if item_type == "event":
        payload = timeline_for_event(
            conn,
            event_id=item_id,
            before=args.before,
            after=args.after,
            snippet_chars=args.snippet_chars,
        )
    else:
        payload = timeline_for_observation(
            conn,
            obs_id=item_id,
            before=args.before,
            after=args.after,
            snippet_chars=args.snippet_chars,
        )
    payload["stage"] = "timeline"
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_get_observations(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    details: List[Dict[str, object]] = []
    for token in args.ids:
        item_type, item_id = parse_item_id(token)
        detail = get_item_detail(conn, item_type, item_id, args.snippet_chars)
        if args.compact:
            detail["content"] = detail["snippet"]
            detail["token_estimate"] = estimate_tokens(str(detail["snippet"]))
        details.append(detail)
    payload = {
        "stage": "get_observations",
        "count": len(details),
        "items": details,
        "token_estimate_total": sum(int(item["token_estimate"]) for item in details),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_fused_prompt(
    question: str,
    memory_details: Sequence[Mapping[str, object]],
    repo_payload: Mapping[str, object],
    snippet_chars: int,
) -> str:
    lines: List[str] = []
    lines.append("System: 你是 Codex 代码助手。优先依据提供的 Memory + Repo 上下文回答。")
    lines.append("若证据不足，明确说明缺失信息。")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Memory Contexts:")
    for idx, item in enumerate(memory_details, start=1):
        lines.append(f"[M{idx}] {item['id']} {item['kind']} {item['title']}")
        lines.append(trim_snippet(str(item.get("content", "")), snippet_chars))
        lines.append("")
    lines.append("Repo Contexts:")
    chunks = repo_payload.get("chunks") if isinstance(repo_payload, dict) else None
    if isinstance(chunks, list):
        for idx, chunk in enumerate(chunks, start=1):
            path = chunk.get("path")
            start = chunk.get("start_line")
            end = chunk.get("end_line")
            snippet = chunk.get("snippet", "")
            lines.append(f"[R{idx}] {path}:{start}-{end}")
            lines.append(trim_snippet(str(snippet), snippet_chars))
            lines.append("")
    else:
        lines.append("- <repo context unavailable>")
    return "\n".join(lines).strip()


def cmd_ask(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    meta = fetch_meta(conn)
    vector_dim = int(meta.get("vector_dim", str(DEFAULT_VECTOR_DIM)))

    layer1 = blended_search(
        conn,
        query=args.question,
        project=args.project,
        session_id=args.session_id,
        limit=args.search_limit,
        vector_dim=vector_dim,
        alpha=args.alpha,
    )
    selected_ids = [item.item_id for item in layer1[: args.detail_limit]]
    details = []
    for ident in selected_ids:
        typ, iid = parse_item_id(ident)
        details.append(get_item_detail(conn, typ, iid, args.snippet_chars))

    repo_payload = run_repo_query(
        root,
        question=args.question,
        top_k=args.code_top_k,
        module_limit=args.code_module_limit,
        snippet_chars=args.snippet_chars,
        index_dir=args.repo_index_dir,
    )

    memory_l1_tokens = sum(item.token_estimate for item in layer1)
    memory_l3_tokens = sum(int(item["token_estimate"]) for item in details)
    code_tokens = 0
    if isinstance(repo_payload, dict):
        chunks = repo_payload.get("chunks")
        if isinstance(chunks, list):
            code_tokens = sum(estimate_tokens(str(c.get("snippet", ""))) for c in chunks)

    prompt = build_fused_prompt(args.question, details, repo_payload, args.snippet_chars)
    payload = {
        "question": args.question,
        "stage": "fused_ask",
        "layer1_search": [
            {
                "id": item.item_id,
                "item_type": item.item_type,
                "kind": item.kind,
                "title": item.title,
                "score": round(item.score, 4),
                "token_estimate": item.token_estimate,
                "session_id": item.session_id,
                "created_at": item.created_at,
            }
            for item in layer1
        ],
        "layer3_observations": details,
        "repo_context": repo_payload,
        "token_estimate": {
            "memory_layer1": memory_l1_tokens,
            "memory_layer3": memory_l3_tokens,
            "repo_context": code_tokens,
            "total": memory_l1_tokens + memory_l3_tokens + code_tokens,
        },
        "suggested_prompt": prompt,
    }
    if args.prompt_only:
        print(prompt)
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Codex local memory with progressive disclosure retrieval."
    )
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR, help=f"Index dir (default: {DEFAULT_INDEX_DIR}).")

    sub = parser.add_subparsers(dest="command", required=True)

    def add_project_arg(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--project", default=DEFAULT_PROJECT)

    p_init = sub.add_parser("init", help="Initialize memory database.")
    add_project_arg(p_init)
    p_init.set_defaults(func=cmd_init)

    p_start = sub.add_parser("session-start", help="Lifecycle hook: SessionStart.")
    add_project_arg(p_start)
    p_start.add_argument("session_id")
    p_start.add_argument("--title", default=None)
    p_start.add_argument("--content", default="")
    p_start.set_defaults(func=cmd_session_start)

    p_prompt = sub.add_parser("user-prompt-submit", help="Lifecycle hook: UserPromptSubmit.")
    add_project_arg(p_prompt)
    p_prompt.add_argument("session_id")
    p_prompt.add_argument("prompt")
    p_prompt.add_argument("--title", default=None)
    p_prompt.set_defaults(func=cmd_user_prompt_submit)

    p_tool = sub.add_parser("post-tool-use", help="Lifecycle hook: PostToolUse.")
    add_project_arg(p_tool)
    p_tool.add_argument("session_id")
    p_tool.add_argument("tool_name")
    p_tool.add_argument("content")
    p_tool.add_argument("--title", default=None)
    p_tool.add_argument("--file-path", default=None)
    p_tool.add_argument("--exit-code", type=int, default=0)
    p_tool.add_argument("--tag", action="append", default=[])
    p_tool.add_argument("--compact", action="store_true", help="Compact large tool output before storing.")
    p_tool.add_argument(
        "--compact-chars",
        type=int,
        default=DEFAULT_TOOL_COMPACT_CHARS,
        help=f"Target max chars after compaction (default: {DEFAULT_TOOL_COMPACT_CHARS}).",
    )
    p_tool.set_defaults(func=cmd_post_tool_use)

    p_stop = sub.add_parser("stop", help="Lifecycle hook: Stop.")
    add_project_arg(p_stop)
    p_stop.add_argument("session_id")
    p_stop.add_argument("--title", default="Stop")
    p_stop.add_argument("--content", default="")
    p_stop.set_defaults(func=cmd_stop)

    p_end = sub.add_parser("session-end", help="Lifecycle hook: SessionEnd.")
    add_project_arg(p_end)
    p_end.add_argument("session_id")
    p_end.add_argument("--title", default=None)
    p_end.add_argument("--content", default="")
    p_end.add_argument("--skip-summary", action="store_true")
    p_end.set_defaults(func=cmd_session_end)

    p_log = sub.add_parser("log", help="Generic event logging.")
    add_project_arg(p_log)
    p_log.add_argument("session_id")
    p_log.add_argument("event_kind")
    p_log.add_argument("role")
    p_log.add_argument("title")
    p_log.add_argument("content")
    p_log.add_argument("--tool-name", default=None)
    p_log.add_argument("--file-path", default=None)
    p_log.add_argument("--tag", action="append", default=[])
    p_log.add_argument("--metadata-json", default="")
    p_log.set_defaults(func=cmd_log)

    p_sum = sub.add_parser("summarize-session", help="Generate structured session summary.")
    p_sum.add_argument("session_id")
    p_sum.set_defaults(func=cmd_summarize_session)

    p_search = sub.add_parser("search", help="Stage 1 retrieval: compact index hits.")
    add_project_arg(p_search)
    p_search.add_argument("query")
    p_search.add_argument("--session-id", default=None)
    p_search.add_argument("--limit", type=int, default=20)
    p_search.add_argument("--alpha", type=float, default=0.7, help="Lexical/semantic blend.")
    p_search.set_defaults(func=cmd_search)

    p_timeline = sub.add_parser("timeline", help="Stage 2 retrieval: temporal neighborhood.")
    p_timeline.add_argument("id", help="E<ID> or O<ID>")
    p_timeline.add_argument("--before", type=int, default=5)
    p_timeline.add_argument("--after", type=int, default=5)
    p_timeline.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS)
    p_timeline.set_defaults(func=cmd_timeline)

    p_get = sub.add_parser("get-observations", help="Stage 3 retrieval: full details by IDs.")
    p_get.add_argument("ids", nargs="+", help="List of E<ID>/O<ID>")
    p_get.add_argument("--compact", action="store_true")
    p_get.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS)
    p_get.set_defaults(func=cmd_get_observations)

    p_ask = sub.add_parser("ask", help="Fused memory + repo_knowledge retrieval.")
    add_project_arg(p_ask)
    p_ask.add_argument("question")
    p_ask.add_argument("--session-id", default=None)
    p_ask.add_argument("--search-limit", type=int, default=20)
    p_ask.add_argument("--detail-limit", type=int, default=6)
    p_ask.add_argument("--code-top-k", type=int, default=8)
    p_ask.add_argument("--code-module-limit", type=int, default=6)
    p_ask.add_argument("--repo-index-dir", default=".codex_knowledge")
    p_ask.add_argument("--alpha", type=float, default=0.7)
    p_ask.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS)
    p_ask.add_argument("--prompt-only", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
