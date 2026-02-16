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
import time
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from prompt_budgeter import build_prompt_plan
from prompt_mapper import map_prompt_to_profile
from prompt_profiles import get_prompt_profile
from prompt_renderer import render_compact_prompt


INDEX_VERSION = "1"
DEFAULT_INDEX_DIR = ".codex_mem"
DEFAULT_DB_NAME = "codex_mem.sqlite3"
DEFAULT_VECTOR_DIM = 256
DEFAULT_PROJECT = "default"
DEFAULT_SNIPPET_CHARS = 240
DEFAULT_TOOL_COMPACT_CHARS = 4000
DEFAULT_CHANNEL = "stable"
CHANNEL_CHOICES = {"stable", "beta"}

PRIVACY_BLOCK_TAGS = {"no_mem", "block", "skip", "secret_block"}
PRIVACY_PRIVATE_TAGS = {"private", "sensitive", "secret"}
PRIVACY_REDACT_TAGS = {"redact", "mask", "sensitive", "secret"}
DEFAULT_REDACTION_RULES = (
    (
        re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*([^\s,;]+)"),
        r"\1=[REDACTED]",
    ),
    (
        re.compile(r"(?i)bearer\s+[a-z0-9._\-]+"),
        "Bearer [REDACTED]",
    ),
    (
        re.compile(r"\b(?:github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]+)\b"),
        "[REDACTED_TOKEN]",
    ),
    (
        re.compile(
            r"\b(?:sk-[A-Za-z0-9]{10,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|AIza[0-9A-Za-z\\-_]{10,})\b"
        ),
        "[REDACTED_TOKEN]",
    ),
)

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
    if base.exists() and not base.is_dir():
        raise sqlite3.OperationalError(f"index dir is not a directory: {base}")
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise sqlite3.OperationalError(f"failed to prepare index dir {base}: {exc}") from exc

    db_path = base / DEFAULT_DB_NAME
    last_exc: sqlite3.OperationalError | None = None
    for attempt in range(2):
        try:
            conn = sqlite3.connect(str(db_path), timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            init_schema(conn)
            return conn
        except sqlite3.OperationalError as exc:
            last_exc = exc
            # Retry once for transient lock contention.
            if "locked" in str(exc).lower() and attempt == 0:
                time.sleep(0.2)
                continue
            break

    legacy_db_path = base / "memory.db"
    legacy_hint = (
        f" Legacy database detected at {legacy_db_path}; current default is {db_path.name}."
        if legacy_db_path.exists()
        else ""
    )
    detail = str(last_exc) if last_exc else "unknown sqlite operational error"
    raise sqlite3.OperationalError(
        f"failed to open sqlite database at {db_path}: {detail}.{legacy_hint} "
        "Try: `bash Scripts/codex_mem.sh init --project demo` and retry."
    ) from last_exc


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
    ensure_meta_default(conn, "index_version", INDEX_VERSION)
    ensure_meta_default(conn, "vector_dim", str(DEFAULT_VECTOR_DIM))
    ensure_meta_default(conn, "channel", DEFAULT_CHANNEL)
    ensure_meta_default(conn, "viewer_refresh_sec", "3")
    ensure_meta_default(conn, "beta_endless_mode", "0")
    conn.commit()


def upsert_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, value),
    )


def ensure_meta_default(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO NOTHING
        """,
        (key, value),
    )


def fetch_meta(conn: sqlite3.Connection) -> Dict[str, str]:
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def get_runtime_config(conn: sqlite3.Connection) -> Dict[str, object]:
    meta = fetch_meta(conn)
    channel = str(meta.get("channel", DEFAULT_CHANNEL)).strip().lower()
    if channel not in CHANNEL_CHOICES:
        channel = DEFAULT_CHANNEL
    viewer_refresh_sec = int(meta.get("viewer_refresh_sec", "3"))
    viewer_refresh_sec = max(1, min(60, viewer_refresh_sec))
    beta_endless_mode = str(meta.get("beta_endless_mode", "0")).strip() in {"1", "true", "yes"}
    return {
        "channel": channel,
        "viewer_refresh_sec": viewer_refresh_sec,
        "beta_endless_mode": beta_endless_mode,
    }


def set_runtime_config(
    conn: sqlite3.Connection,
    *,
    channel: str | None = None,
    viewer_refresh_sec: int | None = None,
    beta_endless_mode: bool | None = None,
) -> Dict[str, object]:
    if channel is not None:
        val = str(channel).strip().lower()
        if val not in CHANNEL_CHOICES:
            raise ValueError(f"Unsupported channel: {channel}")
        upsert_meta(conn, "channel", val)
    if viewer_refresh_sec is not None:
        sec = max(1, min(60, int(viewer_refresh_sec)))
        upsert_meta(conn, "viewer_refresh_sec", str(sec))
    if beta_endless_mode is not None:
        upsert_meta(conn, "beta_endless_mode", "1" if beta_endless_mode else "0")
    conn.commit()
    return get_runtime_config(conn)


def parse_iso_datetime(value: str) -> dt.datetime:
    txt = value.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(txt)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def parse_iso_datetime_maybe(value: str) -> dt.datetime | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return parse_iso_datetime(raw)
    except ValueError:
        return None


def read_repo_knowledge_meta(db_path: pathlib.Path) -> Dict[str, str]:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
        return {str(r["key"]): str(r["value"]) for r in rows}
    except Exception:
        return {}
    finally:
        conn.close()


def git_read_stdout(root: pathlib.Path, args: List[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


GIT_STATUS_IGNORE_PREFIXES = (
    ".codex_knowledge",
    ".codex_mem",
)


def _is_ignored_git_status_path(path: str) -> bool:
    raw = (path or "").strip()
    if not raw:
        return False
    # Normalize leading "./" so comparisons are stable.
    while raw.startswith("./"):
        raw = raw[2:]
    for prefix in GIT_STATUS_IGNORE_PREFIXES:
        if raw == prefix or raw.startswith(prefix + "/"):
            return True
    return False


def git_status_porcelain_filtered(root: pathlib.Path) -> str:
    """
    Return `git status --porcelain` output, filtered to ignore codex-mem generated dirs.

    Important: we keep the raw line endings stable so the hash matches what repo_knowledge writes.
    """
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        status = proc.stdout or ""
    except Exception:
        return ""

    if not status:
        return ""

    kept: List[str] = []
    for line in status.splitlines(True):  # keepends=True
        # Porcelain v1 format: XY<space>path
        path_part = line[3:].strip() if len(line) >= 4 else ""
        if not path_part:
            kept.append(line)
            continue

        # Rename/copy format: "old -> new"
        candidates = [p.strip() for p in path_part.split(" -> ")] if " -> " in path_part else [path_part]
        if candidates and all(_is_ignored_git_status_path(p) for p in candidates if p):
            continue
        kept.append(line)
    return "".join(kept)


def repo_knowledge_needs_refresh(root: pathlib.Path, db_path: pathlib.Path) -> Tuple[bool, str, Dict[str, str]]:
    """
    Determine whether repo_knowledge index likely needs a rebuild.

    We optimize for correctness on cold start and after upgrades:
    - if git HEAD changed since index build (committed changes) -> rebuild
    - if git working tree status changed since last index build -> rebuild
    - if index meta lacks git_head (older index schema) -> rebuild once
    """
    if not db_path.exists():
        return True, "missing_index", {}

    meta = read_repo_knowledge_meta(db_path)
    created_at = parse_iso_datetime_maybe(meta.get("created_at_utc", ""))

    if (root / ".git").exists():
        head_now = git_read_stdout(root, ["rev-parse", "HEAD"])
        head_index = (meta.get("git_head", "") or "").strip()
        if head_now and head_index and head_now != head_index:
            return True, "git_head_changed", meta

        if head_now and not head_index:
            return True, "missing_git_head_in_meta", meta

        status = git_status_porcelain_filtered(root)
        if status.strip():
            status_hash_now = hashlib.blake2b(status.encode("utf-8"), digest_size=16).hexdigest()
            status_hash_index = (meta.get("git_status_hash", "") or "").strip()
            if status_hash_index and status_hash_index == status_hash_now:
                return False, "up_to_date", meta
            return True, "git_status_changed", meta

        head_committed_at = parse_iso_datetime_maybe(meta.get("git_head_committed_at", ""))
        if created_at and head_committed_at and head_committed_at > created_at:
            return True, "newer_git_commit_than_index", meta

    return False, "up_to_date", meta


def ensure_repo_knowledge_index(*, root: pathlib.Path, script: pathlib.Path, index_dir: str) -> Dict[str, object]:
    """
    Best-effort index refresh to avoid stale cold-start grounding.
    """
    db_path = (root / index_dir / "repo_knowledge.sqlite3").resolve()
    needs, reason, meta = repo_knowledge_needs_refresh(root, db_path)
    if not needs:
        return {"refreshed": False, "reason": reason, "meta": meta}

    cmd = [
        sys.executable,
        str(script),
        "--root",
        str(root),
        "--index-dir",
        index_dir,
        "index",
        "--all-files",
        "--embedding-provider",
        "local",
        "--ignore-dir",
        ".codex_mem",
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if proc.returncode != 0:
        return {
            "refreshed": False,
            "reason": reason,
            "index_time_ms": round(elapsed_ms, 3),
            "index_exit_code": proc.returncode,
            "index_stderr": trim_snippet(proc.stderr, 600),
        }

    return {
        "refreshed": True,
        "reason": reason,
        "index_time_ms": round(elapsed_ms, 3),
    }


def iso_day_range(anchor: dt.datetime) -> Tuple[str, str]:
    local = anchor.astimezone()
    start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + dt.timedelta(days=1, seconds=-1)
    return (
        start_local.astimezone(dt.timezone.utc).isoformat(),
        end_local.astimezone(dt.timezone.utc).isoformat(),
    )


def parse_natural_query(query: str, now: dt.datetime | None = None) -> Dict[str, object]:
    """
    Parse lightweight natural-language filters from query text.
    Supports relative time phrases and intent hints such as bugfix/release/test.
    """
    src = query.strip()
    current = (now or dt.datetime.now(dt.timezone.utc)).astimezone()
    lowered = src.lower()
    since: str | None = None
    until: str | None = None
    normalized = src

    def cut(pattern: str) -> bool:
        nonlocal normalized
        m = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not m:
            return False
        normalized = (normalized[: m.start()] + " " + normalized[m.end() :]).strip()
        return True

    if cut(r"\blast\s+week\b"):
        weekday = current.weekday()
        start_this_week = current - dt.timedelta(days=weekday)
        start_last_week = (start_this_week - dt.timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_last_week = start_last_week + dt.timedelta(days=7, seconds=-1)
        since = start_last_week.astimezone(dt.timezone.utc).isoformat()
        until = end_last_week.astimezone(dt.timezone.utc).isoformat()
    elif cut(r"\bthis\s+week\b"):
        weekday = current.weekday()
        start = (current - dt.timedelta(days=weekday)).replace(hour=0, minute=0, second=0, microsecond=0)
        since = start.astimezone(dt.timezone.utc).isoformat()
        until = current.astimezone(dt.timezone.utc).isoformat()
    elif cut(r"\byesterday\b"):
        anchor = current - dt.timedelta(days=1)
        since, until = iso_day_range(anchor)
    elif cut(r"\btoday\b"):
        since, until = iso_day_range(current)
    else:
        m_days = re.search(r"\blast\s+(\d+)\s+days?\b", lowered)
        if m_days:
            n = max(1, min(365, int(m_days.group(1))))
            since_dt = current - dt.timedelta(days=n)
            since = since_dt.astimezone(dt.timezone.utc).isoformat()
            until = current.astimezone(dt.timezone.utc).isoformat()
            cut(r"\blast\s+\d+\s+days?\b")

    intent = "general"
    intent_keywords: List[str] = []
    if any(k in lowered for k in ("bug", "fix", "regression", "incident", "hotfix")):
        intent = "bugfix"
        intent_keywords = ["bug", "fix", "regression", "incident", "hotfix", "error"]
    elif any(k in lowered for k in ("release", "launch", "ship")):
        intent = "release"
        intent_keywords = ["release", "launch", "ship"]
    elif any(k in lowered for k in ("test", "coverage", "ci")):
        intent = "test"
        intent_keywords = ["test", "coverage", "ci"]
    elif any(k in lowered for k in ("refactor", "cleanup", "architecture")):
        intent = "refactor"
        intent_keywords = ["refactor", "cleanup", "architecture"]

    normalized_query = re.sub(r"\s+", " ", normalized).strip()
    if not normalized_query:
        normalized_query = src
    return {
        "raw_query": src,
        "normalized_query": normalized_query,
        "since": since,
        "until": until,
        "intent": intent,
        "intent_keywords": intent_keywords,
    }


def redact_sensitive_text(text: str) -> str:
    out = text
    for pattern, replacement in DEFAULT_REDACTION_RULES:
        out = pattern.sub(replacement, out)
    return out


def anonymize_text_for_share(text: str) -> str:
    out = redact_sensitive_text(text or "")
    out = re.sub(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", "<EMAIL>", out)
    out = re.sub(r"(?:[A-Za-z]:\\|/)[^\s\"']+", "/ABS/PATH", out)
    out = re.sub(r"\b(?:github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]+)\b", "<REDACTED_TOKEN>", out)
    out = re.sub(r"\b(?:sk-[A-Za-z0-9]{10,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|AIza[0-9A-Za-z\-_]{10,})\b", "<REDACTED_TOKEN>", out)
    out = re.sub(r"\b[a-f0-9]{32,}\b", "<REDACTED_HEX>", out, flags=re.IGNORECASE)
    return out


def scrub_json_for_share(value: object) -> object:
    if isinstance(value, dict):
        out: Dict[str, object] = {}
        for key, val in value.items():
            lower = str(key).lower()
            if any(tok in lower for tok in ("token", "secret", "password", "api_key", "apikey", "authorization")):
                out[str(key)] = "<REDACTED>"
            else:
                out[str(key)] = scrub_json_for_share(val)
        return out
    if isinstance(value, list):
        return [scrub_json_for_share(v) for v in value]
    if isinstance(value, str):
        return anonymize_text_for_share(value)
    return value


def db_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for table in ("sessions", "events", "observations"):
        try:
            out[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        except Exception:
            out[table] = 0
    return out


def read_file_snippet(path: pathlib.Path, *, max_bytes: int = 64 * 1024) -> str:
    """
    Read a privacy-safer snippet for seeding.
    - decodes as utf-8 with replacement
    - for large files: head+tail
    """
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if not data:
        return ""
    if len(data) <= max_bytes:
        return data.decode("utf-8", errors="replace")
    half = max_bytes // 2
    head = data[:half].decode("utf-8", errors="replace")
    tail = data[-half:].decode("utf-8", errors="replace")
    return f"{head}\n...\n{tail}"


def describe_repo_root(root: pathlib.Path, *, max_entries: int = 120) -> str:
    lines: List[str] = []
    lines.append("Repo root snapshot (auto-generated):")
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        entries = []
    for p in entries:
        name = p.name
        if not name or name.startswith("."):
            continue
        if name in {DEFAULT_INDEX_DIR}:
            continue
        kind = "dir" if p.is_dir() else "file"
        lines.append(f"- {kind}: {name}")
        if len(lines) >= max_entries:
            lines.append("- ...<trimmed>")
            break
    return "\n".join(lines).strip()


def repo_seed_files(root: pathlib.Path) -> List[pathlib.Path]:
    # Minimal, high-signal, usually safe docs/manifests.
    candidates = [
        "README.md",
        "README.MD",
        "README",
        "README.txt",
        "README.rst",
        "AGENTS.md",
        "pyproject.toml",
        "package.json",
        "go.mod",
        "Cargo.toml",
        "Package.swift",
        "Podfile",
        "Gemfile",
        "requirements.txt",
    ]
    out: List[pathlib.Path] = []
    # Deduplicate case-insensitive aliases (macOS) and hardlinks.
    seen_inodes: set[tuple[int, int]] = set()
    seen_paths: set[str] = set()
    for name in candidates:
        p = root / name
        if p.exists() and p.is_file():
            try:
                st = p.stat()
                inode_key = (int(st.st_dev), int(st.st_ino))
            except OSError:
                inode_key = None
            if inode_key is not None:
                if inode_key in seen_inodes:
                    continue
                seen_inodes.add(inode_key)
            path_key = str(p)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            out.append(p)
    return out[:6]


def seed_repo_baseline(conn: sqlite3.Connection, root: pathlib.Path, project: str, trigger_query: str) -> bool:
    """
    When the DB is empty, seed a minimal baseline so Stage-1 search can return IDs.
    This avoids dead-ends where search/timeline/get cannot proceed.
    """
    counts = db_counts(conn)
    if counts.get("events", 0) > 0 or counts.get("observations", 0) > 0:
        return False

    seed_session_id = f"seed-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ensure_session(
        conn,
        session_id=seed_session_id,
        project=project,
        title="Repo baseline seed (auto)",
        metadata={"hook": "AutoSeed"},
    )

    # Add bilingual keywords to improve cold-start hit rate for common onboarding queries.
    keyword_header = (
        "Keywords: 学习这个项目 项目 架构 模块 入口 主流程 风险 learn this project project overview architecture modules entrypoint flow risks"
    )

    seed_event_ids: List[int] = []

    root_snapshot = anonymize_text_for_share(describe_repo_root(root))
    root_snapshot, compact_meta = compact_tool_output(f"{keyword_header}\n\n{root_snapshot}", 2000)
    seed_event_ids.append(
        insert_event(
            conn,
            session_id=seed_session_id,
            project=project,
            event_kind="repo_seed",
            role="tool",
            title="Seed: repo root snapshot",
            content=root_snapshot,
            tool_name="repo_seed",
            file_path=None,
            tags=["seed", "repo", "learn", "architecture"],
            metadata={"hook": "AutoSeed", "trigger_query": trigger_query, "compaction": compact_meta},
        )
    )

    seeded_files = repo_seed_files(root)
    for p in seeded_files:
        snippet = anonymize_text_for_share(read_file_snippet(p, max_bytes=64 * 1024))
        snippet, compact_meta = compact_tool_output(f"{keyword_header}\n\nFile: {p.name}\n\n{snippet}", 3000)
        seed_event_ids.append(
            insert_event(
                conn,
                session_id=seed_session_id,
                project=project,
                event_kind="repo_seed",
                role="tool",
                title=f"Seed: {p.name}",
                content=snippet,
                tool_name="repo_seed",
                file_path=str(p.relative_to(root)),
                tags=["seed", "repo", "doc"],
                metadata={"hook": "AutoSeed", "trigger_query": trigger_query, "compaction": compact_meta},
            )
        )

    seeded_sources = [f"- {p.name}" for p in seeded_files] if seeded_files else ["- (none)"]
    obs_body_lines = [
        "Auto-generated baseline seeded because the memory database was empty.",
        "",
        f"Trigger query: {anonymize_text_for_share(trigger_query)}",
        "",
        "Seeded sources:",
        *seeded_sources,
        "",
        "What this enables:",
        "- Stage 1 search can return IDs immediately",
        "- Stage 2 timeline / Stage 3 get-observations can proceed without dead-ends",
        "",
        "Next step to build real memory:",
        "- start a real session: session-start -> post-tool-use -> session-end",
    ]
    insert_observation(
        conn,
        session_id=seed_session_id,
        project=project,
        observation_type="repo_seed",
        title=f"Repo baseline (auto) ({seed_session_id})",
        body="\n".join(obs_body_lines).strip(),
        source_event_ids=seed_event_ids[:30],
        metadata={"auto_generated": 1, "hook": "AutoSeed"},
    )

    conn.execute(
        "UPDATE sessions SET ended_at = ?, status = 'ended' WHERE session_id = ?",
        (now_iso(), seed_session_id),
    )
    conn.commit()
    return True


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
    since: str | None,
    until: str | None,
    include_private: bool,
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
    if since:
        clauses.append("e.created_at >= ?")
        params.append(since)
    if until:
        clauses.append("e.created_at <= ?")
        params.append(until)
    if not include_private:
        clauses.append("COALESCE(json_extract(e.metadata_json, '$.privacy.visibility'), 'public') != 'private'")
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
    since: str | None,
    until: str | None,
    include_private: bool,
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
    if since:
        clauses.append("o.created_at >= ?")
        params.append(since)
    if until:
        clauses.append("o.created_at <= ?")
        params.append(until)
    if not include_private:
        clauses.append("COALESCE(json_extract(o.metadata_json, '$.privacy.visibility'), 'public') != 'private'")
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
    since: str | None,
    until: str | None,
    include_private: bool,
    limit: int,
    vector_dim: int,
    alpha: float,
) -> List[SearchResult]:
    event_rows = search_events(
        conn,
        query=query,
        project=project,
        session_id=session_id,
        since=since,
        until=until,
        include_private=include_private,
        limit=max(10, limit * 3),
    )
    obs_rows = search_observations(
        conn,
        query=query,
        project=project,
        session_id=session_id,
        since=since,
        until=until,
        include_private=include_private,
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


def filter_results_by_intent(
    conn: sqlite3.Connection,
    results: Sequence[SearchResult],
    *,
    intent_keywords: Sequence[str],
    snippet_chars: int,
    project: str | None,
    include_private: bool,
) -> List[SearchResult]:
    if not intent_keywords:
        return list(results)
    kept: List[SearchResult] = []
    lower_kw = [k.lower() for k in intent_keywords if k.strip()]
    for item in results:
        typ, iid = parse_item_id(item.item_id)
        try:
            detail = get_item_detail(
                conn,
                typ,
                iid,
                snippet_chars=snippet_chars,
                project=project,
                include_private=include_private,
            )
        except ValueError:
            continue
        text = f"{detail.get('title', '')}\n{detail.get('content', '')}".lower()
        if any(k in text for k in lower_kw):
            kept.append(item)
    return kept


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
    project: str | None,
    before: int,
    after: int,
    snippet_chars: int,
    include_private: bool = False,
) -> Dict[str, object]:
    anchor = conn.execute(
        """
        SELECT id, session_id, project, event_kind, title, content, tool_name, file_path, created_at, metadata_json
        FROM events WHERE id = ?
        """,
        (event_id,),
    ).fetchone()
    if not anchor:
        raise ValueError(f"event not found: E{event_id}")
    if project and str(anchor["project"]) != str(project):
        raise ValueError(f"event not in project '{project}': E{event_id}")
    anchor_meta = json.loads(anchor["metadata_json"]) if anchor["metadata_json"] else {}
    anchor_visibility = str(((anchor_meta.get("privacy") or {}).get("visibility", "public"))).lower()
    if anchor_visibility == "private" and not include_private:
        raise ValueError(f"event is private: E{event_id}")

    session_id = str(anchor["session_id"])
    created_at = str(anchor["created_at"])
    private_clause = ""
    if not include_private:
        private_clause = " AND COALESCE(json_extract(metadata_json, '$.privacy.visibility'), 'public') != 'private'"
    before_rows = conn.execute(
        """
        SELECT id, event_kind, title, content, created_at
        FROM events
        WHERE session_id = ? AND created_at < ?""" + private_clause + """
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (session_id, created_at, max(0, before)),
    ).fetchall()
    after_rows = conn.execute(
        """
        SELECT id, event_kind, title, content, created_at
        FROM events
        WHERE session_id = ? AND created_at > ?""" + private_clause + """
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
    project: str | None,
    before: int,
    after: int,
    snippet_chars: int,
    include_private: bool = False,
) -> Dict[str, object]:
    obs = conn.execute(
        """
        SELECT id, session_id, project, observation_type, title, body, source_event_ids_json, created_at, metadata_json
        FROM observations WHERE id = ?
        """,
        (obs_id,),
    ).fetchone()
    if not obs:
        raise ValueError(f"observation not found: O{obs_id}")
    if project and str(obs["project"]) != str(project):
        raise ValueError(f"observation not in project '{project}': O{obs_id}")
    obs_meta = json.loads(obs["metadata_json"]) if obs["metadata_json"] else {}
    obs_visibility = str(((obs_meta.get("privacy") or {}).get("visibility", "public"))).lower()
    if obs_visibility == "private" and not include_private:
        raise ValueError(f"observation is private: O{obs_id}")
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
            project=project,
            before=before,
            after=after,
            snippet_chars=snippet_chars,
            include_private=include_private,
        )
        timeline["before"] = linked["before"]
        timeline["after"] = linked["after"]
    return timeline


def get_item_detail(
    conn: sqlite3.Connection,
    item_type: str,
    item_id: int,
    snippet_chars: int,
    project: str | None = None,
    include_private: bool = False,
) -> Dict[str, object]:
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
        if project and str(row["project"]) != str(project):
            raise ValueError(f"event not in project '{project}': E{item_id}")
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        visibility = str(((metadata.get("privacy") or {}).get("visibility", "public"))).lower()
        if visibility == "private" and not include_private:
            raise ValueError(f"event is private: E{item_id}")
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
            "metadata": metadata,
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
    if project and str(row["project"]) != str(project):
        raise ValueError(f"observation not in project '{project}': O{item_id}")
    metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    visibility = str(((metadata.get("privacy") or {}).get("visibility", "public"))).lower()
    if visibility == "private" and not include_private:
        raise ValueError(f"observation is private: O{item_id}")
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
        "metadata": metadata,
        "created_at": str(row["created_at"]),
        "token_estimate": estimate_tokens(body),
    }


def summarize_session(conn: sqlite3.Connection, session_id: str) -> Dict[str, object]:
    rows = conn.execute(
        """
        SELECT id, event_kind, role, title, content, tool_name, created_at, metadata_json
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
    visible_rows = []
    for row in rows:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        visibility = str(((metadata.get("privacy") or {}).get("visibility", "public"))).lower()
        if visibility == "private":
            continue
        visible_rows.append(row)

    source_ids = [int(row["id"]) for row in visible_rows]

    for row in visible_rows:
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

    if not investigation and visible_rows:
        investigation = [str(row["title"]) for row in visible_rows if str(row["title"]).strip()][:5]
    if not learnings:
        learnings = [
            trim_snippet(str(row["content"]), 120) for row in visible_rows if str(row["content"]).strip()
        ][:3]
    if not completed:
        completed = [item for item in investigation[:3]]
    if not next_steps:
        next_steps = ["Continue with the next task and validate regression risks first."]

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
        "event_count": len(visible_rows),
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

    index_refresh = ensure_repo_knowledge_index(root=root, script=script, index_dir=index_dir)
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
            "index_refresh": index_refresh,
        }
    try:
        payload = json.loads(proc.stdout)
        if isinstance(payload, dict):
            payload["index_refresh"] = index_refresh
        return payload
    except json.JSONDecodeError:
        return {"warning": "invalid json from repo_knowledge query", "index_refresh": index_refresh}


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
    runtime_cfg = get_runtime_config(conn)
    semantic_tags = ["tool", args.tool_name.strip().lower()]
    if args.tag:
        semantic_tags.extend(args.tag)
    semantic_tags_clean = [t.strip().lower() for t in semantic_tags if t and t.strip()]

    privacy_tags = [t.strip().lower() for t in (args.privacy_tag or []) if t and t.strip()]
    if any(tag in PRIVACY_BLOCK_TAGS for tag in privacy_tags):
        print(
            json.dumps(
                {
                    "ok": True,
                    "hook": "PostToolUse",
                    "skipped": True,
                    "reason": "blocked_by_privacy_tag",
                    "blocked_tags": [t for t in privacy_tags if t in PRIVACY_BLOCK_TAGS],
                },
                ensure_ascii=False,
            )
        )
        return 0

    content = args.content
    compact_meta: Dict[str, int] = {"raw_chars": len(content), "final_chars": len(content), "compacted": 0}
    auto_compact = bool(runtime_cfg.get("channel") == "beta" and runtime_cfg.get("beta_endless_mode"))
    compact_enabled = bool(args.compact or auto_compact)
    compact_chars = args.compact_chars if args.compact else min(args.compact_chars, 2000)
    if compact_enabled:
        content, compact_meta = compact_tool_output(content, compact_chars)

    redacted = False
    if any(tag in PRIVACY_REDACT_TAGS for tag in privacy_tags):
        content = redact_sensitive_text(content)
        redacted = True

    visibility = "private" if any(tag in PRIVACY_PRIVATE_TAGS for tag in privacy_tags) else "public"
    privacy_meta = {
        "tags": privacy_tags,
        "visibility": visibility,
        "redacted": redacted,
    }

    meta: Dict[str, object] = {
        "hook": "PostToolUse",
        "exit_code": args.exit_code,
        "compaction": compact_meta,
        "auto_compaction": bool(auto_compact and not args.compact),
        "privacy": privacy_meta,
    }
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
        tags=semantic_tags_clean,
        metadata=meta,
    )
    conn.commit()
    payload = {
        "ok": True,
        "event_id": f"E{event_id}",
        "hook": "PostToolUse",
        "compaction": compact_meta,
        "auto_compaction": bool(auto_compact and not args.compact),
        "privacy": privacy_meta,
    }
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


def cmd_config_get(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    payload = {
        "ok": True,
        "config": get_runtime_config(conn),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_config_set(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    beta_endless_mode: bool | None = None
    if args.beta_endless_mode is not None:
        beta_endless_mode = args.beta_endless_mode == "on"
    updated = set_runtime_config(
        conn,
        channel=args.channel,
        viewer_refresh_sec=args.viewer_refresh_sec,
        beta_endless_mode=beta_endless_mode,
    )
    print(json.dumps({"ok": True, "config": updated}, ensure_ascii=False, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    meta = fetch_meta(conn)
    vector_dim = int(meta.get("vector_dim", str(DEFAULT_VECTOR_DIM)))
    since = args.since
    until = args.until
    if since:
        since = parse_iso_datetime(since).isoformat()
    if until:
        until = parse_iso_datetime(until).isoformat()
    auto_seeded = seed_repo_baseline(conn, root, args.project, trigger_query=args.query)
    results = blended_search(
        conn,
        query=args.query,
        project=args.project,
        session_id=args.session_id,
        since=since,
        until=until,
        include_private=bool(args.include_private),
        limit=args.limit,
        vector_dim=vector_dim,
        alpha=args.alpha,
    )
    payload = {
        "query": args.query,
        "stage": "search",
        "filters": {
            "project": args.project,
            "session_id": args.session_id,
            "since": since,
            "until": until,
            "include_private": bool(args.include_private),
        },
        "auto_seeded": bool(auto_seeded),
        "db_counts": db_counts(conn),
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


def cmd_nl_search(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    meta = fetch_meta(conn)
    vector_dim = int(meta.get("vector_dim", str(DEFAULT_VECTOR_DIM)))
    auto_seeded = seed_repo_baseline(conn, root, args.project, trigger_query=args.query)

    parsed = parse_natural_query(args.query)
    since = parsed["since"]
    until = parsed["until"]
    if args.since:
        since = parse_iso_datetime(args.since).isoformat()
    if args.until:
        until = parse_iso_datetime(args.until).isoformat()

    raw_results = blended_search(
        conn,
        query=str(parsed["normalized_query"]),
        project=args.project,
        session_id=args.session_id,
        since=since,
        until=until,
        include_private=bool(args.include_private),
        limit=max(10, args.limit * 2),
        vector_dim=vector_dim,
        alpha=args.alpha,
    )
    filtered = filter_results_by_intent(
        conn,
        raw_results,
        intent_keywords=parsed["intent_keywords"],
        snippet_chars=args.snippet_chars,
        project=args.project,
        include_private=bool(args.include_private),
    )
    results = filtered[: args.limit] if filtered else raw_results[: args.limit]

    payload = {
        "query": args.query,
        "stage": "nl-search",
        "interpreted": parsed,
        "filters": {
            "project": args.project,
            "session_id": args.session_id,
            "since": since,
            "until": until,
            "include_private": bool(args.include_private),
        },
        "auto_seeded": bool(auto_seeded),
        "db_counts": db_counts(conn),
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
    try:
        if item_type == "event":
            payload = timeline_for_event(
                conn,
                event_id=item_id,
                project=args.project,
                before=args.before,
                after=args.after,
                snippet_chars=args.snippet_chars,
                include_private=bool(args.include_private),
            )
        else:
            payload = timeline_for_observation(
                conn,
                obs_id=item_id,
                project=args.project,
                before=args.before,
                after=args.after,
                snippet_chars=args.snippet_chars,
                include_private=bool(args.include_private),
            )
    except ValueError as exc:
        print(json.dumps({"stage": "timeline", "error": str(exc), "id": args.id}, ensure_ascii=False, indent=2))
        return 1
    payload["stage"] = "timeline"
    payload["filters"] = {"project": args.project}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_get_observations(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    details: List[Dict[str, object]] = []
    skipped_ids: List[str] = []
    for token in args.ids:
        item_type, item_id = parse_item_id(token)
        try:
            detail = get_item_detail(
                conn,
                item_type,
                item_id,
                args.snippet_chars,
                project=args.project,
                include_private=bool(args.include_private),
            )
        except ValueError:
            skipped_ids.append(token)
            continue
        if args.compact:
            detail["content"] = detail["snippet"]
            detail["token_estimate"] = estimate_tokens(str(detail["snippet"]))
        details.append(detail)
    payload = {
        "stage": "get_observations",
        "filters": {"project": args.project},
        "count": len(details),
        "skipped_ids": skipped_ids,
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
    lines.append("System: You are a Codex coding assistant. Prioritize the provided Memory + Repo contexts.")
    lines.append("If evidence is insufficient, explicitly state what is missing.")
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


_REPO_CATEGORY_ORDER = ("entrypoint", "persistence", "ai_generation", "backend", "code")


def _match_any(text: str, terms: Sequence[str]) -> bool:
    if not text:
        return False
    return any(term in text for term in terms)


def infer_repo_categories(
    path: str,
    *,
    symbol_hint: str = "",
    snippet: str = "",
    existing_category: str = "",
) -> List[str]:
    lower_path = (path or "").lower()
    lower_symbol = (symbol_hint or "").lower()
    lower_snippet = (snippet or "").lower()
    allow_snippet_signals = not lower_path.endswith((".md", ".markdown", ".rst", ".txt"))

    categories = set()
    existing = str(existing_category or "").strip().lower()
    if existing in {"entrypoint", "persistence", "ai_generation", "backend"}:
        categories.add(existing)

    if _match_any(
        lower_path,
        (
            "app.swift",
            "main.swift",
            "appdelegate",
            "scenedelegate",
            "entrypoint",
            "startup",
            "bootstrap",
            "/main.py",
            "/cli.py",
        ),
    ) or _match_any(
        lower_symbol,
        ("main", "run_cli", "parse_args", "build_parser", "appdelegate", "scenedelegate", "bootstrap", "entrypoint"),
    ) or (
        allow_snippet_signals
        and _match_any(
        lower_snippet,
        (
            "entrypoint",
            "startup",
            "bootstrap",
            "if __name__ == \"__main__\"",
            "argparse",
            "sub.add_parser",
            "appdelegate",
            "scenedelegate",
        ),
        )
    ):
        categories.add("entrypoint")

    if _match_any(
        lower_path,
        ("database", "bootstrapper", "swiftdata", "coredata", "modelcontext", "sqlite", "migration", "store", "persistence"),
    ) or _match_any(
        lower_symbol,
        ("save", "load", "migrate", "bootstrap", "modelcontext", "sqlite", "storage", "persistence"),
    ) or (
        allow_snippet_signals
        and _match_any(
        lower_snippet,
        (
            "persistence",
            "database",
            "storage",
            "sqlite",
            "swiftdata",
            "coredata",
            "modelcontext",
            "migration",
            "save(",
            "commit(",
        ),
        )
    ):
        categories.add("persistence")

    if _match_any(
        lower_path,
        ("generation", "stream", "organize", "prism", "ai_", "/ai/", "llm", "model"),
    ) or _match_any(
        lower_symbol,
        ("generate", "generation", "stream", "organize", "prism", "ai", "model"),
    ) or (
        allow_snippet_signals
        and _match_any(
        lower_snippet,
        (
            "ai generation",
            "generation flow",
            "streaming",
            "prism",
            "organize",
            "model output",
            "prompt plan",
            "token budget",
        ),
        )
    ):
        categories.add("ai_generation")

    if _match_any(
        lower_path,
        ("backend", "/routes/", "api.ts", "index.ts", "server.ts", "/api/", "controller", "handler"),
    ) or _match_any(
        lower_symbol,
        ("route", "handler", "controller", "server", "api"),
    ) or (
        allow_snippet_signals
        and _match_any(
        lower_snippet,
        ("http", "route", "endpoint", "request", "response", "api", "server"),
        )
    ):
        categories.add("backend")

    categories.add("code")
    return [cat for cat in _REPO_CATEGORY_ORDER if cat in categories]


def infer_repo_category(path: str, *, symbol_hint: str = "", snippet: str = "", existing_category: str = "") -> str:
    categories = infer_repo_categories(
        path,
        symbol_hint=symbol_hint,
        snippet=snippet,
        existing_category=existing_category,
    )
    for cat in categories:
        if cat != "code":
            return cat
    return "code"


def _chunk_category_set(chunk: Mapping[str, object]) -> List[str]:
    raw = chunk.get("categories")
    if isinstance(raw, list):
        normalized = [str(v).strip() for v in raw if str(v).strip()]
        if normalized:
            uniq = []
            seen = set()
            for value in normalized:
                if value in seen:
                    continue
                seen.add(value)
                uniq.append(value)
            return uniq
    cat = str(chunk.get("category", "code")).strip() or "code"
    return [cat]


def _present_repo_categories(chunks: Sequence[Mapping[str, object]]) -> List[str]:
    present = set()
    for item in chunks:
        for cat in _chunk_category_set(item):
            present.add(cat)
    return sorted(present)


def _extract_repo_chunks(payload: Mapping[str, object] | object) -> List[Dict[str, object]]:
    if not isinstance(payload, Mapping):
        return []
    raw = payload.get("chunks")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, object]] = []
    for chunk in raw:
        if not isinstance(chunk, Mapping):
            continue
        row = dict(chunk)
        path = str(row.get("path", ""))
        symbol_hint = str(row.get("symbol_hint", ""))
        snippet = str(row.get("snippet", "") or row.get("text", ""))
        existing_category = str(row.get("category", ""))
        categories = infer_repo_categories(
            path,
            symbol_hint=symbol_hint,
            snippet=snippet,
            existing_category=existing_category,
        )
        row["categories"] = categories
        row["category"] = infer_repo_category(
            path,
            symbol_hint=symbol_hint,
            snippet=snippet,
            existing_category=existing_category,
        )
        out.append(row)
    return out


def _merge_repo_chunks(base: Sequence[Mapping[str, object]], incoming: Sequence[Mapping[str, object]], limit: int) -> List[Dict[str, object]]:
    merged: List[Dict[str, object]] = []
    seen = set()
    for item in list(base) + list(incoming):
        path = str(item.get("path", "")).strip()
        start = int(item.get("start_line", 0) or 0)
        end = int(item.get("end_line", 0) or 0)
        key = (path, start, end)
        if not path or key in seen:
            continue
        seen.add(key)
        row = dict(item)
        symbol_hint = str(row.get("symbol_hint", ""))
        snippet = str(row.get("snippet", "") or row.get("text", ""))
        existing_category = str(row.get("category", ""))
        categories = infer_repo_categories(
            path,
            symbol_hint=symbol_hint,
            snippet=snippet,
            existing_category=existing_category,
        )
        row["categories"] = categories
        row["category"] = infer_repo_category(
            path,
            symbol_hint=symbol_hint,
            snippet=snippet,
            existing_category=existing_category,
        )
        merged.append(row)
        if len(merged) >= max(1, limit):
            break
    return merged


def ensure_repo_coverage(
    *,
    root: pathlib.Path,
    question: str,
    profile_name: str,
    repo_payload: Mapping[str, object] | object,
    code_top_k: int,
    code_module_limit: int,
    snippet_chars: int,
    repo_index_dir: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    payload = dict(repo_payload) if isinstance(repo_payload, Mapping) else {}
    chunks = _extract_repo_chunks(payload)
    payload["chunks"] = chunks

    required: List[str] = []
    if profile_name == "onboarding":
        required = ["entrypoint", "persistence", "ai_generation"]

    present = _present_repo_categories(chunks)
    missing = [cat for cat in required if cat not in present]
    second_pass_runs: List[Dict[str, object]] = []

    probe_queries = {
        "entrypoint": "entrypoint startup bootstrap main app appdelegate scenedelegate",
        "persistence": "persistence database storage modelcontext swiftdata coredata sqlite migration save",
        "ai_generation": "ai generation note prism organize streaming delta",
    }

    if missing:
        extra_chunks: List[Dict[str, object]] = []
        for cat in missing:
            probe_q = probe_queries.get(cat)
            if not probe_q:
                continue
            probe_payload = run_repo_query(
                root,
                question=f"{question}\n\nCoverage probe: {probe_q}",
                top_k=max(4, min(12, code_top_k)),
                module_limit=max(4, min(12, code_module_limit)),
                snippet_chars=snippet_chars,
                index_dir=repo_index_dir,
            )
            probe_chunks = _extract_repo_chunks(probe_payload)
            extra_chunks.extend(probe_chunks)
            second_pass_runs.append(
                {
                    "category": cat,
                    "query": probe_q,
                    "chunk_count": len(probe_chunks),
                    "warning": probe_payload.get("warning") if isinstance(probe_payload, Mapping) else None,
                }
            )
        chunks = _merge_repo_chunks(chunks, extra_chunks, max(8, code_top_k + len(extra_chunks)))
        payload["chunks"] = chunks

    present = _present_repo_categories(chunks)
    missing = [cat for cat in required if cat not in present]
    gate = {
        "profile": profile_name,
        "required_categories": required,
        "present_categories": present,
        "missing_categories": missing,
        "pass": len(missing) == 0,
        "second_pass_runs": second_pass_runs,
    }
    return payload, gate


def _profile_prompt_template(profile_name: str) -> Tuple[str, str]:
    profile = (profile_name or "").strip().lower()
    if profile == "onboarding":
        zh = (
            "学习这个项目：北极星、架构、模块地图、入口、主流程、持久化、AI 生成链路、风险。"
        )
        en = (
            "learn this project: north star, architecture, module map, entrypoint, main flow, "
            "persistence, ai generation, risks"
        )
        return zh, en
    if profile == "bug_triage":
        zh = "排查这个回归：复现路径、根因链路、最小风险修复方案、验证清单。"
        en = "triage this regression: repro path, root cause chain, minimal-risk fix, verification checklist"
        return zh, en
    if profile == "implementation":
        zh = "实现这个需求：最小改动、兼容性边界、验证步骤、风险点。"
        en = "implement this task: minimal patch, compatibility boundaries, validation steps, risks"
        return zh, en
    zh = "回答这个问题：先给结论，再给证据路径和关键命令结果。"
    en = "answer this question: conclusion first, then evidence paths and key command outputs"
    return zh, en


def build_forced_next_input(
    *,
    root: pathlib.Path,
    profile_name: str,
    coverage_gate: Mapping[str, object],
) -> Dict[str, object]:
    script_path = pathlib.Path(__file__).resolve()
    shell_path = script_path.with_suffix(".sh")
    root_abs = str(root.resolve())
    zh_prompt, en_prompt = _profile_prompt_template(profile_name)
    required_fields = ["mapping_decision", "coverage_gate", "prompt_plan", "prompt_metrics"]

    cmd_zh_one_click = (
        f'bash {shell_path} run-target "/ABS/PATH/TO/OTHER_PROJECT" --project my-project --question "{zh_prompt}"'
    )
    cmd_en_one_click = (
        f'bash {shell_path} run-target "/ABS/PATH/TO/OTHER_PROJECT" --project my-project --question "{en_prompt}"'
    )
    cmd_zh_py = (
        f'python3 {script_path} --root "/ABS/PATH/TO/OTHER_PROJECT" ask "{zh_prompt}" '
        '--project my-project --mapping-debug'
    )
    cmd_en_py = (
        f'python3 {script_path} --root "/ABS/PATH/TO/OTHER_PROJECT" ask "{en_prompt}" '
        '--project my-project --mapping-debug'
    )

    out: Dict[str, object] = {
        "mandatory": True,
        "why": "Always provide the next executable input for another project; do not return generic guidance only.",
        "required_output_fields": required_fields,
        "current_runtime_root": root_abs,
        "next_input": {
            "for_other_project_root": "/ABS/PATH/TO/OTHER_PROJECT",
            "command_template_zh": cmd_zh_one_click,
            "command_template_en": cmd_en_one_click,
            "command_template_py_zh": cmd_zh_py,
            "command_template_py_en": cmd_en_py,
            "prompt_template_zh": zh_prompt,
            "prompt_template_en": en_prompt,
            "preferred_entrypoint": "run-target",
        },
        "acceptance_gate": {
            "onboarding_requires": ["mapping_decision.profile=onboarding", "coverage_gate.pass=true"],
            "fallback_requires": ["mapping_decision.confidence>=0.55"],
        },
    }

    gate_pass = bool(coverage_gate.get("pass", True))
    missing = coverage_gate.get("missing_categories", [])
    missing_text = ", ".join(str(v) for v in missing) if isinstance(missing, list) and missing else ""
    if missing_text:
        refine_suffix = f"\n\n只补齐这些缺失证据类别：{missing_text}"
        out["next_input"]["refine_prompt_zh"] = f"{zh_prompt}{refine_suffix}"
        out["next_input"]["refine_command_template_zh"] = (
            f'bash {shell_path} run-target "/ABS/PATH/TO/OTHER_PROJECT" --project my-project '
            f'--question "{zh_prompt}{refine_suffix}"'
        )
        out["next_input"]["refine_command_template_py_zh"] = (
            f'python3 {script_path} --root "/ABS/PATH/TO/OTHER_PROJECT" ask '
            f'"{zh_prompt}{refine_suffix}" --project my-project --mapping-debug'
        )
    out["status"] = "ready" if gate_pass else "needs_refine"
    return out


def cmd_ask(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    meta = fetch_meta(conn)
    vector_dim = int(meta.get("vector_dim", str(DEFAULT_VECTOR_DIM)))

    parsed_nl = parse_natural_query(args.question)
    mapping_decision = map_prompt_to_profile(
        args.question,
        parsed_nl=parsed_nl,
        mapping_fallback=args.mapping_fallback,
        llm_api_key=os.environ.get("OPENAI_API_KEY", ""),
        llm_model=os.environ.get("OPENAI_PROMPT_ROUTER_MODEL", "gpt-4o-mini"),
        llm_timeout_sec=8,
    )
    profile_name = str(mapping_decision.get("profile", "daily_qa"))
    profile = get_prompt_profile(profile_name)
    defaults = profile.defaults

    search_limit = int(args.search_limit if args.search_limit is not None else defaults.get("search_limit", 20))
    detail_limit = int(args.detail_limit if args.detail_limit is not None else defaults.get("detail_limit", 6))
    code_top_k = int(args.code_top_k if args.code_top_k is not None else defaults.get("code_top_k", 8))
    code_module_limit = int(args.code_module_limit if args.code_module_limit is not None else defaults.get("code_module_limit", 6))
    alpha = float(args.alpha if args.alpha is not None else defaults.get("alpha", 0.7))
    search_limit = max(1, search_limit)
    detail_limit = max(1, detail_limit)
    code_top_k = max(1, code_top_k)
    code_module_limit = max(1, code_module_limit)

    auto_seeded = seed_repo_baseline(conn, root, args.project, trigger_query=args.question)
    normalized_query = str(parsed_nl.get("normalized_query", "")).strip() or args.question
    since_raw = parsed_nl.get("since")
    until_raw = parsed_nl.get("until")
    since = str(since_raw).strip() if isinstance(since_raw, str) and str(since_raw).strip() else None
    until = str(until_raw).strip() if isinstance(until_raw, str) and str(until_raw).strip() else None
    layer1 = blended_search(
        conn,
        query=normalized_query,
        project=args.project,
        session_id=args.session_id,
        since=since,
        until=until,
        include_private=bool(args.include_private),
        limit=search_limit,
        vector_dim=vector_dim,
        alpha=alpha,
    )
    selected_ids = [item.item_id for item in layer1[:detail_limit]]
    details = []
    for ident in selected_ids:
        typ, iid = parse_item_id(ident)
        try:
            details.append(
                get_item_detail(
                    conn,
                    typ,
                    iid,
                    args.snippet_chars,
                    project=args.project,
                    include_private=bool(args.include_private),
                )
            )
        except ValueError:
            continue

    repo_payload = run_repo_query(
        root,
        question=args.question,
        top_k=code_top_k,
        module_limit=code_module_limit,
        snippet_chars=args.snippet_chars,
        index_dir=args.repo_index_dir,
    )

    repo_payload, coverage_gate = ensure_repo_coverage(
        root=root,
        question=args.question,
        profile_name=profile.name,
        repo_payload=repo_payload,
        code_top_k=code_top_k,
        code_module_limit=code_module_limit,
        snippet_chars=args.snippet_chars,
        repo_index_dir=args.repo_index_dir,
    )

    memory_l1_tokens = sum(item.token_estimate for item in layer1)
    memory_l3_tokens = sum(int(item["token_estimate"]) for item in details)
    code_tokens = 0
    if isinstance(repo_payload, dict):
        chunks = repo_payload.get("chunks")
        if isinstance(chunks, list):
            code_tokens = sum(estimate_tokens(str(c.get("snippet", ""))) for c in chunks)

    prompt_plan = build_prompt_plan(
        profile=profile,
        question=args.question,
        memory_details=details,
        repo_payload=repo_payload if isinstance(repo_payload, Mapping) else {},
        total_budget=1800,
        snippet_chars=args.snippet_chars,
    )

    if args.prompt_style == "legacy":
        prompt = build_fused_prompt(args.question, details, repo_payload, args.snippet_chars)
    else:
        prompt = render_compact_prompt(
            question=args.question,
            profile=profile,
            mapping_decision=mapping_decision,
            prompt_plan=prompt_plan,
            coverage_gate=coverage_gate,
        )

    prompt_metrics = {
        "style": args.prompt_style,
        "chars": len(prompt),
        "tokens_est": estimate_tokens(prompt),
        "budget_tokens_target": int(prompt_plan.get("budgets", {}).get("total", 0)),
        "budget_tokens_used": int(prompt_plan.get("usage", {}).get("total_tokens_est", 0)),
    }

    mapping_view = dict(mapping_decision)
    if not bool(args.mapping_debug):
        # Keep payload concise by default while preserving the decision trace.
        mapping_view.pop("profile_scores", None)

    payload = {
        "question": args.question,
        "stage": "fused_ask",
        "filters": {
            "project": args.project,
            "session_id": args.session_id,
            "include_private": bool(args.include_private),
        },
        "effective_params": {
            "profile": profile.name,
            "search_limit": search_limit,
            "detail_limit": detail_limit,
            "code_top_k": code_top_k,
            "code_module_limit": code_module_limit,
            "alpha": alpha,
            "normalized_query": normalized_query,
            "since": since,
            "until": until,
        },
        "auto_seeded": bool(auto_seeded),
        "db_counts": db_counts(conn),
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
        "mapping_decision": mapping_view,
        "coverage_gate": coverage_gate,
        "prompt_plan": prompt_plan,
        "prompt_metrics": prompt_metrics,
        "suggested_prompt": prompt,
        "forced_next_input": build_forced_next_input(
            root=root,
            profile_name=profile.name,
            coverage_gate=coverage_gate,
        ),
    }
    if not bool(coverage_gate.get("pass", True)):
        missing = coverage_gate.get("missing_categories", [])
        missing_text = ", ".join(str(v) for v in missing) if isinstance(missing, list) and missing else "unknown"
        payload["recommended_next_action"] = {
            "type": "ask_refine",
            "reason": f"coverage_gate_missing:{missing_text}",
            "note": "Prefer another targeted ask query; avoid looping mem-search/timeline on empty memory.",
            "example_question": f"{args.question}\n\n只补齐这些缺失证据类别：{missing_text}",
        }
    if args.prompt_only:
        print(prompt)
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_export_session(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    session_id = str(args.session_id).strip()
    if not session_id:
        raise ValueError("session_id is required")

    session = conn.execute(
        """
        SELECT session_id, project, title, started_at, ended_at, status, summary_json, metadata_json
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if not session:
        print(json.dumps({"ok": False, "error": f"session not found: {session_id}"}, ensure_ascii=False, indent=2))
        return 1

    anonymize = str(args.anonymize).lower() != "off"
    include_private = bool(args.include_private)
    max_events = max(1, int(args.max_events))
    max_observations = max(1, int(args.max_observations))

    event_private_clause = ""
    obs_private_clause = ""
    if not include_private:
        event_private_clause = " AND COALESCE(json_extract(metadata_json, '$.privacy.visibility'), 'public') != 'private'"
        obs_private_clause = " AND COALESCE(json_extract(metadata_json, '$.privacy.visibility'), 'public') != 'private'"

    events = conn.execute(
        """
        SELECT id, event_kind, role, title, content, tool_name, file_path, tags_json, metadata_json, created_at
        FROM events
        WHERE session_id = ?""" + event_private_clause + """
        ORDER BY created_at ASC, id ASC
        LIMIT ?
        """,
        (session_id, max_events),
    ).fetchall()

    observations = conn.execute(
        """
        SELECT id, observation_type, title, body, source_event_ids_json, metadata_json, created_at
        FROM observations
        WHERE session_id = ?""" + obs_private_clause + """
        ORDER BY created_at ASC, id ASC
        LIMIT ?
        """,
        (session_id, max_observations),
    ).fetchall()

    session_alias = session_id
    if anonymize:
        session_alias = f"session-{hashlib.sha1(session_id.encode('utf-8')).hexdigest()[:10]}"

    event_id_map: Dict[int, int] = {}
    events_out: List[Dict[str, object]] = []
    for i, row in enumerate(events, start=1):
        original_id = int(row["id"])
        event_id_map[original_id] = i
        tags = json.loads(row["tags_json"]) if row["tags_json"] else []
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        title = str(row["title"] or "")
        content = str(row["content"] or "")
        file_path = str(row["file_path"] or "") if row["file_path"] else None
        if anonymize:
            title = anonymize_text_for_share(title)
            content = anonymize_text_for_share(content)
            if file_path:
                file_path = "/ABS/PATH"
            metadata = scrub_json_for_share(metadata)  # type: ignore[assignment]
        events_out.append(
            {
                "export_event_id": i,
                "event_kind": str(row["event_kind"]),
                "role": str(row["role"]),
                "title": title,
                "content": content,
                "tool_name": row["tool_name"],
                "file_path": file_path,
                "tags": tags,
                "metadata": metadata,
                "created_at": str(row["created_at"]),
            }
        )

    observations_out: List[Dict[str, object]] = []
    for j, row in enumerate(observations, start=1):
        source_ids_raw = json.loads(row["source_event_ids_json"]) if row["source_event_ids_json"] else []
        source_ids = []
        for source_id in source_ids_raw:
            sid = int(source_id)
            if sid in event_id_map:
                source_ids.append(event_id_map[sid])
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        title = str(row["title"] or "")
        body = str(row["body"] or "")
        if anonymize:
            title = anonymize_text_for_share(title)
            body = anonymize_text_for_share(body)
            metadata = scrub_json_for_share(metadata)  # type: ignore[assignment]
        observations_out.append(
            {
                "export_observation_id": j,
                "observation_type": str(row["observation_type"]),
                "title": title,
                "body": body,
                "source_export_event_ids": source_ids,
                "metadata": metadata,
                "created_at": str(row["created_at"]),
            }
        )

    session_summary = json.loads(session["summary_json"]) if session["summary_json"] else {}
    session_metadata = json.loads(session["metadata_json"]) if session["metadata_json"] else {}
    title = str(session["title"] or "")
    if anonymize:
        title = anonymize_text_for_share(title)
        session_summary = scrub_json_for_share(session_summary)  # type: ignore[assignment]
        if isinstance(session_summary, dict):
            session_summary["session_id"] = session_alias
        session_metadata = scrub_json_for_share(session_metadata)  # type: ignore[assignment]

    payload = {
        "ok": True,
        "schema": "codex-mem-session-export-v1",
        "exported_at": now_iso(),
        "anonymized": anonymize,
        "include_private": include_private,
        "session": {
            "session_id": session_alias,
            "project": str(session["project"]),
            "title": title,
            "started_at": str(session["started_at"]),
            "ended_at": str(session["ended_at"]) if session["ended_at"] else None,
            "status": str(session["status"]),
            "summary": session_summary,
            "metadata": session_metadata,
        },
        "events": events_out,
        "observations": observations_out,
        "stats": {
            "event_count": len(events_out),
            "observation_count": len(observations_out),
        },
    }

    output_text = json.dumps(payload, ensure_ascii=False, indent=max(0, int(args.indent)))
    if args.output and args.output != "-":
        out_path = pathlib.Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "ok": True,
                    "stage": "export-session",
                    "output": str(out_path),
                    "stats": payload["stats"],
                    "anonymized": anonymize,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(output_text)
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
    p_tool.add_argument(
        "--privacy-tag",
        action="append",
        default=[],
        help="Privacy policy tag(s): private/sensitive/secret/redact/block/no_mem/...",
    )
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

    p_export = sub.add_parser(
        "export-session",
        help="Export one session as a shareable JSON package (supports anonymization).",
    )
    p_export.add_argument("session_id")
    p_export.add_argument("--include-private", action="store_true")
    p_export.add_argument("--anonymize", choices=["on", "off"], default="on")
    p_export.add_argument("--max-events", type=int, default=1000)
    p_export.add_argument("--max-observations", type=int, default=500)
    p_export.add_argument("--output", default="-", help="Output path or '-' for stdout.")
    p_export.add_argument("--indent", type=int, default=2)
    p_export.set_defaults(func=cmd_export_session)

    p_cfg_get = sub.add_parser("config-get", help="Get runtime configuration (stable/beta, viewer settings).")
    p_cfg_get.set_defaults(func=cmd_config_get)

    p_cfg_set = sub.add_parser("config-set", help="Set runtime configuration.")
    p_cfg_set.add_argument("--channel", choices=sorted(CHANNEL_CHOICES), default=None)
    p_cfg_set.add_argument("--viewer-refresh-sec", type=int, default=None)
    p_cfg_set.add_argument(
        "--beta-endless-mode",
        choices=["on", "off"],
        default=None,
        help="Enable/disable beta endless compaction mode.",
    )
    p_cfg_set.set_defaults(func=cmd_config_set)

    p_search = sub.add_parser("search", help="Stage 1 retrieval: compact index hits.")
    add_project_arg(p_search)
    p_search.add_argument("query")
    p_search.add_argument("--session-id", default=None)
    p_search.add_argument("--since", default=None, help="ISO time lower bound, e.g. 2026-02-01T00:00:00+00:00")
    p_search.add_argument("--until", default=None, help="ISO time upper bound")
    p_search.add_argument("--include-private", action="store_true")
    p_search.add_argument("--limit", type=int, default=20)
    p_search.add_argument("--alpha", type=float, default=0.7, help="Lexical/semantic blend.")
    p_search.set_defaults(func=cmd_search)

    p_nl_search = sub.add_parser("nl-search", help="Natural-language project history search.")
    add_project_arg(p_nl_search)
    p_nl_search.add_argument("query")
    p_nl_search.add_argument("--session-id", default=None)
    p_nl_search.add_argument("--since", default=None, help="Override parsed lower bound (ISO time).")
    p_nl_search.add_argument("--until", default=None, help="Override parsed upper bound (ISO time).")
    p_nl_search.add_argument("--include-private", action="store_true")
    p_nl_search.add_argument("--limit", type=int, default=20)
    p_nl_search.add_argument("--alpha", type=float, default=0.7, help="Lexical/semantic blend.")
    p_nl_search.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS)
    p_nl_search.set_defaults(func=cmd_nl_search)

    # Alias for user-facing wording from mem-search skill.
    p_mem_search = sub.add_parser("mem-search", help="Alias of nl-search.")
    add_project_arg(p_mem_search)
    p_mem_search.add_argument("query")
    p_mem_search.add_argument("--session-id", default=None)
    p_mem_search.add_argument("--since", default=None)
    p_mem_search.add_argument("--until", default=None)
    p_mem_search.add_argument("--include-private", action="store_true")
    p_mem_search.add_argument("--limit", type=int, default=20)
    p_mem_search.add_argument("--alpha", type=float, default=0.7)
    p_mem_search.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS)
    p_mem_search.set_defaults(func=cmd_nl_search)

    p_timeline = sub.add_parser("timeline", help="Stage 2 retrieval: temporal neighborhood.")
    add_project_arg(p_timeline)
    p_timeline.add_argument("id", help="E<ID> or O<ID>")
    p_timeline.add_argument("--before", type=int, default=5)
    p_timeline.add_argument("--after", type=int, default=5)
    p_timeline.add_argument("--include-private", action="store_true")
    p_timeline.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS)
    p_timeline.set_defaults(func=cmd_timeline)

    p_get = sub.add_parser("get-observations", help="Stage 3 retrieval: full details by IDs.")
    add_project_arg(p_get)
    p_get.add_argument("ids", nargs="+", help="List of E<ID>/O<ID>")
    p_get.add_argument("--compact", action="store_true")
    p_get.add_argument("--include-private", action="store_true")
    p_get.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS)
    p_get.set_defaults(func=cmd_get_observations)

    p_ask = sub.add_parser("ask", help="Fused memory + repo_knowledge retrieval.")
    add_project_arg(p_ask)
    p_ask.add_argument("question")
    p_ask.add_argument("--session-id", default=None)
    p_ask.add_argument("--search-limit", type=int, default=None)
    p_ask.add_argument("--detail-limit", type=int, default=None)
    p_ask.add_argument("--code-top-k", type=int, default=None)
    p_ask.add_argument("--code-module-limit", type=int, default=None)
    p_ask.add_argument("--repo-index-dir", default=".codex_knowledge")
    p_ask.add_argument("--alpha", type=float, default=None)
    p_ask.add_argument("--include-private", action="store_true")
    p_ask.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS)
    p_ask.add_argument("--prompt-style", choices=["compact", "legacy"], default="compact")
    p_ask.add_argument("--mapping-fallback", choices=["auto", "off"], default="auto")
    p_ask.add_argument("--mapping-debug", action="store_true")
    p_ask.add_argument("--prompt-only", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except sqlite3.OperationalError as exc:
        payload = {
            "ok": False,
            "stage": str(getattr(args, "command", "unknown")),
            "error": "sqlite_operational_error",
            "message": str(exc),
            "hint": "Run `bash Scripts/codex_mem.sh init --project demo`, then retry.",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
