#!/usr/bin/env python3
"""
Local repository knowledge index and hybrid retrieval CLI.

Design goals:
- Keep context windows small by retrieving only relevant modules/chunks.
- Work offline with no external model dependencies.
- Support large repos via a reusable local index.
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
import urllib.error
import urllib.request
from typing import Dict, List, Mapping, Sequence, Tuple


INDEX_VERSION = "1"
DEFAULT_INDEX_DIR = ".codex_knowledge"
DEFAULT_DB_NAME = "repo_knowledge.sqlite3"

DEFAULT_MAX_FILE_BYTES = 300_000
DEFAULT_CHUNK_CHARS = 1_800
DEFAULT_CHUNK_OVERLAP_LINES = 8
DEFAULT_VECTOR_DIM = 256
DEFAULT_MODULE_DEPTH = 3

DEFAULT_EMBEDDING_PROVIDER = "local"
EMBEDDING_PROVIDERS = {"local", "openai"}
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_BATCH_SIZE = 64
DEFAULT_OPENAI_TIMEOUT_SEC = 60

DEFAULT_IGNORED_DIRS = {
    ".git",
    ".codex_knowledge",
    ".idea",
    ".vscode",
    ".venv",
    "__pycache__",
    "node_modules",
    "Pods",
    "build",
    "dist",
    "coverage",
    ".trash_logs",
    "DerivedData",
}

EXCLUDED_FILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "podfile.lock",
    "contents.json",
}

EXCLUDED_EXTENSIONS = {
    ".pbxproj",
    ".xcscheme",
    ".xcworkspacedata",
    ".storekit",
}

EXTENSION_TO_LANG = {
    ".swift": "swift",
    ".m": "objc",
    ".mm": "objc",
    ".h": "objc",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".go": "go",
    ".py": "python",
    ".rb": "ruby",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".sh": "shell",
    ".zsh": "shell",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".plist": "xml",
    ".pbxproj": "text",
    ".xcscheme": "xml",
    ".xcconfig": "text",
    ".entitlements": "xml",
}

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,}|[0-9]+|[\u4e00-\u9fff]+")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")

# Cold-start and cross-language retrieval guardrails:
# - codex-mem users often ask in Chinese, while code tokens are mostly English.
# - local embeddings are lexical/IDF based; query expansion improves module recall significantly.
ONBOARDING_TRIGGER_TERMS = {
    "learn",
    "onboard",
    "architecture",
    "module",
    "modules",
    "module map",
    "entrypoint",
    "main flow",
    "persistence",
    "database",
    "storage",
    "risk",
    "risks",
    "北极星",
    "学习",
    "架构",
    "模块",
    "入口",
    "主流程",
    "落库",
    "持久化",
    "风险",
}

LANG_SYMBOL_PATTERNS: Dict[str, Sequence[re.Pattern[str]]] = {
    "swift": (
        re.compile(r"\b(?:struct|class|actor|enum|protocol|extension)\s+([A-Za-z_]\w*)"),
        re.compile(r"\bfunc\s+([A-Za-z_]\w*)"),
        re.compile(r"\b(?:var|let)\s+([A-Za-z_]\w*)"),
    ),
    "typescript": (
        re.compile(r"\b(?:class|interface|type|enum)\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"\b(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)"),
    ),
    "javascript": (
        re.compile(r"\b(?:class)\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"\b(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)"),
    ),
    "python": (
        re.compile(r"\b(?:class|def)\s+([A-Za-z_]\w*)"),
    ),
    "go": (
        re.compile(r"\b(?:func|type|var|const)\s+([A-Za-z_]\w*)"),
    ),
}

GENERIC_SYMBOL_PATTERNS = (
    re.compile(r"\b(?:class|struct|interface|enum|protocol|type)\s+([A-Za-z_]\w*)"),
    re.compile(r"\b(?:func|function|def)\s+([A-Za-z_]\w*)"),
)


@dataclasses.dataclass
class FileDraft:
    path: str
    lang: str
    module_key: str
    byte_size: int
    line_count: int
    content_hash: str
    summary: str
    symbols: List[str]


@dataclasses.dataclass
class ChunkDraft:
    path: str
    lang: str
    start_line: int
    end_line: int
    text: str
    token_count: int
    tf: collections.Counter[str]
    symbol_hint: str


@dataclasses.dataclass
class ModuleDraft:
    file_count: int = 0
    chunk_count: int = 0
    langs: collections.Counter[str] = dataclasses.field(default_factory=collections.Counter)
    symbols: collections.Counter[str] = dataclasses.field(default_factory=collections.Counter)
    paths: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class QueryResult:
    chunk_id: int
    path: str
    start_line: int
    end_line: int
    text: str
    bm25: float
    semantic: float
    score: float
    symbol_hint: str


@dataclasses.dataclass
class EmbeddingConfig:
    provider: str = DEFAULT_EMBEDDING_PROVIDER
    vector_dim: int = DEFAULT_VECTOR_DIM
    openai_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
    openai_batch_size: int = DEFAULT_OPENAI_BATCH_SIZE
    openai_timeout_sec: int = DEFAULT_OPENAI_TIMEOUT_SEC
    openai_dimensions: int | None = None


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


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def is_onboarding_query(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if any(term in text for term in ("学习这个项目", "北极星", "模块地图", "主流程", "落库")):
        return True
    for term in ONBOARDING_TRIGGER_TERMS:
        if term in lowered:
            return True
    return False


def _ordered_unique(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        vv = (v or "").strip()
        if not vv:
            continue
        key = vv.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(vv)
    return out


def retrieval_hints(query: str) -> List[str]:
    """
    Lightweight query expansion to bridge non-English prompts and English code tokens.

    Goal: improve *recall* for cold-start onboarding and architecture questions without
    requiring any external translation/model calls.
    """
    q = (query or "").lower()
    raw = query or ""
    hints: List[str] = []

    onboarding = is_onboarding_query(raw)
    if onboarding:
        hints += [
            "learn project",
            "north star goal vision",
            "architecture module map components",
            "entrypoint main bootstrap startup",
            "main flow pipeline dataflow",
            "persistence database storage",
            "api routes backend server",
            "ai service model generation note prism organize",
            "top risks concurrency performance",
        ]

    if ("入口" in raw) or ("entrypoint" in q) or ("startup" in q) or ("bootstrap" in q):
        hints += ["entrypoint", "main", "startup", "bootstrap", "app", "appdelegate", "scenedelegate"]
    if ("主流程" in raw) or ("流程" in raw) or ("main flow" in q) or ("pipeline" in q):
        hints += ["main flow", "pipeline", "data flow", "request handler", "controller", "service"]
    if ("落库" in raw) or ("持久化" in raw) or ("数据库" in raw) or ("persistence" in q) or ("database" in q):
        hints += [
            "persistence",
            "database",
            "storage",
            "sqlite",
            "swiftdata",
            "coredata",
            "modelcontext",
            "save",
            "migration",
        ]
    if ("ai" in q) or ("模型" in raw) or ("生成" in raw) or ("generation" in q):
        hints += ["ai", "service", "model", "generation", "note", "prism", "organize"]
    if ("api" in q) or ("backend" in q) or ("后端" in raw) or ("接口" in raw) or ("路由" in raw):
        hints += ["backend", "api", "routes", "server", "client", "index.ts", "api.ts"]
    if ("风险" in raw) or ("risk" in q) or ("bug" in q) or ("incident" in q):
        hints += ["risk", "edge cases", "failure modes", "concurrency", "performance", "security"]

    return _ordered_unique(hints)


def onboarding_facet_queries(root: pathlib.Path, question: str) -> List[str]:
    """
    On cold start, a single broad query tends to overfit to docs or one module.
    Use a small fixed set of facet queries to guarantee coverage of:
    - entrypoint/startup
    - persistence/storage
    - AI/generation flows
    - backend/API surface (if present)
    """
    raw = question or ""
    facets: List[str] = []

    # 1) Entrypoint / startup
    facets.append("entrypoint main startup bootstrap swiftui @main App AppDelegate SceneDelegate")

    # 2) Persistence / storage
    if any(term in raw for term in ("落库", "持久化", "数据库")) or any(term in raw.lower() for term in ("persistence", "database", "storage")):
        facets.append("persistence database storage sqlite swiftdata coredata modelcontext save migration bootstrapper")
    else:
        # Still include a lightweight persistence probe for cold-start architecture mapping.
        facets.append("database persistence storage modelcontext save bootstrapper")

    # 3) AI / generation flows
    facets.append("ai service model generation note generation prism generation organize streaming delta")

    # 4) Backend/API surface (only if repo looks like it has one)
    has_backend = (root / "Backend").exists() or (root / "backend").exists() or (root / "api").exists()
    if has_backend or any(term in raw.lower() for term in ("backend", "api", "route", "server")) or any(term in raw for term in ("后端", "接口", "路由")):
        facets.append("backend api routes server index.ts api.ts")

    return _ordered_unique(facets)


def retrieve_onboarding_chunks(
    *,
    root: pathlib.Path,
    conn: sqlite3.Connection,
    meta: Mapping[str, str],
    vector_dim: int,
    embedding_provider: str,
    question: str,
    top_k: int,
    module_limit: int,
    alpha: float,
) -> Tuple[List[Tuple[str, float, str]], List[QueryResult], Dict[str, object]]:
    facets = onboarding_facet_queries(root, question)
    # Collect a larger pool, then let `diversify_chunks` choose the final top_k.
    per_facet_top_k = max(6, int(top_k))
    per_facet_module_limit = max(12, int(module_limit))

    combined: Dict[int, QueryResult] = {}
    facet_debug: List[Dict[str, object]] = []
    first_modules: List[Tuple[str, float, str]] = []

    for facet in facets:
        q_vec = build_query_vector(
            conn=conn,
            query=facet,
            meta=meta,
            embedding_provider=embedding_provider,
            vector_dim=vector_dim,
        )
        modules = choose_modules(
            conn=conn,
            query=facet,
            query_vec=q_vec,
            limit=per_facet_module_limit,
            vector_dim=vector_dim,
        )
        if not first_modules:
            first_modules = modules
        module_keys = [item[0] for item in modules]
        raw = retrieve_chunks(
            conn=conn,
            query=facet,
            top_k=per_facet_top_k,
            module_keys=module_keys,
            query_vec=q_vec,
            vector_dim=vector_dim,
            alpha=alpha,
        )
        for item in raw:
            prev = combined.get(item.chunk_id)
            if prev is None or item.score > prev.score:
                combined[item.chunk_id] = item
        facet_debug.append(
            {
                "facet": facet,
                "module_count": len(modules),
                "chunk_count": len(raw),
                "top_paths": [r.path for r in raw[:5]],
            }
        )

    merged = sorted(combined.values(), key=lambda item: item.score, reverse=True)

    def is_noise_path(p: str) -> bool:
        # Heuristic: if the repo has clear app/backend roots, treat Scripts/ as secondary for onboarding.
        has_app_like = (root / "App").exists() or (root / "Backend").exists() or (root / "src").exists()
        if has_app_like and (p.startswith("Scripts/") or p.startswith("scripts/")):
            return True
        return False

    merged = [item for item in merged if not is_noise_path(item.path)]
    selected = diversify_chunks(merged, int(top_k), doc_cap=2)

    def force_include_path_like(pattern: str, *, protect: Sequence[str]) -> None:
        nonlocal selected
        if any(pattern.strip("%").lower() in (item.path or "").lower() for item in selected):
            return
        row = conn.execute(
            """
            SELECT id, path, start_line, end_line, text, token_count, vector, symbol_hint
            FROM chunks
            WHERE path LIKE ?
            ORDER BY start_line ASC
            LIMIT 1
            """,
            (pattern,),
        ).fetchone()
        if not row:
            return
        forced = QueryResult(
            chunk_id=int(row["id"]),
            path=str(row["path"]),
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            text=str(row["text"]),
            bm25=0.0,
            semantic=0.0,
            score=0.0,
            symbol_hint=str(row["symbol_hint"]),
        )
        if is_noise_path(forced.path):
            return
        if any(item.chunk_id == forced.chunk_id or item.path == forced.path for item in selected):
            return

        protect_set = {p for p in protect if p}
        if len(selected) >= int(top_k):
            replacement_idx = None
            replacement_score = 1e18
            for idx, item in enumerate(selected):
                if chunk_category(item.path) in protect_set:
                    continue
                if item.score < replacement_score:
                    replacement_score = item.score
                    replacement_idx = idx
            if replacement_idx is None:
                replacement_idx = len(selected) - 1
                replacement_score = selected[replacement_idx].score
            forced.score = float(replacement_score)
            selected[replacement_idx] = forced
        else:
            selected.append(forced)

    q_lower = (question or "").lower()
    wants_generation = ("生成" in (question or "")) or ("generation" in q_lower) or ("ai" in q_lower) or ("stream" in q_lower)
    if wants_generation:
        force_include_path_like("%Generation%", protect=("entrypoint", "persistence"))

    wants_persistence = ("落库" in (question or "")) or ("持久化" in (question or "")) or ("database" in q_lower) or ("persistence" in q_lower)
    if wants_persistence:
        force_include_path_like("%Bootstrapper%", protect=("entrypoint", "ai_generation"))

    debug = {"facets": facets, "facet_runs": facet_debug}
    return first_modules, selected, debug


def effective_query_for_retrieval(query: str) -> Tuple[str, Dict[str, object]]:
    hints = retrieval_hints(query)
    if not hints:
        return query, {"expanded": False, "hints": []}

    # If the user prompt is mostly CJK, prefer an English-only retrieval query for the local TF/IDF embedding.
    # This avoids penalizing coverage/path overlap by including many non-matching CJK tokens.
    if contains_cjk(query):
        ascii_terms = _ordered_unique(ASCII_WORD_RE.findall(query or ""))
        effective = " ".join(hints + ascii_terms).strip()
        return effective, {"expanded": True, "strategy": "cjk_hints_only", "hints": hints, "ascii_terms": ascii_terms}

    effective = (query or "").strip() + "\n\nHints: " + " ".join(hints)
    return effective.strip(), {"expanded": True, "strategy": "append_hints", "hints": hints}


DOC_EXTENSIONS = {".md", ".rst", ".txt"}

GIT_STATUS_IGNORE_PREFIXES = (
    ".codex_knowledge",
    ".codex_mem",
)


def _is_ignored_git_status_path(path: str) -> bool:
    raw = (path or "").strip()
    if not raw:
        return False
    while raw.startswith("./"):
        raw = raw[2:]
    for prefix in GIT_STATUS_IGNORE_PREFIXES:
        if raw == prefix or raw.startswith(prefix + "/"):
            return True
    return False


def filter_git_status_porcelain(status: str) -> str:
    """
    Filter `git status --porcelain` output to ignore codex-mem generated index dirs.

    We preserve raw newlines so hashes match between `repo_knowledge` and `codex_mem`.
    """
    if not status:
        return ""
    kept: List[str] = []
    for line in status.splitlines(True):  # keepends=True
        path_part = line[3:].strip() if len(line) >= 4 else ""
        if not path_part:
            kept.append(line)
            continue
        candidates = [p.strip() for p in path_part.split(" -> ")] if " -> " in path_part else [path_part]
        if candidates and all(_is_ignored_git_status_path(p) for p in candidates if p):
            continue
        kept.append(line)
    return "".join(kept)


def is_doc_path(path: str) -> bool:
    try:
        return pathlib.Path(path).suffix.lower() in DOC_EXTENSIONS
    except Exception:
        return False


def chunk_category(path: str) -> str:
    lower = (path or "").lower()
    if is_doc_path(path):
        return "doc"
    if lower.endswith("app.swift") or "appdelegate" in lower or "scenedelegate" in lower or lower.endswith("main.swift"):
        return "entrypoint"
    if any(tok in lower for tok in ("database", "bootstrapper", "swiftdata", "coredata", "modelcontext", "sqlite", "migration", "store")):
        return "persistence"
    if "generation" in lower or "stream" in lower:
        return "ai_generation"
    if any(tok in lower for tok in ("/ai/", "ai", "llm", "prompt")):
        return "ai_service"
    if any(tok in lower for tok in ("backend", "/routes/", "api.ts", "index.ts", "server.ts")):
        return "backend"
    return "code"


def diversify_chunks(chunks: Sequence[QueryResult], top_k: int, *, doc_cap: int = 3) -> List[QueryResult]:
    """
    Onboarding tends to suffer from repeated hits in the same file/module.
    Prefer unique paths first, but also cap doc-heavy results so entrypoints/flows don't get crowded out.
    """
    if top_k <= 0:
        return []

    docs: List[QueryResult] = []
    code: List[QueryResult] = []
    for item in chunks:
        (docs if chunk_category(item.path) == "doc" else code).append(item)

    out: List[QueryResult] = []
    seen_ids: set[int] = set()
    seen_paths: set[str] = set()

    def emit_from(pool: Sequence[QueryResult], limit: int | None = None) -> None:
        nonlocal out
        for item in pool:
            if item.chunk_id in seen_ids:
                continue
            if item.path in seen_paths:
                continue
            out.append(item)
            seen_ids.add(item.chunk_id)
            seen_paths.add(item.path)
            if limit is not None and len(out) >= limit:
                return
            if len(out) >= top_k:
                return

    def emit_one(pool: Sequence[QueryResult]) -> bool:
        for item in pool:
            if item.chunk_id in seen_ids:
                continue
            if item.path in seen_paths:
                continue
            out.append(item)
            seen_ids.add(item.chunk_id)
            seen_paths.add(item.path)
            return True
        return False

    # Keep a small amount of docs for "north star / design intent", then prioritize code.
    doc_cap = max(0, int(doc_cap))
    if doc_cap:
        emit_from(docs, limit=min(top_k, doc_cap))

    # For onboarding, force a minimal coverage pack so entrypoints and core flows don't get crowded out.
    # This is intentionally heuristic and file-path driven (no extra model calls).
    by_cat: Dict[str, List[QueryResult]] = {"entrypoint": [], "persistence": [], "ai_generation": [], "backend": [], "ai_service": [], "code": []}
    for item in code:
        by_cat.setdefault(chunk_category(item.path), []).append(item)

    required = ["entrypoint", "persistence", "ai_generation"]
    for cat in required:
        if len(out) >= top_k:
            break
        emit_one(by_cat.get(cat, []))

    # Prefer AI service over generic code when generation flow isn't found.
    if len(out) < top_k and not any(chunk_category(item.path) == "ai_generation" for item in out):
        emit_one(by_cat.get("ai_service", []))

    emit_from(code)
    # If still not enough, fill with remaining (allow duplicates by path last).
    if len(out) < top_k:
        for item in chunks:
            if item.chunk_id in seen_ids:
                continue
            out.append(item)
            seen_ids.add(item.chunk_id)
            if len(out) >= top_k:
                break
    return out


def norm_vector(values: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return values
    return [v / norm for v in values]


def pack_vector(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes, dim: int) -> List[float]:
    expected = struct.calcsize(f"<{dim}f")
    if len(blob) != expected:
        raise ValueError(f"Vector size mismatch: expected={expected}, got={len(blob)}")
    return list(struct.unpack(f"<{dim}f", blob))


def vectorize_tf(
    tf: Mapping[str, int],
    idf: Mapping[str, float],
    dim: int,
) -> List[float]:
    vec = [0.0] * dim
    for token, count in tf.items():
        base = 1.0 + math.log(count)
        weight = base * idf.get(token, 1.0)
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign * weight
    return norm_vector(vec)


def chunked(values: Sequence[str], size: int) -> List[List[str]]:
    out: List[List[str]] = []
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    for idx in range(0, len(values), size):
        out.append(list(values[idx : idx + size]))
    return out


def openai_embed_texts(
    texts: Sequence[str],
    model: str,
    api_key: str,
    timeout_sec: int,
    batch_size: int,
    dimensions: int | None,
) -> List[List[float]]:
    if not texts:
        return []
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")

    url = "https://api.openai.com/v1/embeddings"
    vectors: List[List[float]] = []
    batches = chunked(list(texts), batch_size)
    total = len(batches)
    for idx, batch in enumerate(batches, start=1):
        payload: Dict[str, object] = {
            "model": model,
            "input": batch,
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions
        req = urllib.request.Request(
            url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI embeddings HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI embeddings request failed: {exc}") from exc

        data = json.loads(raw)
        rows = data.get("data")
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected OpenAI embeddings response: {raw[:300]}")
        rows_sorted = sorted(rows, key=lambda item: int(item.get("index", 0)))
        for row in rows_sorted:
            embedding = row.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("OpenAI embeddings response missing `embedding` array.")
            vec = [float(v) for v in embedding]
            vectors.append(norm_vector(vec))
        print(f"Embedding batch {idx}/{total} ({len(batch)} texts)")

    if len(vectors) != len(texts):
        raise RuntimeError(
            f"OpenAI embedding count mismatch: expected={len(texts)} got={len(vectors)}"
        )
    return vectors


def build_local_query_vector(
    conn: sqlite3.Connection,
    query: str,
    vector_dim: int,
) -> List[float]:
    meta = fetch_meta(conn)
    num_chunks = int(meta.get("chunk_count", "1"))
    q_tokens = tokenize(query)
    q_idf = query_tokens_to_idf(conn, q_tokens, num_chunks)
    return vectorize_tf(collections.Counter(q_tokens), q_idf, vector_dim) if q_tokens else []


def build_query_vector(
    conn: sqlite3.Connection,
    query: str,
    meta: Mapping[str, str],
    embedding_provider: str,
    vector_dim: int,
) -> List[float]:
    if embedding_provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print(
                "Warning: index uses OpenAI embeddings but OPENAI_API_KEY is missing; "
                "falling back to lexical-only retrieval.",
                file=sys.stderr,
            )
            return []
        model = meta.get("embedding_model", DEFAULT_OPENAI_EMBEDDING_MODEL)
        timeout_sec = int(meta.get("openai_timeout_sec", str(DEFAULT_OPENAI_TIMEOUT_SEC)))
        batch_size = int(meta.get("openai_batch_size", str(DEFAULT_OPENAI_BATCH_SIZE)))
        requested_dims = meta.get("openai_dimensions", "")
        dimensions = int(requested_dims) if requested_dims else None
        try:
            vec = openai_embed_texts(
                [query],
                model=model,
                api_key=api_key,
                timeout_sec=timeout_sec,
                batch_size=max(1, batch_size),
                dimensions=dimensions,
            )[0]
            if len(vec) != vector_dim:
                print(
                    f"Warning: query embedding dim={len(vec)} but index dim={vector_dim}; "
                    "ignoring semantic term for this query.",
                    file=sys.stderr,
                )
                return []
            return vec
        except Exception as exc:
            print(
                f"Warning: failed to call OpenAI embeddings ({exc}); "
                "falling back to lexical-only retrieval.",
                file=sys.stderr,
            )
            return []

    return build_local_query_vector(conn, query, vector_dim)


def cosine_sim(left: Sequence[float], right: Sequence[float]) -> float:
    total = 0.0
    for a, b in zip(left, right):
        total += a * b
    return total


def module_key_for_path(rel_path: pathlib.Path, depth: int) -> str:
    parent_parts = rel_path.parent.parts
    if not parent_parts:
        return "."
    use_depth = min(depth, len(parent_parts))
    return "/".join(parent_parts[:use_depth])


def looks_textual(path: pathlib.Path) -> bool:
    ext = path.suffix.lower()
    if ext in EXTENSION_TO_LANG:
        return True
    filename = path.name
    if filename in {"Dockerfile", "Makefile"}:
        return True
    return False


def path_is_ignored(rel_path: pathlib.Path, ignored_dirs: set[str]) -> bool:
    for part in rel_path.parts:
        if part in ignored_dirs:
            return True
    return False


def is_noise_file(rel_path: pathlib.Path) -> bool:
    lower_name = rel_path.name.lower()
    if lower_name in EXCLUDED_FILE_NAMES:
        # Keep generic `Contents.json`; skip only asset catalog metadata.
        if lower_name == "contents.json" and any(part.endswith(".xcassets") for part in rel_path.parts):
            return True
        if lower_name != "contents.json":
            return True
    if rel_path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True
    for part in rel_path.parts:
        if part.endswith(".xcodeproj") or part.endswith(".xcworkspace"):
            return True
    return False


def detect_lang(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext in EXTENSION_TO_LANG:
        return EXTENSION_TO_LANG[ext]
    if path.name in {"Dockerfile", "Makefile"}:
        return "text"
    return "text"


def read_text_file(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def git_tracked_paths(root: pathlib.Path) -> List[pathlib.Path] | None:
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    paths: List[pathlib.Path] = []
    for raw in proc.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        paths.append(pathlib.Path(raw))
    return paths


def discover_files(
    root: pathlib.Path,
    use_git_tracked: bool,
    max_file_bytes: int,
    ignored_dirs: set[str],
) -> List[pathlib.Path]:
    candidates: List[pathlib.Path] = []

    tracked = git_tracked_paths(root) if use_git_tracked else None
    if tracked is not None:
        for rel in tracked:
            if path_is_ignored(rel, ignored_dirs):
                continue
            abs_path = root / rel
            if not abs_path.exists() or not abs_path.is_file():
                continue
            if abs_path.stat().st_size > max_file_bytes:
                continue
            if is_noise_file(rel):
                continue
            if not looks_textual(rel):
                continue
            candidates.append(rel)
        return sorted(candidates)

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = pathlib.Path(dirpath).relative_to(root)
        dirnames[:] = [d for d in dirnames if d not in ignored_dirs]
        if path_is_ignored(rel_dir, ignored_dirs):
            continue
        for filename in filenames:
            rel_path = rel_dir / filename
            if path_is_ignored(rel_path, ignored_dirs):
                continue
            abs_path = root / rel_path
            if not abs_path.is_file():
                continue
            if abs_path.stat().st_size > max_file_bytes:
                continue
            if is_noise_file(rel_path):
                continue
            if not looks_textual(rel_path):
                continue
            candidates.append(rel_path)

    return sorted(candidates)


def extract_symbols(lang: str, text: str, limit: int = 30) -> List[str]:
    patterns = LANG_SYMBOL_PATTERNS.get(lang, ()) + GENERIC_SYMBOL_PATTERNS
    out: List[str] = []
    seen: set[str] = set()

    for pattern in patterns:
        for match in pattern.finditer(text):
            symbol = match.group(1)
            if len(symbol) <= 1:
                continue
            if symbol in seen:
                continue
            seen.add(symbol)
            out.append(symbol)
            if len(out) >= limit:
                return out
    return out


def extract_leading_comment(lines: Sequence[str], max_scan: int = 20) -> str:
    comment_chunks: List[str] = []
    for line in lines[:max_scan]:
        stripped = line.strip()
        if not stripped:
            if comment_chunks:
                break
            continue
        if stripped.startswith("//"):
            comment_chunks.append(stripped.lstrip("/ ").strip())
        elif stripped.startswith("#"):
            comment_chunks.append(stripped.lstrip("# ").strip())
        elif stripped.startswith("/*") or stripped.startswith("*"):
            cleaned = stripped.lstrip("/* ").rstrip("*/ ").strip()
            if cleaned:
                comment_chunks.append(cleaned)
        else:
            if comment_chunks:
                break
    return " ".join(comment_chunks[:3]).strip()


def summarize_file(
    path: str,
    lang: str,
    lines: Sequence[str],
    symbols: Sequence[str],
) -> str:
    comment = extract_leading_comment(lines)
    symbol_text = ", ".join(symbols[:6]) if symbols else "n/a"
    if comment:
        return f"{lang} file `{path}`: {comment}. symbols: {symbol_text}"
    return f"{lang} file `{path}` with {len(lines)} lines. symbols: {symbol_text}"


def chunk_lines(
    lines: Sequence[str],
    max_chars: int,
    overlap_lines: int,
) -> List[Tuple[int, int, str]]:
    if not lines:
        return []

    chunks: List[Tuple[int, int, str]] = []
    start = 0
    total = len(lines)

    while start < total:
        end = start
        char_count = 0

        while end < total:
            line_len = len(lines[end])
            if end > start and (char_count + line_len) > max_chars:
                break
            char_count += line_len
            end += 1
            if char_count >= max_chars:
                break

        if end <= start:
            end = min(total, start + 1)

        # Prefer ending on a nearby blank line for cleaner chunk boundaries.
        for probe in range(end, min(total, end + 8)):
            if lines[probe - 1].strip() == "":
                end = probe
                break

        body = "".join(lines[start:end]).strip()
        if body:
            chunks.append((start + 1, end, body))

        if end >= total:
            break

        next_start = end - overlap_lines
        if next_start <= start:
            next_start = start + 1
        start = max(0, next_start)

    return chunks


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            lang TEXT NOT NULL,
            module_key TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            line_count INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            summary TEXT NOT NULL,
            symbols_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            lang TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            text TEXT NOT NULL,
            vector BLOB NOT NULL,
            symbol_hint TEXT NOT NULL,
            FOREIGN KEY(path) REFERENCES files(path) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);

        CREATE TABLE IF NOT EXISTS postings (
            token TEXT NOT NULL,
            chunk_id INTEGER NOT NULL,
            tf INTEGER NOT NULL,
            PRIMARY KEY(token, chunk_id),
            FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_postings_token ON postings(token);

        CREATE TABLE IF NOT EXISTS token_df (
            token TEXT PRIMARY KEY,
            df INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS modules (
            module_key TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            file_count INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            top_symbols TEXT NOT NULL,
            top_paths TEXT NOT NULL,
            vector BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def clear_index(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DELETE FROM postings;
        DELETE FROM chunks;
        DELETE FROM files;
        DELETE FROM token_df;
        DELETE FROM modules;
        DELETE FROM meta;
        """
    )


def bm25(
    tf: int,
    doc_len: int,
    avg_len: float,
    df: int,
    num_docs: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    if tf <= 0 or doc_len <= 0 or avg_len <= 0.0 or df <= 0:
        return 0.0
    idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
    numer = tf * (k1 + 1.0)
    denom = tf + k1 * (1.0 - b + b * (doc_len / avg_len))
    return idf * (numer / denom)


def normalize_scores(scores: Mapping[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def build_index(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    index_dir = (root / args.index_dir).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / DEFAULT_DB_NAME

    ignored_dirs = set(DEFAULT_IGNORED_DIRS)
    ignored_dirs.update(args.ignore_dir or [])

    rel_paths = discover_files(
        root=root,
        use_git_tracked=not args.all_files,
        max_file_bytes=args.max_file_bytes,
        ignored_dirs=ignored_dirs,
    )
    if not rel_paths:
        print("No indexable files found.")
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    file_rows: List[FileDraft] = []
    chunk_rows: List[ChunkDraft] = []
    token_df: collections.Counter[str] = collections.Counter()
    module_drafts: Dict[str, ModuleDraft] = {}
    skipped: List[str] = []

    for rel_path in rel_paths:
        abs_path = root / rel_path
        try:
            text = read_text_file(abs_path)
        except Exception:
            skipped.append(str(rel_path))
            continue

        lines = text.splitlines(keepends=True)
        lang = detect_lang(rel_path)
        symbols = extract_symbols(lang, text)
        module_key = module_key_for_path(rel_path, args.module_depth)
        summary = summarize_file(str(rel_path), lang, lines, symbols)
        content_hash = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

        file_rows.append(
            FileDraft(
                path=str(rel_path),
                lang=lang,
                module_key=module_key,
                byte_size=abs_path.stat().st_size,
                line_count=len(lines),
                content_hash=content_hash,
                summary=summary,
                symbols=symbols,
            )
        )

        chunks = chunk_lines(
            lines=lines,
            max_chars=args.chunk_chars,
            overlap_lines=args.chunk_overlap_lines,
        )
        module_state = module_drafts.setdefault(module_key, ModuleDraft())
        module_state.file_count += 1
        module_state.langs[lang] += 1
        module_state.paths.append(str(rel_path))
        for symbol in symbols:
            module_state.symbols[symbol] += 1

        for start_line, end_line, body in chunks:
            tokens = tokenize(body)
            if not tokens:
                continue
            tf = collections.Counter(tokens)
            for token in tf:
                token_df[token] += 1
            chunk_rows.append(
                ChunkDraft(
                    path=str(rel_path),
                    lang=lang,
                    start_line=start_line,
                    end_line=end_line,
                    text=body,
                    token_count=len(tokens),
                    tf=tf,
                    symbol_hint=", ".join(symbols[:4]),
                )
            )
            module_state.chunk_count += 1

    if not file_rows or not chunk_rows:
        print("No textual content available after filtering.")
        return 1

    num_chunks = len(chunk_rows)
    avg_chunk_tokens = sum(c.token_count for c in chunk_rows) / num_chunks
    idf = {
        token: math.log((num_chunks + 1.0) / (df + 0.5)) + 1.0
        for token, df in token_df.items()
    }

    embedding_provider = str(args.embedding_provider).strip().lower()
    if embedding_provider not in EMBEDDING_PROVIDERS:
        print(
            f"Unsupported embedding provider `{embedding_provider}`. "
            f"Choose one of: {', '.join(sorted(EMBEDDING_PROVIDERS))}.",
            file=sys.stderr,
        )
        return 2

    embedding_cfg = EmbeddingConfig(
        provider=embedding_provider,
        vector_dim=args.vector_dim,
        openai_model=args.openai_model,
        openai_batch_size=max(1, args.openai_batch_size),
        openai_timeout_sec=max(1, args.openai_timeout_sec),
        openai_dimensions=args.openai_dimensions,
    )

    module_payloads: List[Tuple[str, str, int, int, List[str], List[str]]] = []
    for module_key, state in module_drafts.items():
        top_symbols = [name for name, _ in state.symbols.most_common(10)]
        top_paths = sorted(state.paths)[:6]
        langs = ", ".join(f"{name}:{count}" for name, count in state.langs.most_common())
        symbol_text = ", ".join(top_symbols) if top_symbols else "n/a"
        summary = (
            f"module `{module_key}` with {state.file_count} files and {state.chunk_count} chunks. "
            f"langs: {langs}. symbols: {symbol_text}"
        )
        module_payloads.append(
            (
                module_key,
                summary,
                state.file_count,
                state.chunk_count,
                top_symbols,
                top_paths,
            )
        )

    chunk_vectors: List[List[float]] = []
    module_vectors: List[List[float]] = []
    actual_vector_dim = embedding_cfg.vector_dim

    if embedding_cfg.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print(
                "OPENAI_API_KEY is required when using --embedding-provider openai.",
                file=sys.stderr,
            )
            return 2

        print("Generating OpenAI embeddings for chunks...")
        chunk_vectors = openai_embed_texts(
            texts=[chunk.text for chunk in chunk_rows],
            model=embedding_cfg.openai_model,
            api_key=api_key,
            timeout_sec=embedding_cfg.openai_timeout_sec,
            batch_size=embedding_cfg.openai_batch_size,
            dimensions=embedding_cfg.openai_dimensions,
        )
        if not chunk_vectors:
            print("OpenAI embedding returned no vectors for chunks.", file=sys.stderr)
            return 2
        actual_vector_dim = len(chunk_vectors[0])
        if any(len(vec) != actual_vector_dim for vec in chunk_vectors):
            print("Inconsistent chunk embedding dimensions from OpenAI.", file=sys.stderr)
            return 2

        print("Generating OpenAI embeddings for modules...")
        module_vectors = openai_embed_texts(
            texts=[item[1] for item in module_payloads],
            model=embedding_cfg.openai_model,
            api_key=api_key,
            timeout_sec=embedding_cfg.openai_timeout_sec,
            batch_size=embedding_cfg.openai_batch_size,
            dimensions=embedding_cfg.openai_dimensions,
        )
        if len(module_vectors) != len(module_payloads):
            print("OpenAI embedding returned invalid module vector count.", file=sys.stderr)
            return 2
        if any(len(vec) != actual_vector_dim for vec in module_vectors):
            print("Inconsistent module embedding dimensions from OpenAI.", file=sys.stderr)
            return 2
    else:
        actual_vector_dim = embedding_cfg.vector_dim
        chunk_vectors = [vectorize_tf(chunk.tf, idf, actual_vector_dim) for chunk in chunk_rows]
        for module_key, summary, _file_count, _chunk_count, top_symbols, _top_paths in module_payloads:
            module_tf = collections.Counter(tokenize(summary + " " + " ".join(top_symbols)))
            module_vectors.append(vectorize_tf(module_tf, idf, actual_vector_dim))

    with conn:
        clear_index(conn)
        conn.executemany(
            """
            INSERT INTO files(path, lang, module_key, byte_size, line_count, content_hash, summary, symbols_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f.path,
                    f.lang,
                    f.module_key,
                    f.byte_size,
                    f.line_count,
                    f.content_hash,
                    f.summary,
                    json.dumps(f.symbols, ensure_ascii=False),
                )
                for f in file_rows
            ],
        )

        for chunk, vec in zip(chunk_rows, chunk_vectors):
            cursor = conn.execute(
                """
                INSERT INTO chunks(path, lang, start_line, end_line, token_count, text, vector, symbol_hint)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.path,
                    chunk.lang,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.token_count,
                    chunk.text,
                    pack_vector(vec),
                    chunk.symbol_hint,
                ),
            )
            chunk_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO postings(token, chunk_id, tf) VALUES(?, ?, ?)",
                [(token, chunk_id, tf) for token, tf in chunk.tf.items()],
            )

        conn.executemany(
            "INSERT INTO token_df(token, df) VALUES(?, ?)",
            [(token, df) for token, df in token_df.items()],
        )

        for (module_key, summary, file_count, chunk_count, top_symbols, top_paths), module_vec in zip(
            module_payloads,
            module_vectors,
        ):
            conn.execute(
                """
                INSERT INTO modules(module_key, summary, file_count, chunk_count, top_symbols, top_paths, vector)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    module_key,
                    summary,
                    file_count,
                    chunk_count,
                    json.dumps(top_symbols, ensure_ascii=False),
                    json.dumps(top_paths, ensure_ascii=False),
                    pack_vector(module_vec),
                ),
            )

        git_head = ""
        git_head_committed_at = ""
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
            git_head = proc.stdout.strip()
        except Exception:
            git_head = ""
        try:
            proc = subprocess.run(
                ["git", "show", "-s", "--format=%cI", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
            git_head_committed_at = proc.stdout.strip()
        except Exception:
            git_head_committed_at = ""

        git_status = ""
        git_dirty = "0"
        git_status_hash = ""
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
            git_status = filter_git_status_porcelain(proc.stdout or "")
        except Exception:
            git_status = ""
        if git_status.strip():
            git_dirty = "1"
        git_status_hash = hashlib.blake2b(git_status.encode("utf-8"), digest_size=16).hexdigest()

        meta_rows = {
            "index_version": INDEX_VERSION,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "root": str(root),
            "git_head": git_head,
            "git_head_committed_at": git_head_committed_at,
            "git_dirty": git_dirty,
            "git_status_hash": git_status_hash,
            "file_count": str(len(file_rows)),
            "chunk_count": str(num_chunks),
            "avg_chunk_tokens": f"{avg_chunk_tokens:.3f}",
            "vector_dim": str(actual_vector_dim),
            "module_count": str(len(module_drafts)),
            "use_git_tracked": "0" if args.all_files else "1",
            "embedding_provider": embedding_cfg.provider,
            "embedding_model": embedding_cfg.openai_model if embedding_cfg.provider == "openai" else "",
            "openai_batch_size": str(embedding_cfg.openai_batch_size),
            "openai_timeout_sec": str(embedding_cfg.openai_timeout_sec),
            "openai_dimensions": str(embedding_cfg.openai_dimensions or ""),
        }
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?)",
            list(meta_rows.items()),
        )

    summary_payload = {
        "db_path": str(db_path),
        "file_count": len(file_rows),
        "chunk_count": num_chunks,
        "module_count": len(module_drafts),
        "avg_chunk_tokens": round(avg_chunk_tokens, 3),
        "embedding_provider": embedding_cfg.provider,
        "vector_dim": actual_vector_dim,
        "skipped_files": skipped[:20],
    }
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


def open_db(root: pathlib.Path, index_dir: str) -> sqlite3.Connection:
    db_path = (root / index_dir / DEFAULT_DB_NAME).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Index not found: {db_path}. Run `index` first.")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_meta(conn: sqlite3.Connection) -> Dict[str, str]:
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {row["key"]: row["value"] for row in rows}


def query_tokens_to_idf(
    conn: sqlite3.Connection,
    tokens: Sequence[str],
    num_chunks: int,
) -> Dict[str, float]:
    if not tokens:
        return {}
    uniq = sorted(set(tokens))
    placeholders = ",".join("?" for _ in uniq)
    rows = conn.execute(
        f"SELECT token, df FROM token_df WHERE token IN ({placeholders})",
        uniq,
    ).fetchall()
    df_map = {row["token"]: int(row["df"]) for row in rows}
    out: Dict[str, float] = {}
    for token in uniq:
        df = df_map.get(token, 1)
        out[token] = math.log((num_chunks + 1.0) / (df + 0.5)) + 1.0
    return out


def choose_modules(
    conn: sqlite3.Connection,
    query: str,
    query_vec: Sequence[float],
    limit: int,
    vector_dim: int,
) -> List[Tuple[str, float, str]]:
    rows = conn.execute(
        "SELECT module_key, summary, file_count, chunk_count, top_symbols, top_paths, vector FROM modules"
    ).fetchall()
    if not rows:
        return []

    q_tokens = tokenize(query)
    q_token_set = set(q_tokens)
    scored: List[Tuple[str, float, str, int]] = []
    for row in rows:
        module_key = row["module_key"]
        summary = row["summary"]
        file_count = int(row["file_count"])
        module_tokens = set(tokenize(summary))
        top_symbols = " ".join(json.loads(row["top_symbols"]))
        top_paths = " ".join(json.loads(row["top_paths"]))
        symbol_tokens = set(tokenize(top_symbols))
        path_tokens = set(tokenize(top_paths))
        summary_overlap = (len(q_token_set & module_tokens) / len(q_token_set)) if q_token_set else 0.0
        symbol_overlap = (len(q_token_set & symbol_tokens) / len(q_token_set)) if q_token_set else 0.0
        path_overlap = (len(q_token_set & path_tokens) / len(q_token_set)) if q_token_set else 0.0
        overlap = 0.5 * summary_overlap + 0.3 * symbol_overlap + 0.2 * path_overlap
        sim = 0.0
        if query_vec:
            vec = unpack_vector(row["vector"], vector_dim)
            sim = max(0.0, cosine_sim(query_vec, vec))
        score = 0.6 * overlap + 0.4 * sim
        scored.append((module_key, score, summary, file_count))

    scored.sort(key=lambda item: item[1], reverse=True)

    # Fallback for very short/novel queries where overlap and semantic scores are near-zero.
    if scored and scored[0][1] <= 1e-9:
        scored.sort(key=lambda item: item[3], reverse=True)

    selected: List[Tuple[str, float, str]] = []
    seen: set[str] = set()

    # If confidence is low, blend score-first and size-first selections.
    low_confidence = bool(scored) and scored[0][1] < 0.30
    if low_confidence:
        primary = max(1, limit // 2)
        for key, score, summary, _file_count in scored[:primary]:
            if key in seen:
                continue
            selected.append((key, score, summary))
            seen.add(key)

        by_size = sorted(scored, key=lambda item: item[3], reverse=True)
        for key, score, summary, _file_count in by_size:
            if key in seen:
                continue
            selected.append((key, score, summary))
            seen.add(key)
            if len(selected) >= limit:
                break
    else:
        for key, score, summary, _file_count in scored:
            if key in seen:
                continue
            selected.append((key, score, summary))
            seen.add(key)
            if len(selected) >= limit:
                break

    return selected[:limit]


def make_module_filter_sql(module_keys: Sequence[str]) -> Tuple[str, List[str]]:
    keys = [k for k in module_keys if k and k != "."]
    if not keys:
        return "", []
    clauses = []
    params: List[str] = []
    for key in keys:
        clauses.append("path LIKE ?")
        params.append(f"{key}%")
    return "(" + " OR ".join(clauses) + ")", params


def retrieve_chunks(
    conn: sqlite3.Connection,
    query: str,
    top_k: int,
    module_keys: Sequence[str],
    query_vec: Sequence[float],
    vector_dim: int,
    alpha: float,
) -> List[QueryResult]:
    tokens = tokenize(query)
    uniq_tokens = sorted(set(tokens))
    meta = fetch_meta(conn)
    num_chunks = int(meta.get("chunk_count", "1"))
    avg_chunk_tokens = float(meta.get("avg_chunk_tokens", "200.0"))
    q_vec = list(query_vec)

    module_sql, module_params = make_module_filter_sql(module_keys)
    where_clause = f"WHERE {module_sql}" if module_sql else ""

    chunk_rows = conn.execute(
        f"""
        SELECT id, path, start_line, end_line, text, token_count, vector, symbol_hint
        FROM chunks
        {where_clause}
        """,
        module_params,
    ).fetchall()
    if not chunk_rows:
        return []

    candidate_ids = [int(row["id"]) for row in chunk_rows]
    candidate_id_set = set(candidate_ids)
    chunk_by_id = {int(row["id"]): row for row in chunk_rows}

    bm25_scores: Dict[int, float] = collections.defaultdict(float)
    coverage_scores: Dict[int, float] = {}
    matched_terms: Dict[int, set[str]] = collections.defaultdict(set)
    if tokens:
        token_placeholders = ",".join("?" for _ in uniq_tokens)
        sql = (
            "SELECT p.chunk_id, p.token, p.tf, c.token_count "
            "FROM postings p JOIN chunks c ON c.id = p.chunk_id "
            f"WHERE p.token IN ({token_placeholders})"
        )
        params: List[object] = list(uniq_tokens)
        if module_sql:
            sql += f" AND {module_sql}"
            params.extend(module_params)
        posting_rows = conn.execute(sql, params).fetchall()

        df_rows = conn.execute(
            f"SELECT token, df FROM token_df WHERE token IN ({token_placeholders})",
            uniq_tokens,
        ).fetchall()
        token_df = {row["token"]: int(row["df"]) for row in df_rows}
        for row in posting_rows:
            chunk_id = int(row["chunk_id"])
            if chunk_id not in candidate_id_set:
                continue
            tf = int(row["tf"])
            doc_len = int(row["token_count"])
            token = row["token"]
            matched_terms[chunk_id].add(token)
            df = token_df.get(token, 1)
            bm25_scores[chunk_id] += bm25(
                tf=tf,
                doc_len=doc_len,
                avg_len=avg_chunk_tokens,
                df=df,
                num_docs=num_chunks,
            )
        if uniq_tokens:
            denom = float(len(uniq_tokens))
            for chunk_id, terms in matched_terms.items():
                coverage_scores[chunk_id] = len(terms) / denom

    semantic_scores: Dict[int, float] = {}
    if q_vec:
        for chunk_id in candidate_ids:
            row = chunk_by_id[chunk_id]
            vec = unpack_vector(row["vector"], vector_dim)
            semantic_scores[chunk_id] = max(0.0, cosine_sim(q_vec, vec))

    bm25_norm = normalize_scores(bm25_scores)
    semantic_norm = normalize_scores(semantic_scores)
    coverage_norm = coverage_scores

    all_ids = set(candidate_ids)
    ranked: List[QueryResult] = []
    q_token_set = set(uniq_tokens)
    for chunk_id in all_ids:
        bm = bm25_norm.get(chunk_id, 0.0)
        sm = semantic_norm.get(chunk_id, 0.0)
        row = chunk_by_id[chunk_id]
        path_tokens = set(tokenize(str(row["path"])))
        symbol_tokens = set(tokenize(str(row["symbol_hint"])))
        if q_token_set:
            path_overlap = len(path_tokens & q_token_set) / len(q_token_set)
            symbol_overlap = len(symbol_tokens & q_token_set) / len(q_token_set)
            structural = 0.6 * path_overlap + 0.4 * symbol_overlap
        else:
            structural = 0.0
        coverage = coverage_norm.get(chunk_id, 0.0)
        lexical_semantic = alpha * bm + (1.0 - alpha) * sm
        score = 0.75 * lexical_semantic + 0.15 * coverage + 0.10 * structural
        ranked.append(
            QueryResult(
                chunk_id=chunk_id,
                path=row["path"],
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                text=row["text"],
                bm25=bm,
                semantic=sm,
                score=score,
                symbol_hint=row["symbol_hint"],
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


def trim_snippet(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...<trimmed>..."


def cmd_map(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    rows = conn.execute(
        """
        SELECT module_key, summary, file_count, chunk_count, top_paths
        FROM modules
        ORDER BY file_count DESC, chunk_count DESC
        LIMIT ?
        """,
        (args.limit,),
    ).fetchall()
    if not rows:
        print("No modules found.")
        return 1

    for idx, row in enumerate(rows, start=1):
        paths = json.loads(row["top_paths"])
        sample = ", ".join(paths[:3])
        print(
            f"{idx:02d}. {row['module_key']} "
            f"(files={row['file_count']}, chunks={row['chunk_count']})"
        )
        print(f"    {row['summary']}")
        if sample:
            print(f"    sample: {sample}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    meta = fetch_meta(conn)
    vector_dim = int(meta.get("vector_dim", str(DEFAULT_VECTOR_DIM)))
    embedding_provider = meta.get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER)
    onboarding = is_onboarding_query(args.question)
    effective_query, expansion = effective_query_for_retrieval(args.question)

    effective_module_limit = int(args.module_limit)
    if onboarding:
        effective_module_limit = max(effective_module_limit, 12)

    debug: Dict[str, object] = {}
    if onboarding:
        modules, chunks, debug = retrieve_onboarding_chunks(
            root=root,
            conn=conn,
            meta=meta,
            vector_dim=vector_dim,
            embedding_provider=embedding_provider,
            question=args.question,
            top_k=int(args.top_k),
            module_limit=effective_module_limit,
            alpha=args.alpha,
        )
    else:
        q_vec = build_query_vector(
            conn=conn,
            query=effective_query,
            meta=meta,
            embedding_provider=embedding_provider,
            vector_dim=vector_dim,
        )
        modules = choose_modules(
            conn=conn,
            query=effective_query,
            query_vec=q_vec,
            limit=effective_module_limit,
            vector_dim=vector_dim,
        )
        module_keys = [item[0] for item in modules]
        chunks = retrieve_chunks(
            conn=conn,
            query=effective_query,
            top_k=int(args.top_k),
            module_keys=module_keys,
            query_vec=q_vec,
            vector_dim=vector_dim,
            alpha=args.alpha,
        )

    if args.json:
        payload = {
            "question": args.question,
            "effective_query": effective_query if expansion.get("expanded") else args.question,
            "query_expansion": expansion,
            "effective_module_limit": effective_module_limit,
            "onboarding_mode": onboarding,
            "onboarding_debug": debug,
            "modules": [
                {"module": module, "score": round(score, 4), "summary": summary}
                for module, score, summary in modules
            ],
            "chunks": [
                {
                    "path": item.path,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "score": round(item.score, 4),
                    "bm25": round(item.bm25, 4),
                    "semantic": round(item.semantic, 4),
                    "symbol_hint": item.symbol_hint,
                    "snippet": trim_snippet(item.text, args.snippet_chars),
                }
                for item in chunks
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Question: {args.question}")
    print()
    print("Module recall:")
    for module, score, _summary in modules:
        print(f"- {module} (score={score:.3f})")
    if not modules:
        print("- <none>")
    print()

    print("Top chunks:")
    for idx, item in enumerate(chunks, start=1):
        print(
            f"{idx}. {item.path}:{item.start_line}-{item.end_line} "
            f"(score={item.score:.3f}, bm25={item.bm25:.3f}, semantic={item.semantic:.3f})"
        )
        if item.symbol_hint:
            print(f"   symbols: {item.symbol_hint}")
        snippet = trim_snippet(item.text, args.snippet_chars).replace("\n", "\n   ")
        print(f"   {snippet}")
        print()

    if not chunks:
        print("No matching chunks. Try a broader question.")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    conn = open_db(root, args.index_dir)
    meta = fetch_meta(conn)
    vector_dim = int(meta.get("vector_dim", str(DEFAULT_VECTOR_DIM)))
    embedding_provider = meta.get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER)
    onboarding = is_onboarding_query(args.question)
    effective_query, expansion = effective_query_for_retrieval(args.question)

    effective_module_limit = int(args.module_limit)
    if onboarding:
        effective_module_limit = max(effective_module_limit, 12)

    debug: Dict[str, object] = {}
    if onboarding:
        modules, chunks, debug = retrieve_onboarding_chunks(
            root=root,
            conn=conn,
            meta=meta,
            vector_dim=vector_dim,
            embedding_provider=embedding_provider,
            question=args.question,
            top_k=int(args.top_k),
            module_limit=effective_module_limit,
            alpha=args.alpha,
        )
    else:
        q_vec = build_query_vector(
            conn=conn,
            query=effective_query,
            meta=meta,
            embedding_provider=embedding_provider,
            vector_dim=vector_dim,
        )
        modules = choose_modules(
            conn=conn,
            query=effective_query,
            query_vec=q_vec,
            limit=effective_module_limit,
            vector_dim=vector_dim,
        )
        module_keys = [item[0] for item in modules]
        chunks = retrieve_chunks(
            conn=conn,
            query=effective_query,
            top_k=int(args.top_k),
            module_keys=module_keys,
            query_vec=q_vec,
            vector_dim=vector_dim,
            alpha=args.alpha,
        )

    print("System: You are assisting with this repository. Use only the provided contexts.")
    print("If you are uncertain, explicitly say what is missing.")
    print()
    print(f"Question: {args.question}")
    if expansion.get("expanded"):
        print(f"(retrieval query expanded: {expansion.get('strategy','')})")
    if onboarding:
        print("(onboarding mode: facet retrieval enabled)")
    print()
    print("Contexts:")
    for idx, item in enumerate(chunks, start=1):
        print(f"[{idx}] {item.path}:{item.start_line}-{item.end_line}")
        print(trim_snippet(item.text, args.snippet_chars))
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/query a local hybrid index for large codebase learning."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root path (default: current directory).",
    )
    parser.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help=f"Index directory under root (default: {DEFAULT_INDEX_DIR}).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build or rebuild repository index.")
    p_index.add_argument(
        "--all-files",
        action="store_true",
        help="Scan filesystem instead of only git-tracked files.",
    )
    p_index.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help=f"Skip files larger than this size (default: {DEFAULT_MAX_FILE_BYTES}).",
    )
    p_index.add_argument(
        "--chunk-chars",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        help=f"Chunk target size in characters (default: {DEFAULT_CHUNK_CHARS}).",
    )
    p_index.add_argument(
        "--chunk-overlap-lines",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_LINES,
        help=f"Chunk overlap in lines (default: {DEFAULT_CHUNK_OVERLAP_LINES}).",
    )
    p_index.add_argument(
        "--embedding-provider",
        choices=sorted(EMBEDDING_PROVIDERS),
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Embedding provider: local (offline) or openai.",
    )
    p_index.add_argument(
        "--vector-dim",
        type=int,
        default=DEFAULT_VECTOR_DIM,
        help=f"Vector dimension for local provider (default: {DEFAULT_VECTOR_DIM}).",
    )
    p_index.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_EMBEDDING_MODEL,
        help=f"OpenAI embedding model (default: {DEFAULT_OPENAI_EMBEDDING_MODEL}).",
    )
    p_index.add_argument(
        "--openai-batch-size",
        type=int,
        default=DEFAULT_OPENAI_BATCH_SIZE,
        help=f"OpenAI embedding batch size (default: {DEFAULT_OPENAI_BATCH_SIZE}).",
    )
    p_index.add_argument(
        "--openai-timeout-sec",
        type=int,
        default=DEFAULT_OPENAI_TIMEOUT_SEC,
        help=f"OpenAI request timeout seconds (default: {DEFAULT_OPENAI_TIMEOUT_SEC}).",
    )
    p_index.add_argument(
        "--openai-dimensions",
        type=int,
        default=None,
        help="Optional OpenAI embedding dimensions (supported by text-embedding-3* models).",
    )
    p_index.add_argument(
        "--module-depth",
        type=int,
        default=DEFAULT_MODULE_DEPTH,
        help=f"Path depth used as module key (default: {DEFAULT_MODULE_DEPTH}).",
    )
    p_index.add_argument(
        "--ignore-dir",
        action="append",
        help="Additional directory names to ignore. Repeatable.",
    )
    p_index.set_defaults(handler=build_index)

    p_map = sub.add_parser("map", help="Show top-level module map.")
    p_map.add_argument("--limit", type=int, default=25, help="Module rows to print.")
    p_map.set_defaults(handler=cmd_map)

    p_query = sub.add_parser("query", help="Run hybrid retrieval query.")
    p_query.add_argument("question", help="Question text.")
    p_query.add_argument("--top-k", type=int, default=8, help="Number of chunks to return.")
    p_query.add_argument(
        "--module-limit",
        type=int,
        default=4,
        help="Number of recalled modules for second-stage retrieval.",
    )
    p_query.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Lexical/BM25 weight in [0,1] (default: 0.65).",
    )
    p_query.add_argument(
        "--snippet-chars",
        type=int,
        default=600,
        help="Max characters shown for each chunk snippet.",
    )
    p_query.add_argument("--json", action="store_true", help="Print JSON output.")
    p_query.set_defaults(handler=cmd_query)

    p_prompt = sub.add_parser("prompt", help="Emit a ready-to-use RAG prompt context.")
    p_prompt.add_argument("question", help="Question text.")
    p_prompt.add_argument("--top-k", type=int, default=8, help="Number of chunks to include.")
    p_prompt.add_argument("--module-limit", type=int, default=4, help="Recalled modules.")
    p_prompt.add_argument("--alpha", type=float, default=0.65, help="BM25 blend weight.")
    p_prompt.add_argument(
        "--snippet-chars",
        type=int,
        default=1000,
        help="Snippet length per chunk.",
    )
    p_prompt.set_defaults(handler=cmd_prompt)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
