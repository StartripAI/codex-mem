#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

from prompt_profiles import PromptProfile


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _dedupe_lines(text: str, *, max_lines: int) -> str:
    seen = set()
    out: List[str] = []
    for line in (text or "").splitlines():
        cleaned = line.rstrip()
        if not cleaned:
            continue
        key = cleaned.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= max_lines:
            break
    return "\n".join(out)


def _trim_to_token_budget(text: str, token_budget: int) -> str:
    budget = max(1, int(token_budget))
    txt = text.strip()
    if estimate_tokens(txt) <= budget:
        return txt
    char_budget = max(48, budget * 4)
    trimmed = txt[:char_budget].rstrip()
    if len(trimmed) < len(txt):
        trimmed += " ...<trimmed>"
    return trimmed


def _item_score(item: Mapping[str, object], fallback: float = 0.0) -> float:
    try:
        return float(item.get("score", fallback))
    except (TypeError, ValueError):
        return fallback


def _prepare_memory_items(memory_details: Sequence[Mapping[str, object]], snippet_chars: int) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for idx, item in enumerate(memory_details):
        content = str(item.get("content", ""))
        snippet = _dedupe_lines(content, max_lines=28)
        if not snippet:
            snippet = str(item.get("snippet", "")).strip()
        snippet = snippet[: max(120, int(snippet_chars))]
        items.append(
            {
                "id": str(item.get("id", f"M{idx+1}")),
                "kind": str(item.get("kind", item.get("item_type", "memory"))),
                "title": str(item.get("title", "")),
                "score": _item_score(item, fallback=max(0.0, 1.0 - idx * 0.05)),
                "snippet": snippet,
            }
        )
    items.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return items


def _prepare_repo_items(repo_payload: Mapping[str, object], snippet_chars: int) -> List[Dict[str, object]]:
    chunks = repo_payload.get("chunks") if isinstance(repo_payload, dict) else None
    if not isinstance(chunks, list):
        return []

    prepared: List[Dict[str, object]] = []
    seen_paths = set()
    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            continue
        path = str(chunk.get("path", "")).strip()
        if not path:
            continue
        if path in seen_paths:
            continue
        seen_paths.add(path)
        snippet = _dedupe_lines(str(chunk.get("snippet", "")), max_lines=36)
        if not snippet:
            snippet = _dedupe_lines(str(chunk.get("text", "")), max_lines=36)
        snippet = snippet[: max(180, int(snippet_chars))]
        prepared.append(
            {
                "path": path,
                "start_line": int(chunk.get("start_line", 0) or 0),
                "end_line": int(chunk.get("end_line", 0) or 0),
                "score": _item_score(chunk, fallback=max(0.0, 1.0 - idx * 0.03)),
                "category": str(chunk.get("category", "code")),
                "snippet": snippet,
            }
        )
    prepared.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return prepared


def _take_by_budget(items: Sequence[Mapping[str, object]], token_budget: int, per_item_cap: int, max_items: int) -> List[Dict[str, object]]:
    remaining = max(0, int(token_budget))
    cap = max(24, int(per_item_cap))
    out: List[Dict[str, object]] = []
    for raw in items:
        if len(out) >= max_items:
            break
        if remaining <= 0:
            break
        snippet = str(raw.get("snippet", "")).strip()
        if not snippet:
            continue
        budget_for_item = min(cap, remaining)
        trimmed = _trim_to_token_budget(snippet, budget_for_item)
        tok = estimate_tokens(trimmed)
        if tok <= 0:
            continue
        row = dict(raw)
        row["snippet"] = trimmed
        row["token_estimate"] = tok
        out.append(row)
        remaining -= tok
    return out


def build_prompt_plan(
    *,
    profile: PromptProfile,
    question: str,
    memory_details: Sequence[Mapping[str, object]],
    repo_payload: Mapping[str, object],
    total_budget: int = 1800,
    snippet_chars: int = 1000,
) -> Dict[str, object]:
    total = max(600, int(total_budget))
    header_budget = max(80, int(total * float(profile.budget_ratios.get("header", 0.15))))
    memory_budget = max(100, int(total * float(profile.budget_ratios.get("memory", 0.30))))
    repo_budget = max(100, total - header_budget - memory_budget)

    memory_items = _prepare_memory_items(memory_details, snippet_chars)
    repo_items = _prepare_repo_items(repo_payload, snippet_chars)

    mem_per_item_cap = min(120, max(72, memory_budget // max(1, min(3, len(memory_items) or 1))))
    repo_per_item_cap = min(140, max(88, repo_budget // max(1, min(6, len(repo_items) or 1))))

    selected_memory = _take_by_budget(memory_items, memory_budget, mem_per_item_cap, max_items=3)
    selected_repo = _take_by_budget(repo_items, repo_budget, repo_per_item_cap, max_items=6)

    mem_used = sum(int(item.get("token_estimate", 0)) for item in selected_memory)
    repo_used = sum(int(item.get("token_estimate", 0)) for item in selected_repo)
    header_used = min(header_budget, estimate_tokens(question) + 70)

    return {
        "profile": profile.name,
        "budgets": {
            "total": total,
            "header": header_budget,
            "memory": memory_budget,
            "repo": repo_budget,
        },
        "selected_memory": selected_memory,
        "selected_repo": selected_repo,
        "usage": {
            "header_tokens_est": header_used,
            "memory_tokens_est": mem_used,
            "repo_tokens_est": repo_used,
            "total_tokens_est": header_used + mem_used + repo_used,
        },
        "dropped": {
            "memory": max(0, len(memory_items) - len(selected_memory)),
            "repo": max(0, len(repo_items) - len(selected_repo)),
        },
    }
