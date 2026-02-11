#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Mapping, Sequence

from prompt_profiles import PromptProfile


def _line(text: str) -> str:
    return (text or "").strip()


def render_compact_prompt(
    *,
    question: str,
    profile: PromptProfile,
    mapping_decision: Mapping[str, object],
    prompt_plan: Mapping[str, object],
    coverage_gate: Mapping[str, object],
) -> str:
    lines: List[str] = []

    confidence = mapping_decision.get("confidence", 0)
    source = mapping_decision.get("source", "rule")

    lines.append("System Capsule:")
    lines.append(f"- {profile.system_capsule}")
    lines.append("- Cite evidence IDs/paths. If evidence is insufficient, state missing evidence explicitly.")
    lines.append("")

    lines.append("Task Profile:")
    lines.append(f"- profile: {profile.name} ({profile.display_name})")
    lines.append(f"- mapping: source={source} confidence={confidence}")
    lines.append("")

    lines.append("Goal:")
    lines.append(f"- {profile.goal_template}")
    lines.append(f"- user_question: {question}")
    lines.append("")

    lines.append("Memory Evidence:")
    mem_items = prompt_plan.get("selected_memory") if isinstance(prompt_plan, Mapping) else None
    if isinstance(mem_items, list) and mem_items:
        for idx, item in enumerate(mem_items, start=1):
            lines.append(f"[M{idx}] {item.get('id')} {item.get('kind')} {item.get('title')}")
            lines.append(_line(str(item.get("snippet", ""))))
            lines.append("")
    else:
        lines.append("- <none>")
        lines.append("")

    lines.append("Repo Evidence:")
    repo_items = prompt_plan.get("selected_repo") if isinstance(prompt_plan, Mapping) else None
    if isinstance(repo_items, list) and repo_items:
        for idx, item in enumerate(repo_items, start=1):
            path = item.get("path", "")
            start = item.get("start_line", "")
            end = item.get("end_line", "")
            category = item.get("category", "code")
            lines.append(f"[R{idx}] {path}:{start}-{end} category={category}")
            lines.append(_line(str(item.get("snippet", ""))))
            lines.append("")
    else:
        lines.append("- <none>")
        lines.append("")

    lines.append("Coverage Gate:")
    required = coverage_gate.get("required_categories", []) if isinstance(coverage_gate, Mapping) else []
    present = coverage_gate.get("present_categories", []) if isinstance(coverage_gate, Mapping) else []
    gate_pass = coverage_gate.get("pass", False) if isinstance(coverage_gate, Mapping) else False
    lines.append(f"- required: {', '.join(required) if required else '<none>'}")
    lines.append(f"- present: {', '.join(present) if present else '<none>'}")
    lines.append(f"- pass: {bool(gate_pass)}")
    lines.append("")

    lines.append("Missing Evidence:")
    missing = coverage_gate.get("missing_categories", []) if isinstance(coverage_gate, Mapping) else []
    if isinstance(missing, list) and missing:
        for item in missing:
            lines.append(f"- {item}")
    else:
        lines.append("- <none>")

    return "\n".join(lines).strip()
