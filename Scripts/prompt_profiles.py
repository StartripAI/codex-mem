#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
from typing import Dict, Mapping, Sequence


@dataclasses.dataclass(frozen=True)
class PromptProfile:
    name: str
    display_name: str
    system_capsule: str
    goal_template: str
    defaults: Mapping[str, float | int]
    budget_ratios: Mapping[str, float]
    coverage_required: Sequence[str]


PROFILES: Dict[str, PromptProfile] = {
    "onboarding": PromptProfile(
        name="onboarding",
        display_name="Project Onboarding",
        system_capsule=(
            "You are grounding on a new repository. Prioritize architectural correctness over speed. "
            "Do not guess when evidence is incomplete."
        ),
        goal_template=(
            "Build a reliable project model: north star, architecture map, entrypoint/startup, "
            "main flow, persistence/output, and top risks."
        ),
        defaults={
            "search_limit": 8,
            "detail_limit": 4,
            "code_top_k": 10,
            "code_module_limit": 8,
            "alpha": 0.65,
        },
        budget_ratios={"header": 0.15, "memory": 0.20, "repo": 0.65},
        coverage_required=("entrypoint", "persistence", "ai_generation"),
    ),
    "daily_qa": PromptProfile(
        name="daily_qa",
        display_name="Daily Q&A",
        system_capsule=(
            "You are answering incrementally. Use the smallest sufficient evidence set and keep response direct."
        ),
        goal_template="Answer the user question precisely using current memory + repo evidence.",
        defaults={
            "search_limit": 6,
            "detail_limit": 3,
            "code_top_k": 6,
            "code_module_limit": 4,
            "alpha": 0.70,
        },
        budget_ratios={"header": 0.15, "memory": 0.45, "repo": 0.40},
        coverage_required=(),
    ),
    "bug_triage": PromptProfile(
        name="bug_triage",
        display_name="Bug Triage",
        system_capsule=(
            "You are triaging an issue. Prioritize reproducibility, failure modes, and minimal-risk fix options."
        ),
        goal_template=(
            "Identify probable root cause, reproduction path, and verification checklist with evidence."
        ),
        defaults={
            "search_limit": 12,
            "detail_limit": 6,
            "code_top_k": 12,
            "code_module_limit": 6,
            "alpha": 0.65,
        },
        budget_ratios={"header": 0.15, "memory": 0.35, "repo": 0.50},
        coverage_required=(),
    ),
    "implementation": PromptProfile(
        name="implementation",
        display_name="Implementation",
        system_capsule=(
            "You are implementing a change. Keep patches minimal, maintain compatibility unless explicitly asked otherwise, "
            "and surface assumptions."
        ),
        goal_template="Implement safely with clear impact boundaries and validation evidence.",
        defaults={
            "search_limit": 8,
            "detail_limit": 5,
            "code_top_k": 10,
            "code_module_limit": 6,
            "alpha": 0.70,
        },
        budget_ratios={"header": 0.15, "memory": 0.30, "repo": 0.55},
        coverage_required=(),
    ),
}

DEFAULT_PROFILE = "daily_qa"


def get_prompt_profile(name: str) -> PromptProfile:
    key = (name or "").strip().lower()
    if key in PROFILES:
        return PROFILES[key]
    return PROFILES[DEFAULT_PROFILE]


def list_profiles() -> Sequence[str]:
    return tuple(PROFILES.keys())
