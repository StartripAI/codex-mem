#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Mapping, Sequence, Tuple

from prompt_profiles import list_profiles

LOW_CONFIDENCE_THRESHOLD = 0.55

_RULE_KEYWORDS: Dict[str, Sequence[str]] = {
    "onboarding": (
        "learn",
        "onboard",
        "architecture",
        "module",
        "entrypoint",
        "main flow",
        "persistence",
        "north star",
        "学习",
        "架构",
        "模块",
        "入口",
        "主流程",
        "持久化",
        "落库",
        "北极星",
    ),
    "bug_triage": (
        "bug",
        "fix",
        "regression",
        "incident",
        "crash",
        "error",
        "triage",
        "hotfix",
        "故障",
        "报错",
        "异常",
        "修复",
    ),
    "implementation": (
        "implement",
        "implementation",
        "code",
        "patch",
        "refactor",
        "apply",
        "change",
        "feature",
        "实现",
        "改造",
        "重构",
        "编码",
        "修改",
    ),
    "daily_qa": (
        "question",
        "why",
        "how",
        "what",
        "explain",
        "summary",
        "qa",
        "问",
        "解释",
        "总结",
        "说明",
    ),
}


def _ordered_unique(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        txt = str(value or "").strip()
        if not txt:
            continue
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(txt)
    return out


def _safe_retrieval_hints(question: str) -> List[str]:
    try:
        from repo_knowledge import retrieval_hints  # lazy import to avoid hard runtime dependency

        hints = retrieval_hints(question)
        if isinstance(hints, list):
            return [str(v) for v in hints if str(v).strip()]
    except Exception:
        pass
    return []


def _rule_scores(question: str, parsed_nl: Mapping[str, object] | None) -> Dict[str, float]:
    text = str(question or "")
    lower = text.lower()
    scores: Dict[str, float] = {name: 0.0 for name in list_profiles()}

    for profile, keywords in _RULE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lower or keyword in text:
                if " " in keyword:
                    scores[profile] += 1.2
                else:
                    scores[profile] += 0.7

    # Intents from codex_mem.parse_natural_query
    if parsed_nl:
        intent = str(parsed_nl.get("intent", "")).strip().lower()
        if intent == "bugfix":
            scores["bug_triage"] += 2.5
        elif intent == "refactor":
            scores["implementation"] += 1.6
        elif intent == "release":
            scores["implementation"] += 1.1
        elif intent == "test":
            scores["implementation"] += 0.8
            scores["daily_qa"] += 0.6

    hints = _safe_retrieval_hints(text)
    if hints:
        hints_join = " ".join(hints).lower()
        if any(k in hints_join for k in ("north star", "entrypoint", "architecture", "persistence")):
            scores["onboarding"] += 2.0
        if any(k in hints_join for k in ("risk", "failure", "incident")):
            scores["bug_triage"] += 0.8

    # Short prompts often indicate daily QA unless strong other signals exist.
    if len(text.strip()) <= 24:
        scores["daily_qa"] += 0.6

    if max(scores.values()) <= 0:
        scores["daily_qa"] = 1.0

    return scores


def _confidence_from_scores(scores: Mapping[str, float]) -> Tuple[str, float]:
    ranked = sorted(((k, float(v)) for k, v in scores.items()), key=lambda x: x[1], reverse=True)
    top_profile, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = 0.0 if top_score <= 0 else max(0.0, (top_score - second_score) / max(top_score, 1e-6))
    strength = min(1.0, top_score / 4.0)
    confidence = 0.42 + 0.36 * gap + 0.22 * strength
    confidence = max(0.0, min(0.97, confidence))
    return top_profile, confidence


def _extract_json_object(raw: str) -> Mapping[str, object] | None:
    txt = (raw or "").strip()
    if not txt:
        return None
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj2 = json.loads(m.group(0))
        if isinstance(obj2, dict):
            return obj2
    except Exception:
        return None
    return None


def _call_openai_router(
    question: str,
    *,
    api_key: str,
    model: str,
    timeout_sec: int,
    rule_scores: Mapping[str, float],
) -> Mapping[str, object] | None:
    if not api_key:
        return None

    system = (
        "You are a strict routing classifier. "
        "Classify user intent into one profile: onboarding, daily_qa, bug_triage, implementation. "
        "Return JSON only: {\"profile\":...,\"confidence\":0..1,\"reason\":...}."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Rule scores:\n{json.dumps(dict(rule_scores), ensure_ascii=False)}\n\n"
        "Choose the best profile."
    )

    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None

    try:
        data = json.loads(raw)
    except Exception:
        return None

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else ""
    if not isinstance(content, str):
        return None
    return _extract_json_object(content)


def map_prompt_to_profile(
    question: str,
    *,
    parsed_nl: Mapping[str, object] | None,
    mapping_fallback: str = "auto",
    llm_api_key: str | None = None,
    llm_model: str | None = None,
    llm_timeout_sec: int = 8,
) -> Dict[str, object]:
    scores = _rule_scores(question, parsed_nl)
    rule_top_profile, rule_confidence = _confidence_from_scores(scores)

    final_profile = rule_top_profile
    final_confidence = rule_confidence
    source = "rule"
    llm_attempted = False
    llm_used = False
    llm_error = ""

    fallback_mode = str(mapping_fallback or "auto").strip().lower()
    llm_key = (llm_api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
    model = (llm_model or os.environ.get("OPENAI_PROMPT_ROUTER_MODEL", "gpt-4o-mini")).strip() or "gpt-4o-mini"

    if fallback_mode == "auto" and rule_confidence < LOW_CONFIDENCE_THRESHOLD and llm_key:
        llm_attempted = True
        llm_obj = _call_openai_router(
            question,
            api_key=llm_key,
            model=model,
            timeout_sec=max(2, int(llm_timeout_sec)),
            rule_scores=scores,
        )
        if isinstance(llm_obj, dict):
            profile = str(llm_obj.get("profile", "")).strip().lower()
            conf_raw = llm_obj.get("confidence", 0.0)
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = 0.0
            if profile in set(list_profiles()):
                final_profile = profile
                final_confidence = max(0.0, min(1.0, conf)) if conf > 0 else rule_confidence
                source = "llm"
                llm_used = True
            else:
                llm_error = "invalid_profile_from_llm"
        else:
            llm_error = "llm_unavailable_or_invalid"

    low_confidence = final_confidence < LOW_CONFIDENCE_THRESHOLD
    if low_confidence and source != "llm":
        # User-required policy: choose highest score directly on low confidence.
        final_profile = rule_top_profile
        source = "fallback-max-score"

    rounded_scores = {k: round(float(v), 4) for k, v in scores.items()}

    return {
        "profile": final_profile,
        "confidence": round(float(final_confidence), 4),
        "low_confidence": bool(low_confidence),
        "source": source,
        "threshold": LOW_CONFIDENCE_THRESHOLD,
        "profile_scores": rounded_scores,
        "rule_top_profile": rule_top_profile,
        "rule_confidence": round(float(rule_confidence), 4),
        "llm_attempted": llm_attempted,
        "llm_used": llm_used,
        "llm_error": llm_error,
    }
