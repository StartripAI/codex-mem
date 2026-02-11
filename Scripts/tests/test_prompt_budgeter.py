from __future__ import annotations

import pathlib
import sys
import unittest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prompt_budgeter import build_prompt_plan
from prompt_profiles import get_prompt_profile


class PromptBudgeterTests(unittest.TestCase):
    def test_budget_plan_caps_and_usage(self) -> None:
        profile = get_prompt_profile("onboarding")
        memory = [
            {
                "id": "E1",
                "kind": "tool",
                "title": "Long event",
                "content": "\n".join(["repeat line"] * 80) + "\n" + "signal line",
                "score": 0.9,
            },
            {
                "id": "O2",
                "kind": "learning",
                "title": "Observation",
                "content": "insight " * 400,
                "score": 0.8,
            },
        ]
        repo_payload = {
            "chunks": [
                {
                    "path": "App/Main.swift",
                    "start_line": 1,
                    "end_line": 120,
                    "score": 0.95,
                    "snippet": "main flow " * 500,
                    "category": "entrypoint",
                },
                {
                    "path": "App/Main.swift",
                    "start_line": 1,
                    "end_line": 120,
                    "score": 0.90,
                    "snippet": "duplicate should be removed",
                    "category": "entrypoint",
                },
                {
                    "path": "Services/DatabaseBootstrapper.swift",
                    "start_line": 1,
                    "end_line": 180,
                    "score": 0.88,
                    "snippet": "persistence path " * 500,
                    "category": "persistence",
                },
            ]
        }

        plan = build_prompt_plan(
            profile=profile,
            question="Learn this project quickly",
            memory_details=memory,
            repo_payload=repo_payload,
            total_budget=1200,
            snippet_chars=600,
        )

        self.assertEqual(plan.get("profile"), "onboarding")
        usage = plan.get("usage", {})
        self.assertLessEqual(int(usage.get("total_tokens_est", 0)), 1200)

        selected_repo = plan.get("selected_repo", [])
        paths = [str(item.get("path")) for item in selected_repo]
        self.assertEqual(len(paths), len(set(paths)))

        selected_memory = plan.get("selected_memory", [])
        self.assertGreaterEqual(len(selected_memory), 1)
        for item in selected_memory:
            self.assertTrue(str(item.get("snippet", "")).strip())


if __name__ == "__main__":
    unittest.main()
