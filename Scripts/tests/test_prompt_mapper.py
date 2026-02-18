from __future__ import annotations

import pathlib
import sys
import unittest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prompt_mapper import map_prompt_to_profile
from codex_mem import parse_natural_query


class PromptMapperTests(unittest.TestCase):
    def test_runtime_parser_detects_onboarding_intent(self) -> None:
        parsed = parse_natural_query("学习这个项目，给我prompt并跑target project")
        self.assertEqual(parsed.get("intent"), "onboarding")

    def test_profile_accuracy_on_labeled_cases(self) -> None:
        cases = [
            ("Learn this project architecture and entrypoints", "onboarding"),
            ("学习这个项目，梳理目标、架构和主流程", "onboarding"),
            ("给我一段学习任何一个项目的prompt，走target project", "onboarding"),
            ("优化这个项目，先做首读和模块地图", "onboarding"),
            ("What changed in the parser yesterday?", "daily_qa"),
            ("Explain why this query is slow", "daily_qa"),
            ("Triage regression after hotfix and find root cause", "bug_triage"),
            ("线上 incident 报错，先定位根因", "bug_triage"),
            ("Implement compact prompt renderer and keep compatibility", "implementation"),
            ("请实现这个功能并改代码", "implementation"),
            ("How do we run the smoke test?", "daily_qa"),
            ("Need to fix crash and build repro path", "bug_triage"),
        ]

        hits = 0
        for question, expected in cases:
            decision = map_prompt_to_profile(
                question,
                parsed_nl={"intent": "general"},
                mapping_fallback="off",
                llm_api_key="",
            )
            if decision.get("profile") == expected:
                hits += 1

        accuracy = hits / len(cases)
        self.assertGreaterEqual(accuracy, 0.90, f"accuracy={accuracy}")

    def test_low_confidence_uses_max_score_policy(self) -> None:
        decision = map_prompt_to_profile(
            "??",
            parsed_nl={"intent": "general"},
            mapping_fallback="off",
            llm_api_key="",
        )
        self.assertIn(decision.get("source"), {"fallback-max-score", "rule"})
        self.assertIn(decision.get("profile"), {"daily_qa", "implementation", "bug_triage", "onboarding"})

    def test_profile_accuracy_with_runtime_parser(self) -> None:
        cases = [
            ("Learn this project architecture and entrypoints", "onboarding"),
            ("学习这个项目，梳理目标、架构和主流程", "onboarding"),
            ("给我一段学习任何一个项目的prompt，走target project", "onboarding"),
            ("What changed in the parser yesterday?", "daily_qa"),
            ("Triage regression after hotfix and find root cause", "bug_triage"),
            ("线上 incident 报错，先定位根因", "bug_triage"),
            ("Implement compact prompt renderer and keep compatibility", "implementation"),
            ("请实现这个功能并改代码", "implementation"),
            ("How do we run the smoke test?", "daily_qa"),
        ]

        hits = 0
        for question, expected in cases:
            decision = map_prompt_to_profile(
                question,
                parsed_nl=parse_natural_query(question),
                mapping_fallback="off",
                llm_api_key="",
            )
            if decision.get("profile") == expected:
                hits += 1

        accuracy = hits / len(cases)
        self.assertGreaterEqual(accuracy, 0.90, f"runtime accuracy={accuracy}")


if __name__ == "__main__":
    unittest.main()
