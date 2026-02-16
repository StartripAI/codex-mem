from __future__ import annotations

import pathlib
import sys
import unittest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from codex_mem import build_forced_next_input


class ForcedNextInputTests(unittest.TestCase):
    def test_forced_output_includes_command_and_required_fields(self) -> None:
        payload = build_forced_next_input(
            root=pathlib.Path("."),
            profile_name="onboarding",
            coverage_gate={"pass": True, "missing_categories": []},
        )
        self.assertTrue(bool(payload.get("mandatory")))
        required = payload.get("required_output_fields", [])
        self.assertIn("mapping_decision", required)
        self.assertIn("coverage_gate", required)
        self.assertIn("prompt_plan", required)
        self.assertIn("prompt_metrics", required)
        resolution = payload.get("target_root_resolution", {})
        self.assertEqual(str(resolution.get("strategy", "")), "auto-detect-absolute-target-root")
        self.assertEqual(str(resolution.get("blocked_output_token", "")), "TARGET_ROOT_REQUIRED")
        self.assertIn("no_python_fallback", resolution.get("hard_rules", []))
        nxt = payload.get("next_input", {})
        self.assertIn("run-target", str(nxt.get("command_template_zh", "")))
        self.assertIn("/ABS/PATH/TO/OTHER_PROJECT", str(nxt.get("command_template_zh", "")))
        self.assertIn('--project "my-project"', str(nxt.get("command_template_zh", "")))
        self.assertNotIn("command_template_py_zh", nxt)
        self.assertNotIn("command_template_py_en", nxt)
        self.assertEqual(str(nxt.get("callable_prompt_contract", "")), "must_call_codex_mem_only")
        self.assertEqual(str(nxt.get("execution_contract", "")), "must_execute_when_capable")
        self.assertEqual(
            str(nxt.get("learning_completion_contract", "")),
            "no_percent_guess_and_gate_based_completion",
        )
        self.assertEqual(str(nxt.get("learning_complete_token", "")), "LEARNING_COMPLETE")
        self.assertEqual(str(nxt.get("learning_incomplete_token", "")), "INCOMPLETE")
        status_policy = nxt.get("learning_status_policy", {})
        self.assertEqual(
            status_policy.get("allowed_status_tokens"),
            ["INCOMPLETE", "LEARNING_COMPLETE"],
        )
        self.assertTrue(bool(status_policy.get("forbid_numeric_completion")))
        self.assertIn("只能返回状态 token", str(status_policy.get("status_query_rule_zh", "")))
        gate_sections = nxt.get("learning_gate_required_sections", [])
        self.assertIn("north_star", gate_sections)
        self.assertIn("tests", gate_sections)
        self.assertIn("codex-mem run-target", str(nxt.get("callable_prompt_zh", "")))
        self.assertIn("run-target", str(nxt.get("callable_prompt_zh", "")))
        self.assertIn("必须立即执行", str(nxt.get("callable_prompt_zh", "")))
        self.assertIn("学会了吗/完成度多少", str(nxt.get("callable_prompt_zh", "")))
        self.assertIn("LEARNING_COMPLETE", str(nxt.get("callable_prompt_zh", "")))
        forbidden = nxt.get("forbidden_output_patterns", [])
        self.assertIn("non_executable_prompt_only", forbidden)
        self.assertIn("generic_advice_without_codex_mem_call", forbidden)
        self.assertIn("python_direct_entrypoint", forbidden)
        self.assertIn("claim_not_executed_without_attempt", forbidden)
        self.assertIn("completion_percentage_guess", forbidden)
        self.assertIn("completion_numeric_ratio", forbidden)
        self.assertIn("percent_symbol_output", forbidden)
        self.assertIn("TARGET_ROOT_REQUIRED", str(nxt.get("router_prompt_zh", "")))
        self.assertEqual(str(nxt.get("output_if_target_root_missing", "")), "TARGET_ROOT_REQUIRED")
        self.assertEqual(str(nxt.get("output_contract", "")), "single_line_shell_command_only")
        self.assertEqual(payload.get("status"), "ready")

    def test_failed_gate_adds_refine_prompt(self) -> None:
        payload = build_forced_next_input(
            root=pathlib.Path("."),
            profile_name="onboarding",
            coverage_gate={"pass": False, "missing_categories": ["entrypoint", "persistence"]},
        )
        nxt = payload.get("next_input", {})
        self.assertEqual(payload.get("status"), "needs_refine")
        self.assertIn("refine_prompt_zh", nxt)
        self.assertIn("entrypoint, persistence", str(nxt.get("refine_prompt_zh", "")))
        self.assertIn("run-target", str(nxt.get("refine_command_template_zh", "")))
        self.assertNotIn("refine_command_template_py_zh", nxt)


if __name__ == "__main__":
    unittest.main()
