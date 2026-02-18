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
        self.assertEqual(str(nxt.get("callable_prompt_contract", "")), "must_call_dev_mem_only")
        self.assertEqual(str(nxt.get("execution_contract", "")), "must_execute_when_capable")
        self.assertEqual(
            str(nxt.get("learning_completion_contract", "")),
            "completion_ratio_allowed_with_evidence_basis",
        )
        self.assertEqual(str(nxt.get("learning_complete_token", "")), "LEARNING_COMPLETE")
        self.assertEqual(str(nxt.get("learning_partial_token", "")), "PARTIAL")
        self.assertEqual(str(nxt.get("learning_incomplete_token", "")), "INCOMPLETE")
        status_policy = nxt.get("learning_status_policy", {})
        self.assertEqual(
            status_policy.get("allowed_status_tokens"),
            ["INCOMPLETE", "PARTIAL", "LEARNING_COMPLETE"],
        )
        self.assertFalse(bool(status_policy.get("forbid_numeric_completion")))
        self.assertEqual(str(status_policy.get("completion_query_mode", "")), "progress_report_with_completion_ratio")
        self.assertTrue(bool(status_policy.get("use_partial_by_default_when_progress_exists")))
        self.assertTrue(bool(status_policy.get("incomplete_only_when_no_coverage")))
        self.assertIn("项目用途", str(status_policy.get("status_query_rule_zh", "")))
        self.assertIn("仅在证据严重不足时返回 INCOMPLETE", str(status_policy.get("status_query_rule_zh", "")))
        depth_targets = nxt.get("learning_depth_targets", {})
        self.assertEqual(str(depth_targets.get("target_completion_min", "")), "95%")
        self.assertEqual(int(depth_targets.get("min_evidence_per_section", 0)), 3)
        self.assertEqual(depth_targets.get("must_read_order"), ["docs_first", "code_second"])
        self.assertTrue(bool(depth_targets.get("must_continue_on_gaps")))
        self.assertEqual(
            str(depth_targets.get("completion_reporting", "")),
            "percentage_or_range_with_evidence_basis",
        )
        benchmark = nxt.get("benchmark_targets", {})
        self.assertEqual(int(benchmark.get("coverage_min_pct", 0)), 90)
        self.assertEqual(int(benchmark.get("efficiency_gain_min_pct", 0)), 30)
        self.assertEqual(str(benchmark.get("efficiency_metric", "")), "time_plus_token")
        self.assertEqual(str(benchmark.get("result_honesty", "")), "report_as_is")
        gate_sections = nxt.get("learning_gate_required_sections", [])
        self.assertIn("north_star", gate_sections)
        self.assertIn("tests", gate_sections)
        self.assertIn("dev-mem run-target", str(nxt.get("callable_prompt_zh", "")))
        self.assertIn("run-target", str(nxt.get("callable_prompt_zh", "")))
        self.assertIn("TARGET_ROOT_REQUIRED", str(nxt.get("callable_prompt_zh", "")))
        self.assertIn("LEARNING_COMPLETE/PARTIAL/INCOMPLETE", str(nxt.get("callable_prompt_zh", "")))
        self.assertIn("有进展默认 PARTIAL", str(nxt.get("callable_prompt_zh", "")))
        self.assertNotIn("用户", str(nxt.get("callable_prompt_zh", "")))
        self.assertTrue(bool(nxt.get("backend_rules_locked")))
        self.assertIn("MECE 七部分", str(nxt.get("backend_sop_zh", "")))
        self.assertIn("每部分至少 3 条证据", str(nxt.get("backend_sop_zh", "")))
        self.assertIn("先文档后代码与测试", str(nxt.get("backend_sop_zh", "")))
        self.assertIn("覆盖率 >= 90%", str(nxt.get("backend_sop_zh", "")))
        self.assertIn("效率提升 >= 30%", str(nxt.get("backend_sop_zh", "")))
        self.assertIn("优先返回项目用途 + 完成度估计", str(nxt.get("backend_sop_zh", "")))
        completion_policy = nxt.get("completion_response_policy", {})
        self.assertEqual(str(completion_policy.get("default_when_progress_exists", "")), "PARTIAL")
        self.assertEqual(
            str(completion_policy.get("incomplete_only_when", "")),
            "severely_insufficient_evidence",
        )
        forbidden = nxt.get("forbidden_output_patterns", [])
        self.assertIn("non_executable_prompt_only", forbidden)
        self.assertIn("generic_advice_without_dev_mem_call", forbidden)
        self.assertIn("python_direct_entrypoint", forbidden)
        self.assertIn("claim_not_executed_without_attempt", forbidden)
        self.assertNotIn("completion_percentage_guess", forbidden)
        self.assertNotIn("completion_numeric_ratio", forbidden)
        self.assertNotIn("percent_symbol_output", forbidden)
        self.assertIn("shallow_one_pass_summary", forbidden)
        self.assertIn("always_incomplete_on_progress_query", forbidden)
        self.assertIn("TARGET_ROOT_REQUIRED", str(nxt.get("router_prompt_zh", "")))
        self.assertEqual(str(nxt.get("output_if_target_root_missing", "")), "TARGET_ROOT_REQUIRED")
        self.assertEqual(str(nxt.get("output_contract", "")), "single_line_shell_command_only")
        self.assertEqual(str(payload.get("completion_default_status", "")), "LEARNING_COMPLETE")
        self.assertEqual(payload.get("status"), "ready")

    def test_failed_gate_adds_refine_prompt(self) -> None:
        payload = build_forced_next_input(
            root=pathlib.Path("."),
            profile_name="onboarding",
            coverage_gate={"pass": False, "missing_categories": ["entrypoint", "persistence"]},
        )
        nxt = payload.get("next_input", {})
        self.assertEqual(str(payload.get("completion_default_status", "")), "INCOMPLETE")
        self.assertEqual(payload.get("status"), "needs_refine")
        self.assertIn("refine_prompt_zh", nxt)
        self.assertIn("entrypoint, persistence", str(nxt.get("refine_prompt_zh", "")))
        self.assertIn("run-target", str(nxt.get("refine_command_template_zh", "")))
        self.assertNotIn("refine_command_template_py_zh", nxt)

    def test_partial_status_when_some_coverage_exists(self) -> None:
        payload = build_forced_next_input(
            root=pathlib.Path("."),
            profile_name="onboarding",
            coverage_gate={
                "pass": False,
                "required_categories": ["entrypoint", "persistence", "ai_generation"],
                "present_categories": ["entrypoint"],
                "missing_categories": ["persistence", "ai_generation"],
            },
        )
        self.assertEqual(str(payload.get("completion_default_status", "")), "PARTIAL")
        self.assertEqual(str(payload.get("status", "")), "partial_ready")


if __name__ == "__main__":
    unittest.main()
