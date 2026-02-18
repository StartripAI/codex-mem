from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


class CliE2EForcedNextInputTests(unittest.TestCase):
    def test_cli_ask_always_returns_forced_next_input(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        script = repo_root / "Scripts" / "codex_mem.py"
        self.assertTrue(script.exists(), f"missing script: {script}")

        with tempfile.TemporaryDirectory(prefix="codex_mem_e2e_") as tmp:
            root = pathlib.Path(tmp)
            expected_root = str(root.resolve())
            (root / "README.md").write_text("# temp project\n", encoding="utf-8")

            cmd = [
                sys.executable,
                str(script),
                "--root",
                str(root),
                "ask",
                "学习这个项目：北极星、架构、模块地图、入口、持久化、主流程、风险",
                "--project",
                "e2e",
                "--mapping-debug",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads(proc.stdout)
            forced = payload.get("forced_next_input")
            self.assertIsInstance(forced, dict)
            self.assertTrue(bool(forced.get("mandatory")))
            self.assertIn("required_output_fields", forced)
            self.assertIn("next_input", forced)
            self.assertEqual(str(payload.get("executor_mode", "")), "none")
            self.assertFalse(bool(payload.get("execution_attempted")))
            self.assertIsInstance(payload.get("execution_result"), dict)
            self.assertEqual(str(payload.get("execution_result", {}).get("status", "")), "skipped")
            self.assertIsInstance(payload.get("task_spec"), dict)
            self.assertIsInstance(payload.get("execution_plan"), dict)
            self.assertIsInstance(payload.get("coverage_9_section"), dict)
            self.assertIn("pass", payload.get("coverage_9_section", {}))
            self.assertIsInstance(payload.get("evidence_stats"), dict)
            self.assertIsInstance(payload.get("coverage_recovery"), list)
            self.assertIsInstance(payload.get("memory_runtime_layers"), list)
            self.assertEqual(len(payload.get("memory_runtime_layers", [])), 6)

            required_fields = forced.get("required_output_fields", [])
            self.assertIn("mapping_decision", required_fields)
            self.assertIn("coverage_gate", required_fields)
            self.assertIn("prompt_plan", required_fields)
            self.assertIn("prompt_metrics", required_fields)

            next_input = forced.get("next_input", {})
            self.assertIn("run-target", str(next_input.get("command_template_zh", "")))
            self.assertIn(expected_root, str(next_input.get("command_template_zh", "")))
            self.assertIn("--project", str(next_input.get("command_template_zh", "")))
            self.assertNotIn("command_template_py_zh", next_input)
            self.assertNotIn("command_template_py_en", next_input)
            self.assertEqual(str(next_input.get("callable_prompt_contract", "")), "must_call_dev_mem_only")
            self.assertEqual(str(next_input.get("execution_contract", "")), "must_execute_when_capable")
            self.assertEqual(
                str(next_input.get("learning_completion_contract", "")),
                "completion_ratio_allowed_with_evidence_basis",
            )
            self.assertEqual(str(next_input.get("learning_complete_token", "")), "LEARNING_COMPLETE")
            self.assertEqual(str(next_input.get("learning_partial_token", "")), "PARTIAL")
            self.assertEqual(str(next_input.get("learning_incomplete_token", "")), "INCOMPLETE")
            status_policy = next_input.get("learning_status_policy", {})
            self.assertEqual(
                status_policy.get("allowed_status_tokens"),
                ["INCOMPLETE", "PARTIAL", "LEARNING_COMPLETE"],
            )
            self.assertFalse(bool(status_policy.get("forbid_numeric_completion")))
            self.assertTrue(bool(status_policy.get("incomplete_only_when_no_coverage")))
            self.assertEqual(str(status_policy.get("completion_query_mode", "")), "progress_report_with_completion_ratio")
            self.assertTrue(bool(status_policy.get("use_partial_by_default_when_progress_exists")))
            depth_targets = next_input.get("learning_depth_targets", {})
            self.assertEqual(str(depth_targets.get("target_completion_min", "")), "95%")
            self.assertEqual(int(depth_targets.get("min_evidence_per_section", 0)), 3)
            self.assertEqual(
                str(depth_targets.get("completion_reporting", "")),
                "percentage_or_range_with_evidence_basis",
            )
            benchmark = next_input.get("benchmark_targets", {})
            self.assertEqual(int(benchmark.get("coverage_min_pct", 0)), 90)
            self.assertEqual(int(benchmark.get("efficiency_gain_min_pct", 0)), 30)
            self.assertEqual(str(benchmark.get("efficiency_metric", "")), "time_plus_token")
            self.assertIn("run-target", str(next_input.get("callable_prompt_zh", "")))
            self.assertIn("TARGET_ROOT_REQUIRED", str(next_input.get("callable_prompt_zh", "")))
            self.assertIn("LEARNING_COMPLETE/PARTIAL/INCOMPLETE", str(next_input.get("callable_prompt_zh", "")))
            self.assertIn("有进展默认 PARTIAL", str(next_input.get("callable_prompt_zh", "")))
            self.assertNotIn("用户", str(next_input.get("callable_prompt_zh", "")))
            self.assertTrue(bool(next_input.get("backend_rules_locked")))
            self.assertIn("MECE 七部分", str(next_input.get("backend_sop_zh", "")))
            self.assertIn("每部分至少 3 条证据", str(next_input.get("backend_sop_zh", "")))
            self.assertIn("覆盖率 >= 90%", str(next_input.get("backend_sop_zh", "")))
            self.assertIn("优先返回项目用途 + 完成度估计", str(next_input.get("backend_sop_zh", "")))
            completion_policy = next_input.get("completion_response_policy", {})
            self.assertEqual(str(completion_policy.get("default_when_progress_exists", "")), "PARTIAL")
            forbidden = next_input.get("forbidden_output_patterns", [])
            self.assertIn("non_executable_prompt_only", forbidden)
            self.assertIn("claim_not_executed_without_attempt", forbidden)
            self.assertNotIn("completion_percentage_guess", forbidden)
            self.assertNotIn("completion_numeric_ratio", forbidden)
            self.assertNotIn("percent_symbol_output", forbidden)
            self.assertIn("shallow_one_pass_summary", forbidden)
            self.assertIn("always_incomplete_on_progress_query", forbidden)
            self.assertEqual(str(next_input.get("output_if_target_root_missing", "")), "TARGET_ROOT_REQUIRED")
            self.assertIn("TARGET_ROOT_REQUIRED", str(next_input.get("router_prompt_zh", "")))

            resolution = forced.get("target_root_resolution", {})
            self.assertEqual(str(resolution.get("strategy", "")), "auto-detect-absolute-target-root")
            self.assertEqual(str(resolution.get("detected_source", "")), "runtime_root")
            self.assertEqual(str(resolution.get("detected_target_root", "")), expected_root)
            self.assertIn("no_python_fallback", resolution.get("hard_rules", []))

            # In an isolated temp repo without Scripts/repo_knowledge.py, onboarding coverage should fail,
            # and refine instructions must still be returned.
            self.assertEqual(str(forced.get("completion_default_status", "")), "INCOMPLETE")
            self.assertEqual(forced.get("status"), "needs_refine")
            self.assertIn("refine_prompt_zh", next_input)
            self.assertIn("recommended_next_action", payload)


if __name__ == "__main__":
    unittest.main()
