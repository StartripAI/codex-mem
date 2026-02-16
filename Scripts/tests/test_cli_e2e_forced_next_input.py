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

            required_fields = forced.get("required_output_fields", [])
            self.assertIn("mapping_decision", required_fields)
            self.assertIn("coverage_gate", required_fields)
            self.assertIn("prompt_plan", required_fields)
            self.assertIn("prompt_metrics", required_fields)

            next_input = forced.get("next_input", {})
            self.assertIn("run-target", str(next_input.get("command_template_zh", "")))
            self.assertIn(expected_root, str(next_input.get("command_template_zh", "")))
            self.assertIn("--project", str(next_input.get("command_template_zh", "")))
            self.assertIn(f'--root "{expected_root}"', str(next_input.get("command_template_py_zh", "")))
            self.assertIn("--mapping-debug", str(next_input.get("command_template_py_zh", "")))
            self.assertEqual(str(next_input.get("output_if_target_root_missing", "")), "TARGET_ROOT_REQUIRED")
            self.assertIn("TARGET_ROOT_REQUIRED", str(next_input.get("router_prompt_zh", "")))

            resolution = forced.get("target_root_resolution", {})
            self.assertEqual(str(resolution.get("strategy", "")), "auto-detect-absolute-target-root")
            self.assertEqual(str(resolution.get("detected_source", "")), "runtime_root")
            self.assertEqual(str(resolution.get("detected_target_root", "")), expected_root)

            # In an isolated temp repo without Scripts/repo_knowledge.py, onboarding coverage should fail,
            # and refine instructions must still be returned.
            self.assertEqual(forced.get("status"), "needs_refine")
            self.assertIn("refine_prompt_zh", next_input)
            self.assertIn("recommended_next_action", payload)


if __name__ == "__main__":
    unittest.main()
