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
        nxt = payload.get("next_input", {})
        self.assertIn("run-target", str(nxt.get("command_template_zh", "")))
        self.assertIn("/ABS/PATH/TO/OTHER_PROJECT", str(nxt.get("command_template_zh", "")))
        self.assertIn("--project my-project", str(nxt.get("command_template_zh", "")))
        self.assertIn("--root \"/ABS/PATH/TO/OTHER_PROJECT\"", str(nxt.get("command_template_py_zh", "")))
        self.assertIn("--mapping-debug", str(nxt.get("command_template_py_zh", "")))
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


if __name__ == "__main__":
    unittest.main()
