from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
import unittest


class CliRunTargetTests(unittest.TestCase):
    def test_run_target_executes_ask_on_explicit_root(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        shell_script = repo_root / "Scripts" / "codex_mem.sh"
        self.assertTrue(shell_script.exists(), f"missing shell script: {shell_script}")

        with tempfile.TemporaryDirectory(prefix="codex_mem_target_") as tmp:
            target_root = pathlib.Path(tmp)
            (target_root / "README.md").write_text("# target\n", encoding="utf-8")

            cmd = [
                "bash",
                str(shell_script),
                "run-target",
                str(target_root),
                "--project",
                "target-e2e",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)

            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("stage"), "fused_ask")
            self.assertEqual(payload.get("filters", {}).get("project"), "target-e2e")
            self.assertEqual(payload.get("mapping_decision", {}).get("profile"), "onboarding")
            self.assertIsInstance(payload.get("forced_next_input"), dict)
            self.assertTrue(bool(payload.get("forced_next_input", {}).get("mandatory")))


if __name__ == "__main__":
    unittest.main()
