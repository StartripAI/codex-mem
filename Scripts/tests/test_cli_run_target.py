from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
import unittest


class CliRunTargetTests(unittest.TestCase):
    def test_run_target_executes_ask_on_explicit_root(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        shell_script = repo_root / "Scripts" / "dev_mem.sh"
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
            self.assertEqual(payload.get("executor_mode"), "none")
            self.assertFalse(bool(payload.get("execution_attempted")))
            self.assertIsInstance(payload.get("forced_next_input"), dict)
            self.assertTrue(bool(payload.get("forced_next_input", {}).get("mandatory")))

    def test_run_target_auto_detects_target_root_from_question(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        shell_script = repo_root / "Scripts" / "dev_mem.sh"
        self.assertTrue(shell_script.exists(), f"missing shell script: {shell_script}")

        with tempfile.TemporaryDirectory(prefix="codex_mem_target_auto_") as tmp:
            target_root = pathlib.Path(tmp)
            (target_root / "README.md").write_text("# target\n", encoding="utf-8")
            question = f"learn this project in {target_root} with architecture and persistence"

            cmd = [
                "bash",
                str(shell_script),
                "run-target-auto",
                question,
                "--project",
                "target-auto",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("stage"), "fused_ask")
            self.assertEqual(payload.get("filters", {}).get("project"), "target-auto")

    def test_run_target_auto_returns_required_token_when_unresolved(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        shell_script = repo_root / "Scripts" / "dev_mem.sh"
        self.assertTrue(shell_script.exists(), f"missing shell script: {shell_script}")

        cmd = [
            "bash",
            str(shell_script),
            "run-target-auto",
            "learn this project architecture and risks",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(repo_root))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "TARGET_ROOT_REQUIRED")


if __name__ == "__main__":
    unittest.main()
