import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("train_ldp.sh", "train_tr_ranker.sh", "train_v_ranker.sh", "train_wm.sh")


class TrainingScriptsTest(unittest.TestCase):
    def run_script(self, script, *args):
        with tempfile.TemporaryDirectory() as directory:
            stub = Path(directory) / "python"
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            stub.chmod(0o700)
            env = {**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"]}
            return subprocess.run(["bash", str(ROOT / "scripts" / script), *args],
                                  env=env, text=True, capture_output=True)

    def test_default_single_domain_needs_no_extra_options(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result = self.run_script(script, "--domain", "navigation")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("weights/", result.stdout)

    def test_all_runs_four_separate_domains(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result = self.run_script(script, "--domain", "all")
                self.assertEqual(result.returncode, 0, result.stderr)
                for domain in ("navigation", "driving", "gaming", "robot"):
                    self.assertIn("/" + domain + "/", result.stdout)

    def test_nonempty_options_are_still_forwarded(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result = self.run_script(script, "--domain", "navigation",
                                         "--learning-rate", "0.0002")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines()[-2:],
                                 ["--learning-rate", "0.0002"])


if __name__ == "__main__":
    unittest.main()
