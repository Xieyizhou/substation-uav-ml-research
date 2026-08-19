"""Executable UI state and static integration contracts."""

import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/inspection/static"


class SandboxUIStateTests(unittest.TestCase):
    def test_hash_state_and_guidance_execute_in_node(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is not available")
        script = f"""
const fs=require('fs'),vm=require('vm');
vm.runInThisContext(fs.readFileSync({json.dumps(str(STATIC / 'ui_state.js'))},'utf8'));
const value={{
  workbench: SandboxUIState.tabFromHash('#experiments'),
  operator: SandboxUIState.tabFromHash('#operator'),
  invalid: SandboxUIState.tabFromHash('#unknown'),
  guidance: SandboxUIState.failureGuidance('flight_timeout')
}};
process.stdout.write(JSON.stringify(value));
"""
        result = subprocess.run(
            [node, "-e", script], check=True, capture_output=True, text=True
        )
        value = json.loads(result.stdout)
        self.assertEqual(value["workbench"], "experiments")
        self.assertEqual(value["operator"], "operator")
        self.assertEqual(value["invalid"], "overview")
        self.assertIn("flight log", value["guidance"])

    def test_polling_does_not_activate_or_reload_a_tab(self):
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        body = app.split("async function refreshOperator()", 1)[1].split(
            "async function loadJobLog", 1
        )[0]
        self.assertNotIn("activateTab", body)
        self.assertNotIn("location", body)
        self.assertIn("SandboxUIState.tabFromHash", app)

    def test_first_run_and_inference_controls_are_present(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        for identifier in (
            'id="setup-journey"', 'id="setup-artifacts"',
            'id="workbench-infer"', 'id="inference-primary"',
            'id="inference-comparison"',
        ):
            self.assertIn(identifier, html)


if __name__ == "__main__":
    unittest.main()
