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
  workbench: SandboxUIState.routeFromHash('#experiments'),
  operator: SandboxUIState.routeFromHash('#operator'),
  lidar: SandboxUIState.routeFromHash('#preflight'),
  recording: SandboxUIState.routeFromHash('#frames'),
  invalid: SandboxUIState.routeFromHash('#unknown'),
  guidance: SandboxUIState.failureGuidance('flight_timeout')
}};
process.stdout.write(JSON.stringify(value));
"""
        result = subprocess.run(
            [node, "-e", script], check=True, capture_output=True, text=True
        )
        value = json.loads(result.stdout)
        self.assertEqual(value["workbench"], "model/train")
        self.assertEqual(value["operator"], "activity/jobs")
        self.assertEqual(value["lidar"], "results/lidar")
        self.assertEqual(value["recording"], "fly/recordings")
        self.assertEqual(value["invalid"], "model/datasets")
        self.assertIn("flight log", value["guidance"])

    def test_polling_does_not_activate_or_reload_a_tab(self):
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        body = app.split("async function refreshOperator()", 1)[1].split(
            "async function loadJobLog", 1
        )[0]
        self.assertNotIn("activateRoute", body)
        self.assertNotIn("location", body)
        navigation = (STATIC / "navigation.js").read_text(encoding="utf-8")
        self.assertIn("SandboxUIState.routeFromHash", navigation)
        self.assertIn("hashchange", navigation)

    def test_task_navigation_and_inference_controls_are_present(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        for identifier in (
            'id="setup-journey"', 'id="setup-artifacts"',
            'id="workbench-infer"', 'id="inference-primary"',
            'id="inference-comparison"', 'class="workspace-sidebar"',
            'data-route="fly/run"', 'data-route="model/datasets"',
            'data-route="results/visual"', 'data-route="activity/jobs"',
            'id="dataset-class-summary"', 'id="dataset-samples"',
            'data-route="maps/studio"', 'id="map-canvas"',
            'id="map-flight-canvas"', 'id="map-record-start"',
            'id="map-record-register"', 'id="map-recording-audit"',
        ):
            self.assertIn(identifier, html)
        canvas_script = (STATIC / "map_canvas.js").read_text(encoding="utf-8")
        action_script = (STATIC / "map_studio_actions.js").read_text(encoding="utf-8")
        studio_script = (STATIC / "map_studio.js").read_text(encoding="utf-8")
        map_script = (STATIC / "map_flight.js").read_text(encoding="utf-8")
        self.assertIn("root.MapCanvas", canvas_script)
        self.assertIn("root.MapStudioActions", action_script)
        self.assertIn("delete record.map_identity_sha256", studio_script)
        self.assertIn("startSelected('map-record')", map_script)
        self.assertIn("/api/maps/register", map_script)
        self.assertNotIn('id="operator-action"', html)
        self.assertEqual(html.count('class="workflow-step'), 6)
        self.assertIn('class="workflow-step active"', html)
        self.assertLess(
            html.index('id="setup-journey"'), html.index('id="setup-progress"')
        )

    def test_desktop_navigation_uses_workflow_sidebar(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        css = (STATIC / "navigation.css").read_text(encoding="utf-8")
        header = html.split('<header class="app-toolbar">', 1)[1].split(
            "</header>", 1
        )[0]
        self.assertNotIn('class="primary-nav"', header)
        self.assertIn('class="primary-nav"', html)
        self.assertIn('class="workspace-sidebar"', html)
        self.assertEqual(html.count('class="task-page-toolbar"'), 4)
        self.assertEqual(html.count('class="secondary-nav"'), 4)
        self.assertIn("body { min-width: 1024px; }", css)
        self.assertIn("grid-template-columns: 315px", css)
        self.assertIn(".dataset-summary-grid", css)
        self.assertNotIn("max-width: 760px", css)


if __name__ == "__main__":
    unittest.main()
