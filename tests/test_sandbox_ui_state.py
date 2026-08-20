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

    def test_scenario_options_only_change_with_scenario_signature(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is not available")
        script = f"""
const fs=require('fs'),vm=require('vm');
globalThis.document={{createElement:()=>({{value:'',textContent:''}})}};
const select={{dataset:{{}},value:'',writes:0,options:[],replaceChildren(...values){{this.options=values;this.writes+=1;}}}};
vm.runInThisContext(fs.readFileSync({json.dumps(str(STATIC / 'scenario_selector.js'))},'utf8'));
const first=[
  {{scenario_id:'alpha',dataset_role:'development'}},
  {{scenario_id:'beta',dataset_role:'validation'}},
];
const changed=[
  {{scenario_id:'beta',dataset_role:'validation'}},
  {{scenario_id:'gamma',dataset_role:'development'}},
];
const events=[];
events.push(SandboxScenarioSelector.sync(select,first));
select.value='beta';
events.push(SandboxScenarioSelector.sync(select,first));
const preserved=select.value;
events.push(SandboxScenarioSelector.sync(select,changed));
const preservedAfterChange=select.value;
events.push(SandboxScenarioSelector.sync(select,[changed[1]]));
process.stdout.write(JSON.stringify({{events,writes:select.writes,preserved,preservedAfterChange,fallback:select.value}}));
"""
        result = subprocess.run(
            [node, "-e", script], check=True, capture_output=True, text=True
        )
        value = json.loads(result.stdout)
        self.assertEqual(value["events"], [True, False, True, True])
        self.assertEqual(value["writes"], 3)
        self.assertEqual(value["preserved"], "beta")
        self.assertEqual(value["preservedAfterChange"], "beta")
        self.assertEqual(value["fallback"], "gamma")

        app = (STATIC / "app.js").read_text(encoding="utf-8")
        polling = app.split("async function refreshOperator()", 1)[1].split(
            "async function loadJobLog", 1
        )[0]
        self.assertNotIn("SandboxScenarioSelector", polling)
        self.assertNotIn("operator-scenario", polling)

    def test_latest_map_preview_and_flight_marker_padding_execute_in_node(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is not available")
        script = f"""
const fs=require('fs'),vm=require('vm');
class Element {{
  constructor() {{ this.children=[]; this.dataset={{}}; this.disabled=false; this.value='headless'; this._html=''; }}
  append(child) {{ this.children.push(child); }}
  set innerHTML(value) {{ this._html=value; this.children=[]; }}
  get innerHTML() {{ return this._html; }}
}}
const elements={{}};
globalThis.document={{getElementById:id => elements[id] ||= new Element()}};
const selected={{
  map_id:'new-map', revision_id:'new-revision', mission_id:'round-trip',
  map_name:'Latest map', map:{{width_m:32,height_m:32,objects:[]}},
  route:{{grid_path:[[0,0],[31,31]]}}
}};
globalThis.sessionStorage={{getItem:() => JSON.stringify(selected)}};
globalThis.MapCanvas={{
  colors:{{}},
  node:(tag,attrs) => ({{tag,attrs,children:[],append(child){{this.children.push(child)}}}})
}};
globalThis.esc=value => String(value);
globalThis.startManaged=()=>{{}};
globalThis.stopManaged=()=>{{}};
globalThis.requestConfirmation=async()=>false;
globalThis.post=async()=>({{}});
globalThis.actionError=()=>{{}};
globalThis.activateRoute=()=>{{}};
globalThis.setTimeout=()=>0;
let response={{
  active:false,
  latest:{{map_id:'old-map',revision_id:'old-revision',mission_id:'old-mission',status:'complete'}},
  map:{{width_m:16,height_m:16,objects:[]}}, route:{{grid_path:[[0,0],[1,1]]}},
  trajectory:[], live:null, recording_audit:null
}};
globalThis.api=async()=>response;
(async()=>{{
  vm.runInThisContext(fs.readFileSync({json.dumps(str(STATIC / 'map_flight.js'))},'utf8'));
  await MapFlight.refresh();
  const preview={{
    facts:elements['map-flight-facts'].innerHTML,
    health:elements['map-flight-health'].textContent
  }};
  response={{
    active:true,
    latest:{{map_id:'new-map',revision_id:'new-revision',mission_id:'round-trip',status:'running'}},
    map:selected.map, route:selected.route, trajectory:[],
    live:{{position:{{east_m:32,north_m:32,down_m:-1.5}},velocity:{{north_m_s:0,east_m_s:0}},yaw_deg:0,elapsed_s:1,phase:'outbound',stale:false}},
    recording_audit:null
  }};
  await MapFlight.refresh();
  const drone=elements['map-flight-canvas'].children.find(item=>item.tag==='g');
  process.stdout.write(JSON.stringify({{preview,transform:drone.attrs.transform}}));
}})().catch(error=>{{console.error(error);process.exit(1)}});
"""
        result = subprocess.run(
            [node, "-e", script], check=True, capture_output=True, text=True
        )
        value = json.loads(result.stdout)
        self.assertIn("Latest map", value["preview"]["facts"])
        self.assertEqual(value["preview"]["health"], "Ready")
        self.assertEqual(value["transform"], "translate(578 22) rotate(0)")

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
        self.assertNotIn('id="operator-state"', html)
        self.assertNotIn('id="operator-stop"', html)
        self.assertNotIn('CURRENT FLIGHT', html)
        canvas_script = (STATIC / "map_canvas.js").read_text(encoding="utf-8")
        action_script = (STATIC / "map_studio_actions.js").read_text(encoding="utf-8")
        studio_script = (STATIC / "map_studio.js").read_text(encoding="utf-8")
        app_script = (STATIC / "app.js").read_text(encoding="utf-8")
        map_script = (STATIC / "map_flight.js").read_text(encoding="utf-8")
        self.assertIn("root.MapCanvas", canvas_script)
        self.assertIn("root.MapStudioActions", action_script)
        self.assertIn("delete record.map_identity_sha256", studio_script)
        self.assertIn("map: clone(state.map)", studio_script)
        self.assertIn("route: route ? clone(route) : null", studio_script)
        self.assertIn("const latestRevisionMap", studio_script)
        self.assertIn("startSelected('map-record')", map_script)
        self.assertIn("/api/maps/register", map_script)
        self.assertIn("const padding = 22", map_script)
        self.assertIn("runMatchesSelection", map_script)
        self.assertIn("globalThis.activeManagedJobId=active?.job_id||''", app_script)
        self.assertIn("stopManaged(globalThis.activeManagedJobId || '')", map_script)
        self.assertNotIn("q('operator-stop')", map_script)
        self.assertNotIn('id="map-select-open"', html)
        self.assertNotIn('id="map-picker"', html)
        self.assertNotIn('src="/map_picker.js"', html)
        self.assertIn('src="/scenario_selector.js"', html)
        self.assertNotIn('id="operator-action"', html)
        self.assertNotIn('class="window-controls"', html)
        self.assertNotIn("Visual model result", html)
        self.assertNotIn("Generalization result", html)
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
