"""Record the focused regression suite, not a whole-repository test claim."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import unittest
from scripts.flight.check_px4_shadow import ROOT
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

MODULES=['test_live_obstacle_map','test_sensor_contracts','test_sim_heading_profile','test_final_heading','test_vehicle_envelope','test_flight_safety','test_lowload_vehicle','test_lidar_arrival_observation','test_envelope_route','test_tracking_envelope','test_arrival_capture','test_turn_hold','test_replan_repeat_campaign','test_lidar_inspector','test_px4_startup_gate','test_live_replan_stop','test_live_replan_sandbox_gate','test_live_replan_display','test_sandbox_operator','test_sandbox_storage','test_sandbox_workflow','test_inspection_sandbox','test_sandbox_ui_state']

def run(output):
    if output.exists():raise ValueError('Preserve existing test evidence; choose a new attempt path')
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.'+m for m in MODULES])
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    paths=[ROOT/'tests'/f'{m}.py' for m in MODULES]+[Path(__file__).resolve()]
    paths += [ROOT/p for p in ['src/sandbox/live_replan_gate.py','src/sandbox/live_replan_stop.py','src/sandbox/live_replan_display.py','src/sandbox/job_commands.py','src/sandbox/job_runtime.py','src/sandbox/operator.py','src/inspection/app.py','src/inspection/service.py','src/inspection/static/live_replan.js','src/inspection/static/index.html','src/inspection/static/app.js','scripts/flight/fly_sandbox_replan.py','scripts/flight/sandbox_replan_lifecycle.py','src/flight/startup_gate.py']]
    output.parent.mkdir(parents=True,exist_ok=True)
    write_record(output,dict(status='focused_regressions_passed' if result.wasSuccessful() else 'failed',created_at=datetime.now(timezone.utc).isoformat(),tests_run=result.testsRun,passed=result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),skipped=[dict(test=str(t),reason=r) for t,r in result.skipped],failures=[dict(test=str(t),trace=r) for t,r in result.failures+result.errors],whole_repository_pass_claimed=False,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();raise SystemExit(run(args.output.resolve()))
