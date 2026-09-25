"""Launch verification for the fixed equipment-directed visual SITL workflow."""

import json
from pathlib import Path

from src.ml.artifacts import file_sha256
from src.sandbox.live_replan_gate import BASE, require_repeat_gate, validate_record

POSITIVE_RUN = "sandbox-replan-v1-e208b6496d5440259e24d8a3b8a09240"
STOP_RUN = "sandbox-replan-v1-e209b6496d5440259e24d8a3b8a09240"
STOP_PROOF = Path("outputs/sandbox/vision-control/stop-probe-001/proof.json")


def summary(root):
    root = Path(root)
    try:
        positive = json.loads((root / BASE / POSITIVE_RUN / "completion.json").read_text())
        stop = json.loads((root / STOP_PROOF).read_text())
    except (OSError, ValueError):
        positive, stop = {}, {}
    ready = positive.get("status") == "visual_stop_replan_resume_standoff_verified" and stop.get("state") == "controlled_visual_stop_verified"
    return dict(ready_for_launch_revalidation=ready, positive_verified=positive.get("status") == "visual_stop_replan_resume_standoff_verified",
                controlled_stop_verified=stop.get("state") == "controlled_visual_stop_verified",
                scope="fixed existing scene; one equipment-directed stand-off route; 3 synchronized confirmations; LiDAR stop remains active",
                model="material-routed-480-7", maximum_feedback_mib=16, physical_flight_certified=False)


def require_visual_gate(root):
    root = Path(root)
    old = require_repeat_gate(root)
    positive = root / BASE / POSITIVE_RUN
    done = validate_record(positive / "completion.json")
    validate_record(positive / "protocol.json")
    runtime = validate_record(positive / "runtime/receipt.json")
    stop = validate_record(root / STOP_PROOF)
    stopped = root / BASE / STOP_RUN
    validate_record(stopped / "protocol.json")
    stop_runtime = validate_record(stopped / "runtime/receipt.json")
    if (done.get("status") != "visual_stop_replan_resume_standoff_verified"
            or stop.get("state") != "controlled_visual_stop_verified"
            or Path(stop.get("run", "")).resolve() != stopped.resolve()):
        raise ValueError("visual flight and controlled-stop evidence incomplete")
    for record in (runtime, stop_runtime):
        if not record.get("landing_confirmed") or record.get("final_armed") is not False or not record.get("owned_processes_exited"):
            raise ValueError("visual landing/cleanup evidence incomplete")
    if "Sandbox requested controlled stop and landing" not in str(stop_runtime.get("error")):
        raise ValueError("controlled-stop failure reason differs")
    return dict(positive_receipt_sha256=file_sha256(positive / "completion.json"),
                stop_receipt_sha256=file_sha256(root / STOP_PROOF), previous_gate=old["campaign_hash"])
