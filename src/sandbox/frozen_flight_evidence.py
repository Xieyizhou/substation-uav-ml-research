"""Verify historical SITL outcomes from frozen bytes, without current paths.

This verifies historical scope only. It grants no authority to fly a changed
runtime or model; current-runtime qualification is a separate decision.
"""

from src.sandbox.evidence_archive import EvidenceArchive

BASE = "data/research/material-shadow-v1/autonomy-avoidance-v1/"
POSITIVE = "sandbox-replan-v1-e208b6496d5440259e24d8a3b8a09240"
STOPPED = "sandbox-replan-v1-e209b6496d5440259e24d8a3b8a09240"
STOP_PROOF = "outputs/sandbox/vision-control/stop-probe-001/proof.json"


def _landed(record):
    if (record.get("landing_confirmed") is not True
            or record.get("final_armed") is not False
            or record.get("owned_processes_exited") is not True):
        raise ValueError("Incomplete frozen landing or cleanup evidence")


def verify_flight_archive(path, identity):
    with EvidenceArchive(path, expected_identity=identity) as archive:
        verified = archive.verify_all()

        def read(relative):
            return archive.record(BASE + relative)

        completion = read("startup-repeat-v2/completion.json")
        protocol = read("startup-repeat-v2/protocol.json")
        expected = {("no_path", 1)} | {(case, n) for case in ("baseline", "west", "later") for n in (1, 2)}
        units, results = protocol["units"], completion["units"]
        if len(units) != 7 or {(u["case"], u["repeat"]) for u in units} != expected:
            raise ValueError("Incomplete frozen repeat matrix")
        if (completion.get("status") != "repeat_campaign_verified"
                or completion.get("sandbox_integration_allowed") is not True
                or completion.get("not_run") or len(results) != 7
                or any(row.get("passed") is not True for row in results)):
            raise ValueError("Frozen repeat campaign did not pass")
        if {(r["case"], r["repeat"], r["directory"]) for r in results} != {(u["case"], u["repeat"], u["directory"]) for u in units}:
            raise ValueError("Frozen result identities differ")
        read("lowload-vehicle-v1/protocol.json")
        read("envelope-aware-scene-v1/protocol.json")
        for unit in units:
            case, n = unit["case"], unit["repeat"]
            prefix = "lowload-repeat-v1" if case in ("no_path", "baseline") or (case, n) == ("west", 1) else "startup-repeat-v2"
            directory = f"{prefix}-{case}-{n}"
            if unit["directory"] != directory:
                raise ValueError("Unexpected frozen flight directory")
            done = read(directory + "/completion.json")
            expected_status = "expected_safe_no_path_rejection_verified" if case == "no_path" else "sensor_stop_replan_resume_goal_verified"
            if done.get("status") != expected_status:
                raise ValueError("Unexpected frozen flight result")
            read(directory + "/protocol.json")
            _landed(read(directory + "/runtime/receipt.json"))
            if read(directory + "/evidence-replay.json").get("status") != "raw_scan_pose_maps_replayed":
                raise ValueError("Missing frozen raw sensor replay")
            if case != "no_path":
                read(directory + "/map-registration.json")
                if read(directory + "/runtime/vision/completion.json").get("status") != "live_complete_not_flight_certified":
                    raise ValueError("Incomplete frozen vision recording")
        done = read(POSITIVE + "/completion.json")
        read(POSITIVE + "/protocol.json")
        _landed(read(POSITIVE + "/runtime/receipt.json"))
        stop = archive.record(STOP_PROOF)
        read(STOPPED + "/protocol.json")
        runtime = read(STOPPED + "/runtime/receipt.json")
        _landed(runtime)
        # Compare opaque original provenance strings, never resolve old paths.
        stopped_key = archive.records[BASE + STOPPED + "/runtime/receipt.json"]
        if (done.get("status") != "visual_stop_replan_resume_standoff_verified"
                or stop.get("state") != "controlled_visual_stop_verified"
                or str(stop.get("run", "")).rstrip("/") + "/runtime/receipt.json" != stopped_key
                or "Sandbox requested controlled stop and landing" not in str(runtime.get("error"))):
            raise ValueError("Incomplete frozen visual or stop evidence")
        return dict(**verified, historical_verified=True, positive_lidar_flights=6,
                    negative_lidar_flights=1, positive_visual_flights=1,
                    controlled_visual_stops=1, current_runtime_certified=False,
                    physical_flight_certified=False)
