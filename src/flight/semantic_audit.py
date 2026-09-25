"""Replay visual localization and verify measured stop/replan/resume in SITL."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.flight.sitl_support import read_samples, point_box_distance
from src.flight.sitl_replay import replay
from src.flight.tracking_envelope import check_route
from src.flight.visual_target_route import plan_visual_standoff
from src.ml.artifacts import file_sha256, object_sha256
from src.planner.local_frame import LocalFrame
from src.sensors.camera_decoder import decode_camera_payload
from src.sensors.gazebo_depth_memory import MemoryDepth, bbox_depth
from src.sensors.types import CameraFrame
from src.vision.canonical.plan import write_record
from src.vision.localization.rgbd_localization import localize_bbox


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(out, assets):
    out = Path(out).resolve()
    load = lambda name: json.loads((out / name).read_text())
    protocol, route, runtime = load("protocol.json"), load("route.json"), load("runtime/receipt.json")
    for path, digest in protocol["inputs"].items():
        require(file_sha256(path) == digest, "frozen input changed: " + path)
    require(route["status"] == "visual_standoff_route_completed", "route did not complete")
    require(runtime["status"] == "sitl_hover_landed_disarmed" and runtime["landing_confirmed"]
            and runtime["final_armed"] is False and runtime["owned_processes_exited"], "landing or cleanup incomplete")
    names = ["visual_target_stop_requested", "actual_stop_confirmed", "route_replaced", "visual_standoff_reached_after_replan"]
    ordered, cursor = [], 0
    for name in names:
        matches = [(i, row) for i, row in enumerate(route["events"]) if i >= cursor and row["event"] == name]
        require(bool(matches), "missing ordered event: " + name)
        index, row = matches[0]
        ordered.append(row)
        cursor = index+1
    require(ordered[0]["speed_m_s"] >= .1 and ordered[1]["speed_m_s"] < .08, "moving detection and actual stop not demonstrated")
    require(len(route["replans"]) == 1, "expected one bounded visual replacement")
    plan = route["replans"][0]
    root = out / "vision/accepted-request"
    saved = json.loads((root / "request.json").read_text())
    request = saved["request"]
    for path, digest in saved["inputs"].items():
        require(file_sha256(path) == digest, "accepted image/depth changed")
    identity = dict(request)
    identity.pop("identity")
    require(object_sha256(identity) == request["identity"], "visual request identity changed")
    samples = {}
    for path in (root / "samples").glob("*.json"):
        record = json.loads(path.read_text())
        frame = CameraFrame.from_record(record["frame"])
        decoded = decode_camera_payload(frame, root)
        require(decoded.image.decoded_content_sha256 == record["rgb_sha256"], "accepted RGB pixels changed")
        samples[record["rgb_sha256"]] = (frame, decoded.image.array)
    poses, localizations = [], []
    require(len(request["evidence"]) == 3, "three confirmations missing")
    for evidence in request["evidence"]:
        frame, pixels = samples[evidence["rgb_sha256"]]
        array = np.load(root / (evidence["depth_sha256"] + ".npz"))["depth"]
        require(hashlib.sha256(array.tobytes()).hexdigest() == evidence["depth_sha256"], "accepted depth pixels changed")
        match = evidence["pose_match"]
        require(abs(match["position_timestamp"]-evidence["capture_timestamp"]) <= .05
                and abs(match["attitude_timestamp"]-evidence["capture_timestamp"]) <= .05
                and abs(evidence["depth_timestamp"]-evidence["capture_timestamp"]) <= .033334, "capture pairing exceeds limits")
        depth = MemoryDepth(evidence["depth_timestamp"], 0, array, evidence["depth_sha256"])
        intrinsics = dict(fx=frame.width/(2*math.tan(1.466/2)), fy=frame.width/(2*math.tan(1.466/2)),
                          cx=frame.width/2, cy=frame.height/2)
        rows = [row for row in evidence["observations"] if row["class_name"] == request["class_name"]]
        require(bool(rows), "request class absent from observations")
        row = max(rows, key=lambda r: r["confidence"])
        distance = bbox_depth(depth, (frame.width, frame.height), row["bbox"])
        require(distance is not None and abs(distance-row["depth_m"]) < 1e-8, "depth cannot be reproduced")
        position = localize_bbox(distance, row["bbox"], intrinsics, match["pose"], dict(forward_m=.18, right_m=0, down_m=-.12))
        require(max(abs(position[k]-row["local_position"][k]) for k in position) < 1e-8, "localization cannot be reproduced")
        localizations.append(position)
        poses.append(match)
    for key in request["local_position"]:
        require(abs(sum(p[key] for p in localizations)/3-request["local_position"][key]) < 1e-8, "target confirmation average changed")
    frame = LocalFrame.from_mapping(load("inair-alignment.json")["local_frame"])
    replayed = plan_visual_standoff(request, now=request["created_monotonic"], frame=frame,
                                   start=plan["points"][0], boxes=protocol["static_boxes"]+plan["observed_boxes"])
    require(np.allclose(replayed["route"], plan["points"], atol=1e-9), "semantic route replay differs")
    require(math.dist(plan["points"][-1], protocol["goal"]) > .5, "vision did not change the destination")
    require(check_route(plan["points"], protocol["static_boxes"]+plan["observed_boxes"], map_uncertainty=.2)["passed"], "route envelope failed")
    samples = route["samples"]
    require(bool(samples), "empty flight telemetry")
    require(all(row["cross_track_m"] <= .18 and row["lidar_m"] > 1.5
                and math.hypot(row["position"]["vn"], row["position"]["ve"]) <= .35 for row in samples), "flight envelope violated")
    resumed = [row for row in samples if row["monotonic"] > ordered[2]["monotonic"]
               and math.hypot(row["position"]["vn"], row["position"]["ve"]) > .1 and math.hypot(*row["command_ned"][:2]) > .1]
    require(bool(resumed), "no measured resumed motion")
    gap = math.dist(samples[-1]["map_position"], plan["points"][-1])
    require(gap < .03, "stand-off waypoint not reached")
    logs = list((out / "runtime/px4-work/log").rglob("*.ulg"))
    require(len(logs) == 1, "ambiguous vehicle log")
    truth = [row for _, multi, row in read_samples(logs[0], {"vehicle_local_position_groundtruth"}) if multi == 0]
    require(bool(truth), "groundtruth audit unavailable")
    envelope = Path(assets) / "envelope.json"
    radius = json.loads(envelope.read_text())["collision_sphere_radius_m"]
    clearance = min(point_box_distance((row["y"]+10, row["x"]+10, -row["z"]), box)-radius
                    for row in truth for box in protocol["static_boxes"])
    require(clearance > 0, "collision clearance failed")
    replay(out)
    metrics = dict(ordered_events=names, detected_class=request["class_name"], replayed_visual_confirmations=3,
                   resumed_samples=len(resumed), final_estimated_gap_m=gap, whole_run_collision_sphere_clearance_m=clearance,
                   max_cross_track_m=max(row["cross_track_m"] for row in samples),
                   min_lidar_m=min(row["lidar_m"] for row in samples), original_goal=protocol["goal"], visual_goal=plan["points"][-1])
    paths = [out / name for name in ("protocol.json", "route.json", "runtime/receipt.json", "runtime/telemetry.jsonl",
             "inair-alignment.json", "prearm-alignment.json", "evidence-replay.json", "vision/receipt.json")]
    paths += [logs[0], envelope, Path(__file__).resolve(), root / "request.json"]
    paths += list(root.rglob("*.json")) + list(root.rglob("*.png")) + list(root.glob("*.npz"))
    value = dict(status="visual_stop_replan_resume_standoff_verified", metrics=metrics,
                 scope="single equipment-directed stand-off replacement in existing fixed PX4 SITL scene",
                 physical_flight_certified=False, training_admitted=False, promotable=False,
                 inputs={str(path): file_sha256(path) for path in paths})
    write_record(out / "completion.json", value)
    print(json.dumps(metrics, indent=2))
    return value


