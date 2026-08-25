"""Live RGB-D semantic planner service attached to flight lifecycle."""

import asyncio
import json
from pathlib import Path

from PIL import Image

from src.flight.semantic_decision_queue import enqueue_decision, pending_queue_depth
from src.planner.active_inspection import ActiveInspectionPlanner, ActiveInspectionPolicy, RuntimeMap, object_identity
from src.planner.executed_coverage import ExecutedCoverageTracker
from src.sensors.gazebo_rgbd_native import NativeGazeboRgbDepthSource
from src.sensors.rgb_depth_sync import depth_for_bbox
from src.vision.active_inspection_live import ActiveInspectionLiveBridge
from src.vision.evaluation.detector import EquipmentDetector
from src.vision.evaluation.temporal_observations import TemporalObservationFilter


class ActiveRgbdRuntime:
    def __init__(self, runtime_map, policy, weights, output_directory, *, scheduler="active_utility", rgb_topic="auto", depth_topic="auto"):
        self.output = Path(output_directory)
        self.output.mkdir(parents=True, exist_ok=True)
        self.source = NativeGazeboRgbDepthSource(self.output / "frames")
        self.planner = ActiveInspectionPlanner(
            RuntimeMap.from_mapping(runtime_map),
            ActiveInspectionPolicy.from_mapping(policy),
            scheduler=scheduler,
        )
        self.coverage = ExecutedCoverageTracker(self.planner.map, radius_m=2.0)
        self.detector = EquipmentDetector(weights, confidence=self.planner.policy.stable_confidence)
        self.temporal = TemporalObservationFilter()
        self.ready = asyncio.Event()
        self.task = None
        self.latest = self.phase_state = self.replan_config = self.replan_state = None
        self.seen_tracking_ids = set()
        self.processed_pairs = 0
        self.start_timestamp = None
        self.exploration_started = False
        self.world = runtime_map.get("name", runtime_map.get("map_id", "substation_complex"))

    async def start(self, latest, phase_state, replan_config, replan_state):
        self.latest, self.phase_state = latest, phase_state
        self.replan_config, self.replan_state = replan_config, replan_state
        await self.source.start()
        self.task = asyncio.create_task(self._run(), name="active-rgbd-runtime")

    async def wait_ready(self, timeout_s):
        ready_task = asyncio.create_task(self.ready.wait(), name="active-rgbd-ready")
        done, _pending = await asyncio.wait(
            {ready_task, self.task},
            timeout=float(timeout_s),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if self.task in done:
            await self.task
            raise RuntimeError("active RGB-D runtime stopped before its first frame")
        if ready_task not in done:
            ready_task.cancel()
            await asyncio.gather(ready_task, return_exceptions=True)
            raise TimeoutError(
                f"active RGB-D runtime produced no paired frame within {timeout_s}s"
            )

    def _pose(self):
        sample = self.latest.get("position_velocity")
        attitude = self.latest.get("attitude")
        if sample is None or attitude is None:
            return None
        position = sample.position
        return {"east_m": float(position.east_m), "north_m": float(position.north_m),
                "altitude_m": -float(position.down_m), "roll_deg": float(attitude.roll_deg),
                "pitch_deg": float(attitude.pitch_deg), "yaw_deg": float(attitude.yaw_deg)}

    def _elapsed(self, timestamp):
        if self.start_timestamp is None:
            self.start_timestamp = float(timestamp)
        return max(0.0, float(timestamp) - self.start_timestamp)

    def _sync_planner_busy(self):
        self.planner.route_active = (
            self.replan_state.get("semantic_replacement_armed") is False
            or pending_queue_depth(self.replan_config, self.replan_state) > 0
        )

    def _append_new_decisions(self, before):
        publisher = self.phase_state.get("_event_publisher")
        for decision in self.planner.decisions[before:]:
            envelope = enqueue_decision(
                self.replan_config,
                self.replan_state,
                decision,
                created_at=decision.get("timestamp_s", self.processed_pairs),
                trigger=decision.get("trigger"),
            )
            if envelope is not None and publisher:
                publisher.publish(
                    "planner_decision_enqueued",
                    decision_id=envelope["decision_id"],
                    route_identity=envelope["route_identity"],
                    kind=envelope["kind"],
                    track_id=envelope.get("track_id"),
                    queue_depth=pending_queue_depth(
                        self.replan_config, self.replan_state
                    ),
                    simulation_timestamp_s=envelope["created_at"],
                )
            if envelope is not None:
                print(
                    "Semantic decision enqueued: "
                    f"{envelope['decision_id'][:12]} "
                    f"kind={envelope['kind']} "
                    f"waypoints={len(envelope['replacement_waypoints'])}"
                )
        if publisher:
            for event in self.planner.events:
                if event["sequence"] > self.replan_config.get("semantic_event_sequence", 0):
                    event_name = event["event"]
                    if event_name == "semantic_route_replaced":
                        event_name = "scheduler_route_proposed"
                    publisher.publish(event_name, **{k: v for k, v in event.items() if k not in {"event", "sequence"}})
                    self.replan_config["semantic_event_sequence"] = event["sequence"]

    def _sync_terminal_state(self):
        if not self.planner.terminated:
            return
        terminal_event = self.planner.events[-1]["event"] if self.planner.events else "mission_completed"
        reason = (
            "budget_exhausted"
            if terminal_event == "mission_budget_exhausted"
            else "mission_completed"
        )
        if self.replan_state.get("semantic_mission_complete"):
            return
        self.replan_state["semantic_mission_complete"] = True
        self.replan_state["terminal_reason"] = reason
        publisher = self.phase_state.get("_event_publisher")
        if publisher:
            publisher.publish(
                "mission_terminal_state",
                reason=reason,
                planner_event=terminal_event,
                scheduler=self.planner.scheduler,
            )

    async def _run(self):
        async for rgb, depth, _skew_ms in self.source.events(timeout_s=15):
            self.ready.set(); self.processed_pairs += 1
            elapsed_s = self._elapsed(rgb.capture_timestamp)
            pose = self._pose()
            if pose is None:
                continue
            self.coverage.observe(pose["east_m"], pose["north_m"])
            self.planner.executed_coverage = self.coverage.coverage
            self._drain_feedback(rgb.capture_timestamp, pose)
            staging = self.output / "frames"
            current_depth = depth
            bridge = ActiveInspectionLiveBridge(
                self.detector,
                lambda rows, timestamp: self.temporal.update(timestamp, rows),
                lambda _frame, row: depth_for_bbox(current_depth, (rgb.width, rgb.height), row["bbox"], staging),
                lambda frame: Image.open(staging / frame.payload_relative_path).convert("RGB"),
                {"fx": rgb.width / (2 * __import__("math").tan(1.466 / 2)), "fy": rgb.width / (2 * __import__("math").tan(1.466 / 2)), "cx": rgb.width / 2, "cy": rgb.height / 2},
                {"forward_m": 0.18, "right_m": 0.0, "down_m": -0.12,
                 "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0},
            )
            record = bridge.record(rgb, pose, elapsed_s=elapsed_s)
            if record is None:
                self._start_exploration(rgb.capture_timestamp, pose)
                continue
            new = [row for row in record["observations"] if row.get("tracking_id") not in self.seen_tracking_ids]
            if not new:
                self._start_exploration(rgb.capture_timestamp, pose)
                continue
            self.seen_tracking_ids.update(row.get("tracking_id") for row in new)
            record["observations"] = new
            before = len(self.planner.decisions)
            self._sync_planner_busy()
            self.planner.process(record)
            self._append_new_decisions(before)
            self._sync_terminal_state()

    def _start_exploration(self, timestamp, pose):
        if self.exploration_started:
            return
        self.exploration_started = True
        exploration_pose = dict(pose)
        exploration_pose["altitude_m"] = max(
            1.5, float(exploration_pose.get("altitude_m", 0))
        )
        before = len(self.planner.decisions)
        self._sync_planner_busy()
        self.planner.process({
            "trigger": "waypoint_reached",
            "timestamp_s": timestamp,
            "elapsed_s": self._elapsed(timestamp),
            "vehicle_pose": exploration_pose,
            "observations": [],
        })
        self._append_new_decisions(before)
        self._sync_terminal_state()

    def _drain_feedback(self, timestamp, pose):
        feedback = self.replan_config.setdefault("semantic_feedback", [])
        while feedback:
            item = feedback.pop(0)
            publisher = self.phase_state.get("_event_publisher")
            if publisher:
                publisher.publish(
                    "planner_feedback_received",
                    feedback=item["event"],
                    decision_id=item.get("decision_id"),
                    route_identity=item.get("route_identity"),
                    track_id=item.get("candidate_id"),
                    queue_depth=pending_queue_depth(
                        self.replan_config, self.replan_state
                    ),
                    simulation_timestamp_s=item.get("timestamp_s", timestamp),
                )
            tracking_ids = []
            candidate_id = item.get("candidate_id")
            track = self.planner.tracks.get(candidate_id) if candidate_id else None
            if item["event"] == "target_completed" and track is not None:
                tracking_ids = sorted(track.source_tracking_ids)
            before = len(self.planner.decisions)
            self._sync_planner_busy()
            self.planner.process({
                "trigger": item["event"],
                "timestamp_s": timestamp,
                "elapsed_s": self._elapsed(timestamp),
                "vehicle_pose": pose,
                "completed_tracking_ids": tracking_ids,
                "completed_target_ids": [candidate_id]
                if item["event"] == "target_completed" and candidate_id else [],
                "observations": [],
            })
            self._append_new_decisions(before)
            self._sync_terminal_state()

    async def stop(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.source.stop()
        pairing = self.source.receipt()
        skew_p95 = pairing["skew_ms"]["p95"]
        blocked_reasons = []
        if pairing["rgb_pairing_success_rate"] < .95:
            blocked_reasons.append("rgb_pairing_rate_below_95_percent")
        if skew_p95 is None or skew_p95 > 33.334:
            blocked_reasons.append("rgb_depth_skew_p95_exceeded")
        receipt = {"schema_version": "1.0", "evidence_level": "gazebo_live_rgbd",
                   "world": self.world, "model": self.planner.policy.model_artifact_identity,
                   "policy": self.planner.policy.artifact_identity,
                   "scheduler": self.planner.scheduler,
                   "processed_pairs": self.processed_pairs,
                   "pairing": pairing,
                   "acceptance": {"status": "blocked" if blocked_reasons else "ready",
                                  "blocked_reasons": blocked_reasons,
                                  "rgb_pairing_rate_min": .95,
                                  "skew_p95_ms_max": 33.334},
                   "trial": self.replan_config.get("semantic_trial"),
                   "trial_state": {
                       key: self.replan_state.get(key)
                       for key in (
                           "semantic_trial_replacements",
                           "semantic_trial_stop_requested",
                           "semantic_mission_complete",
                       )
                   },
                   "planner_summary": {
                       "exploration_coverage": round(self.coverage.coverage, 8),
                       "executed_coverage": self.coverage.receipt(),
                       "stable_classes": sorted({
                           track.class_name for track in self.planner.tracks.values()
                           if not track.ambiguous
                       }),
                       "registered_count": len(self.planner.tracks),
                       "inspected_count": sum(
                           bool(track.inspected) for track in self.planner.tracks.values()
                       ),
                       "ambiguous_count": sum(
                           bool(track.ambiguous) for track in self.planner.tracks.values()
                       ),
                   },
                   "runtime_state": {
                       "inference_pairs": self.processed_pairs,
                       "stable_tracking_ids": len(self.seen_tracking_ids),
                       "target_count": len(self.planner.tracks),
                       "queue_depth": pending_queue_depth(
                           self.replan_config, self.replan_state
                       ),
                       "terminal_reason": self.replan_state.get("terminal_reason"),
                   },
                   "tracks": [track.public() if hasattr(track, "public") else track.__dict__ for track in self.planner.tracks.values()]}
        receipt["artifact_identity"] = object_identity(receipt)
        receipt["identity"] = receipt["artifact_identity"]
        (self.output / "rgb-depth-runtime-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def build_active_rgbd_runtime(args, planner_config):
    policy = json.loads(Path("config/perception/active_inspection_policy.json").read_text())
    runtime_map = {"name": planner_config.get("map_id", "substation_complex"),
                   "width_cells": planner_config["width"], "height_cells": planner_config["height"],
                   "resolution_m": planner_config["resolution_m"],
                   "occupied_cells": [list(cell) for cell in planner_config["inflated_blocking_cells"]]}
    event_path = getattr(args, "visual_mission_events", None)
    run_id = Path(event_path).stem if event_path else "live"
    return ActiveRgbdRuntime(
        runtime_map,
        policy,
        args.equipment_model,
        Path("outputs/sandbox/active_rgbd_runtime") / run_id,
        scheduler=getattr(args, "inspection_scheduler", "active_utility"),
        rgb_topic="auto",
        depth_topic="auto",
    )
