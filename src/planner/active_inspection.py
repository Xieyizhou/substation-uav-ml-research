"""Truth-blind active semantic inspection for sandbox missions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import heapq
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


ALLOWED_CLASSES = ("capacitor_bank", "reactor", "switchgear", "transformer")
ALLOWED_TRIGGERS = {
    "stable_detection", "waypoint_reached", "target_completed", "budget_update",
    "safety_replan_active", "safety_replan_cleared",
}
FROZEN_WEIGHTS = {
    "new_class": 4.0, "uninspected": 3.0, "confidence_gap": 2.0,
    "visibility_improvement": 1.5, "path_length": -0.08, "energy": -0.12,
    "risk": -1.0, "repeat_observation": -0.5,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def object_identity(value: Any) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class ActiveInspectionPolicy:
    policy_id: str
    model_artifact_identity: str
    weights: Mapping[str, float]
    stable_confidence: float = .60
    merge_distance_m: float = 2.0
    conflict_distance_m: float = 1.5
    track_timeout_s: float = 15.0
    preferred_standoff_m: float = 5.0
    inspection_confidence: float = .85
    exploration_stride_cells: int = 4
    exploration_completion: float = .90
    mission_budget_s: float = 600.0
    observation_hold_s: float = 2.0
    low_clearance_cells: int = 2
    confirmation_sweep_passes: int = 1
    confirmation_max_route_cells: int = 80

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ActiveInspectionPolicy":
        if value.get("runtime_mode") != "active_semantic_inspection":
            raise ValueError("policy runtime_mode must be active_semantic_inspection")
        weights = {key: float(value.get("weights", {}).get(key, math.nan)) for key in FROZEN_WEIGHTS}
        if weights != FROZEN_WEIGHTS:
            raise ValueError("active inspection scoring weights are frozen")
        thresholds, limits = value.get("thresholds", {}), value.get("limits", {})
        return cls(
            policy_id=str(value["policy_id"]), model_artifact_identity=str(value["model_artifact_identity"]),
            weights=weights, stable_confidence=float(thresholds.get("stable_confidence", .60)),
            merge_distance_m=float(thresholds.get("merge_distance_m", 2)),
            conflict_distance_m=float(thresholds.get("conflict_distance_m", 1.5)),
            track_timeout_s=float(thresholds.get("track_timeout_s", 15)),
            preferred_standoff_m=float(thresholds.get("preferred_standoff_m", 5)),
            inspection_confidence=float(thresholds.get("inspection_confidence", .85)),
            exploration_stride_cells=int(limits.get("exploration_stride_cells", 4)),
            exploration_completion=float(limits.get("exploration_completion", .90)),
            mission_budget_s=float(limits.get("mission_budget_s", 600)),
            observation_hold_s=float(limits.get("observation_hold_s", 2)),
            low_clearance_cells=int(limits.get("low_clearance_cells", 2)),
            confirmation_sweep_passes=int(limits.get("confirmation_sweep_passes", 1)),
            confirmation_max_route_cells=int(limits.get("confirmation_max_route_cells", 80)),
        )

    @property
    def artifact_identity(self) -> str:
        return object_identity(asdict(self))


@dataclass
class DeviceTrack:
    target_id: str
    class_name: str
    confidence: float
    east_m: float
    north_m: float
    altitude_m: float
    observation_count: int
    best_confidence: float
    best_depth_m: float
    last_seen_s: float
    source_tracking_ids: list[str] = field(default_factory=list)
    inspected: bool = False
    ambiguous: bool = False


@dataclass(frozen=True)
class RuntimeMap:
    width_cells: int
    height_cells: int
    resolution_m: float
    occupied_cells: frozenset[tuple[int, int]]
    map_identity: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeMap":
        resolution = float(value.get("resolution_m", value.get("cell_size_m", 1)))
        width = int(value.get("width_cells", math.ceil(float(value.get("width_m", value.get("width", 0))) / resolution)))
        height = int(value.get("height_cells", math.ceil(float(value.get("height_m", value.get("height", 0))) / resolution)))
        occupied = {(int(cell[0]), int(cell[1])) for cell in value.get("occupied_cells", [])}
        # Object semantics are deliberately ignored; only explicit geometry is read.
        for obj in value.get("objects", []):
            geometry = obj.get("geometry", {})
            occupied.update((int(c[0]), int(c[1])) for c in geometry.get("occupied_cells", geometry.get("cells", [])))
            footprint = geometry.get("footprint", {})
            x, y = int(footprint.get("x_cell", -1)), int(footprint.get("y_cell", -1))
            w, h = int(footprint.get("width_cells", 0)), int(footprint.get("height_cells", 0))
            occupied.update((cx, cy) for cx in range(x, x + w) for cy in range(y, y + h))
            if "east_m" in obj and "north_m" in obj:
                half_width = float(obj.get("width_m", resolution)) / 2
                half_depth = float(obj.get("depth_m", resolution)) / 2
                west = math.floor((float(obj["east_m"]) - half_width) / resolution)
                east = math.ceil((float(obj["east_m"]) + half_width) / resolution)
                south = math.floor((float(obj["north_m"]) - half_depth) / resolution)
                north = math.ceil((float(obj["north_m"]) + half_depth) / resolution)
                occupied.update(
                    (cx, cy)
                    for cx in range(max(0, west), min(width, east))
                    for cy in range(max(0, south), min(height, north))
                )
        if width <= 0 or height <= 0 or resolution <= 0:
            raise ValueError("map bounds and resolution must be positive")
        view = {"width_cells": width, "height_cells": height, "resolution_m": resolution,
                "occupied_cells": sorted([list(c) for c in occupied])}
        return cls(width, height, resolution, frozenset(occupied), object_identity(view))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _astar(grid: RuntimeMap, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    if start in grid.occupied_cells or goal in grid.occupied_cells:
        return None
    queue = [(0, start[1], start[0], start)]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost = {start: 0}
    while queue:
        _, _, _, current = heapq.heappop(queue)
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
            nxt = current[0] + dx, current[1] + dy
            if not (0 <= nxt[0] < grid.width_cells and 0 <= nxt[1] < grid.height_cells):
                continue
            if nxt in grid.occupied_cells or cost[current] + 1 >= cost.get(nxt, math.inf):
                continue
            cost[nxt], parent[nxt] = cost[current] + 1, current
            estimate = cost[nxt] + abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
            heapq.heappush(queue, (estimate, nxt[1], nxt[0], nxt))
    return None


class ActiveInspectionPlanner:
    def __init__(self, runtime_map: RuntimeMap, policy: ActiveInspectionPolicy, scheduler: str = "active_utility") -> None:
        if scheduler not in {"fixed_serpentine", "nearest_target_first", "active_utility"}:
            raise ValueError(f"unsupported live inspection scheduler: {scheduler}")
        self.map, self.policy = runtime_map, policy
        self.scheduler = scheduler
        self.tracks: dict[str, DeviceTrack] = {}
        self.events: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.safety_active = self.terminated = False
        self.route_active = False
        self.executed_coverage = None
        self._next_track = 1
        self._position_history: dict[str, list[tuple[float, float, float]]] = {}
        self._exploration = self._exploration_points()
        self._visited: set[tuple[int, int]] = set()
        self.confirmation_sweeps_completed = 0

    def _emit(self, kind: str, timestamp_s: float, **details: Any) -> None:
        self.events.append({"sequence": len(self.events) + 1, "event": kind, "timestamp_s": timestamp_s, **details})

    def _exploration_points(self) -> list[tuple[int, int]]:
        points = []
        stride = 1 if self.scheduler == "fixed_serpentine" else self.policy.exploration_stride_cells
        for row, y in enumerate(range(1, self.map.height_cells - 1, stride)):
            xs = list(range(1, self.map.width_cells - 1, stride))
            if row % 2:
                xs.reverse()
            points.extend((x, y) for x in xs if (x, y) not in self.map.occupied_cells)
        return points

    @property
    def coverage(self) -> float:
        if self.executed_coverage is not None:
            return float(self.executed_coverage)
        return 1 if not self._exploration else len(self._visited) / len(self._exploration)

    def _localize(self, obs: Mapping[str, Any], pose: Mapping[str, Any]) -> tuple[float, float, float] | None:
        if obs.get("held") or not obs.get("stable", True) or obs.get("class_name") not in ALLOWED_CLASSES:
            return None
        confidence, depth = float(obs.get("confidence", 0)), float(obs.get("depth_m", 0))
        localized = obs.get("localized_position")
        if localized is not None:
            if confidence < self.policy.stable_confidence: return None
            values = tuple(float(localized[key]) for key in ("east_m", "north_m", "altitude_m"))
            return values if all(math.isfinite(value) for value in values) else None
        bbox, camera = obs.get("bbox"), obs.get("camera_intrinsics", {})
        fx = float(camera.get("fx", 0))
        if confidence < self.policy.stable_confidence or not bbox or fx <= 0 or not .2 <= depth <= 100:
            return None
        center_x = (float(bbox[0]) + float(bbox[2])) / 2
        bearing = math.radians(float(pose.get("yaw_deg", 0))) + math.atan2(center_x - float(camera.get("cx", 0)), fx)
        return (float(pose.get("east_m", 0)) + depth * math.cos(bearing),
                float(pose.get("north_m", 0)) + depth * math.sin(bearing), float(pose.get("altitude_m", 0)))

    def _register(self, obs: Mapping[str, Any], pose: Mapping[str, Any], now: float, *, allow_new_target: bool = True) -> None:
        position = self._localize(obs, pose)
        if position is None:
            return
        source_id, class_name = str(obs.get("tracking_id", "")), str(obs["class_name"])
        ordered = sorted(self.tracks.values(), key=lambda t: t.target_id)
        match = next((t for t in ordered if source_id and source_id in t.source_tracking_ids), None)
        lineage_conflict = match is not None and match.class_name != class_name
        if match is None:
            match = next((t for t in ordered if t.class_name == class_name and now - t.last_seen_s <= self.policy.track_timeout_s
                          and _distance((t.east_m, t.north_m), position[:2]) <= max(
                              self.policy.merge_distance_m,
                              self.policy.preferred_standoff_m,
                          )), None)
        if match is None:
            inspected_radius = max(
                self.policy.merge_distance_m,
                2 * self.policy.preferred_standoff_m,
            )
            match = next((
                t for t in ordered
                if t.inspected and t.class_name == class_name
                and _distance((t.east_m, t.north_m), position[:2]) <= inspected_radius
            ), None)
        conflicts = [match] if lineage_conflict else []
        confidence, depth = float(obs["confidence"]), float(obs["depth_m"])
        if match is None:
            if not allow_new_target:
                return
            target_id = f"visual-target-{self._next_track:04d}"
            self._next_track += 1
            match = DeviceTrack(target_id, class_name, confidence, *position, 1, confidence, depth, now,
                                [source_id] if source_id else [])
            self.tracks[target_id] = match
            self._position_history[target_id] = [position]
            self._emit("target_registered", now, target_id=target_id, class_name=class_name)
        else:
            displacement = _distance((match.east_m, match.north_m), position[:2])
            if displacement > self.policy.merge_distance_m:
                self._emit(
                    "localization_outlier_rejected",
                    now,
                    target_id=match.target_id,
                    displacement_m=displacement,
                    rejection_threshold_m=self.policy.merge_distance_m,
                )
                return
            count = match.observation_count + 1
            history = self._position_history.setdefault(
                match.target_id,
                [(match.east_m, match.north_m, match.altitude_m)],
            )
            history.append(position)
            del history[:-31]
            match.east_m = median(value[0] for value in history)
            match.north_m = median(value[1] for value in history)
            match.altitude_m = median(value[2] for value in history)
            match.observation_count, match.confidence, match.last_seen_s = count, confidence, now
            if confidence >= match.best_confidence:
                match.best_confidence, match.best_depth_m = confidence, depth
            if source_id and source_id not in match.source_tracking_ids:
                match.source_tracking_ids.append(source_id)
        if conflicts:
            match.ambiguous = True
            self._emit("target_class_ambiguous", now, target_id=match.target_id,
                       conflicting_target_ids=[t.target_id for t in conflicts],
                       reason="same_temporal_lineage_class_conflict")

    def _cell(self, east: float, north: float) -> tuple[int, int]:
        return round(east / self.map.resolution_m), round(north / self.map.resolution_m)

    def _clearance(self, cell: tuple[int, int]) -> int:
        return min((abs(cell[0] - x) + abs(cell[1] - y) for x, y in self.map.occupied_cells),
                   default=max(self.map.width_cells, self.map.height_cells))

    def _target_candidate(self, start: tuple[int, int], track: DeviceTrack):
        target = self._cell(track.east_m, track.north_m)
        radius = max(1, round(self.policy.preferred_standoff_m / self.map.resolution_m))
        routes = []
        for degrees in range(0, 360, 45):
            angle = math.radians(degrees)
            goal = round(target[0] + radius * math.cos(angle)), round(target[1] + radius * math.sin(angle))
            path = _astar(self.map, start, goal)
            if path:
                routes.append((path, goal))
        if not routes:
            return None
        path, goal = min(routes, key=lambda item: (len(item[0]), item[1][1], item[1][0]))
        distance = (len(path) - 1) * self.map.resolution_m
        inspected_classes = {t.class_name for t in self.tracks.values() if t.inspected and not t.ambiguous}
        features = {
            "new_class": float(track.class_name not in inspected_classes), "uninspected": float(not track.inspected),
            "confidence_gap": max(0, self.policy.inspection_confidence - track.best_confidence),
            "visibility_improvement": min(1, abs(track.best_depth_m - self.policy.preferred_standoff_m) / self.policy.preferred_standoff_m),
            "path_length": distance, "energy": distance + self.policy.observation_hold_s,
            "risk": float(self._clearance(goal) <= self.policy.low_clearance_cells),
            "repeat_observation": float(max(0, track.observation_count - 1)),
        }
        return sum(self.policy.weights[k] * v for k, v in features.items()), track.target_id, path, features

    def _schedule(self, now: float, pose: Mapping[str, Any], trigger: str) -> None:
        if self.route_active:
            self._emit("scheduler_decision", now, reason="route_active", action="update_tracks_only")
            return
        if self.safety_active:
            self._emit("scheduler_decision", now, reason="safety_replan_priority", action="hold_semantic_route")
            return
        start = self._cell(float(pose.get("east_m", 0)), float(pose.get("north_m", 0)))
        candidates = [candidate for track in sorted(self.tracks.values(), key=lambda t: t.target_id)
                      if not track.inspected and not track.ambiguous
                      for candidate in [self._target_candidate(start, track)] if candidate]
        if self.scheduler == "fixed_serpentine":
            candidates = []
        reachable = [(point, _astar(self.map, start, point)) for point in self._exploration if point not in self._visited]
        reachable = [(point, path) for point, path in reachable if path]
        if reachable:
            unvisited_by_row: dict[int, list[tuple[int, int]]] = {}
            for point, _ in reachable:
                unvisited_by_row.setdefault(point[1], []).append(point)
            bands: list[list[tuple[int, int]]] = []
            for selected_y, points in sorted(unvisited_by_row.items()):
                for point in sorted(points):
                    if not bands or point[1] != bands[-1][-1][1] or point[0] - bands[-1][-1][0] > self.policy.exploration_stride_cells:
                        bands.append([point])
                    else:
                        bands[-1].append(point)
            row = min(
                bands,
                key=lambda band: (
                    min(
                        abs(band[0][0] - start[0]) + abs(band[0][1] - start[1]),
                        abs(band[-1][0] - start[0]) + abs(band[-1][1] - start[1]),
                    ),
                    band[0][1],
                    band[0][0],
                ),
            )
            near_index = 0 if abs(row[0][0] - start[0]) <= abs(row[-1][0] - start[0]) else len(row) - 1
            ordered_row = row if near_index == 0 else list(reversed(row))
            required_total = math.ceil(self.policy.exploration_completion * len(self._exploration))
            needed = max(1, required_total - len(self._visited))
            goal = ordered_row[min(len(ordered_row), needed) - 1]
            entry = _astar(self.map, start, ordered_row[0])
            scan = _astar(self.map, ordered_row[0], goal)
            path = entry[:-1] + scan if entry and scan else None
        else:
            path = None
        if path:
            distance = (len(path) - 1) * self.map.resolution_m
            coverage_gain = len(set(path).intersection(self._exploration).difference(self._visited))
            candidates.append((coverage_gain + FROZEN_WEIGHTS["path_length"] * distance + FROZEN_WEIGHTS["energy"] * distance,
                               f"explore:{goal[0]}:{goal[1]}", path,
                               {"exploration_gain": coverage_gain, "path_length": distance}))
        if not candidates:
            self._emit("scheduler_decision", now, reason="no_reachable_candidate", action="hold")
            return
        if self.scheduler == "nearest_target_first":
            equipment = [item for item in candidates if not item[1].startswith("explore:")]
            selected = equipment or candidates
            score, candidate_id, path, features = min(
                selected,
                key=lambda c: (len(c[2]), c[1]),
            )
        else:
            score, candidate_id, path, features = sorted(candidates, key=lambda c: (-c[0], c[1].startswith("explore:"), c[1]))[0]
        kind = "exploration" if candidate_id.startswith("explore:") else "equipment_observation"
        self._visited.update(set(path).intersection(self._exploration))
        waypoints = [{"east_m": c[0] * self.map.resolution_m, "north_m": c[1] * self.map.resolution_m,
                      "altitude_m": float(pose.get("altitude_m", 1.5))} for c in path]
        decision = {"decision_id": f"decision-{len(self.decisions) + 1:04d}", "timestamp_s": now,
                    "trigger": trigger, "candidate_id": candidate_id, "kind": kind, "utility": round(score, 8),
                    "scheduler": self.scheduler,
                    "features": features, "replacement_waypoints": waypoints, "transit_cells": [list(c) for c in path]}
        self.decisions.append(decision)
        self._emit("scheduler_decision", now, decision_id=decision["decision_id"], candidate_id=candidate_id,
                   utility=decision["utility"])
        self._emit("semantic_route_replaced", now, decision_id=decision["decision_id"], waypoint_count=len(waypoints))

    def _schedule_confirmation_sweep(
        self, now: float, pose: Mapping[str, Any], trigger: str
    ) -> bool:
        start = self._cell(
            float(pose.get("east_m", 0)), float(pose.get("north_m", 0))
        )
        band = max(1, self.policy.exploration_stride_cells)
        remaining = {
            point for point in self._exploration
            if min(
                point[0], point[1],
                self.map.width_cells - 1 - point[0],
                self.map.height_cells - 1 - point[1],
            ) <= band
        }
        route, current = [], start
        anchor_indices = set()
        while remaining and len(route) < self.policy.confirmation_max_route_cells:
            target = min(
                remaining,
                key=lambda point: (
                    abs(point[0] - current[0]) + abs(point[1] - current[1]),
                    point[1], point[0],
                ),
            )
            segment = _astar(self.map, current, target)
            remaining.remove(target)
            if not segment:
                continue
            available = self.policy.confirmation_max_route_cells - len(route)
            addition = segment if not route else segment[1:]
            route.extend(addition[:available])
            current = route[-1]
            if current == target:
                anchor_indices.add(len(route) - 1)
        if not route:
            self._emit("confirmation_sweep_unavailable", now, reason="no_reachable_boundary_route")
            return False
        waypoints = [
            {
                "east_m": cell[0] * self.map.resolution_m,
                "north_m": cell[1] * self.map.resolution_m,
                "altitude_m": float(pose.get("altitude_m", 1.5)),
                "confirmation_anchor": index in anchor_indices,
            }
            for index, cell in enumerate(route)
        ]
        decision = {
            "decision_id": f"decision-{len(self.decisions) + 1:04d}",
            "timestamp_s": now,
            "trigger": trigger,
            "candidate_id": f"confirmation:{self.confirmation_sweeps_completed + 1}",
            "kind": "exploration",
            "utility": 0.0,
            "scheduler": self.scheduler,
            "features": {
                "confirmation_sweep": True,
                "route_cells": len(route),
                "anchor_count": len(anchor_indices),
            },
            "replacement_waypoints": waypoints,
            "transit_cells": [list(cell) for cell in route],
        }
        self.decisions.append(decision)
        self.confirmation_sweeps_completed += 1
        self._emit(
            "confirmation_sweep_started", now,
            decision_id=decision["decision_id"],
            pass_index=self.confirmation_sweeps_completed,
            route_cells=len(route), anchor_count=len(anchor_indices),
        )
        self._emit(
            "semantic_route_replaced", now,
            decision_id=decision["decision_id"], waypoint_count=len(waypoints),
        )
        return True

    def process(self, record: Mapping[str, Any]) -> None:
        if self.terminated:
            return
        trigger = str(record.get("trigger", ""))
        if trigger not in ALLOWED_TRIGGERS:
            raise ValueError(f"unsupported active inspection trigger: {trigger}")
        now, elapsed = float(record.get("timestamp_s", 0)), float(record.get("elapsed_s", record.get("timestamp_s", 0)))
        if elapsed >= self.policy.mission_budget_s:
            self.terminated = True
            self._emit("mission_budget_exhausted", now, elapsed_s=elapsed)
            return
        if trigger == "safety_replan_active":
            self.safety_active = True
        elif trigger == "safety_replan_cleared":
            self.safety_active = False
        pose = record.get("vehicle_pose", {})
        for obs in record.get("observations", []):
            class_name = str(obs.get("class_name", ""))
            self._register(
                obs,
                pose,
                now,
                allow_new_target=(
                    not self.route_active
                    or not any(track.class_name == class_name for track in self.tracks.values())
                ),
            )
        completed = {str(item) for item in record.get("completed_tracking_ids", [])}
        completed_targets = {str(item) for item in record.get("completed_target_ids", [])}
        for track in sorted(self.tracks.values(), key=lambda t: t.target_id):
            if (
                track.target_id in completed_targets
                or completed.intersection(track.source_tracking_ids)
            ) and not track.ambiguous and not track.inspected:
                track.inspected = True
                self._emit("target_inspected", now, target_id=track.target_id)
        valid = [track for track in self.tracks.values() if not track.ambiguous]
        completion_feedback = trigger in {"waypoint_reached", "target_completed"}
        if (
            completion_feedback
            and self.coverage >= self.policy.exploration_completion
            and all(track.inspected for track in valid)
        ):
            if self.confirmation_sweeps_completed < self.policy.confirmation_sweep_passes:
                if not self._schedule_confirmation_sweep(now, pose, trigger):
                    self.terminated = True
                    self._emit("exploration_completed", now, coverage=self.coverage)
            else:
                self.terminated = True
                self._emit("exploration_completed", now, coverage=self.coverage)
        else:
            self._schedule(now, pose, trigger)

    def result(self, observation_identity: str) -> dict[str, Any]:
        result = {"schema_version": 1, "runtime_mode": "active_semantic_inspection",
                  "identities": {"runtime_map": self.map.map_identity, "model": self.policy.model_artifact_identity,
                                 "policy": self.policy.artifact_identity, "observation_stream": observation_identity},
                  "tracks": [asdict(t) for t in sorted(self.tracks.values(), key=lambda t: t.target_id)],
                  "decisions": self.decisions, "events": self.events,
                  "exploration_coverage": round(self.coverage, 8), "terminated": self.terminated}
        result["artifact_identity"] = object_identity(result)
        return result


def run_active_inspection(map_value: Mapping[str, Any], policy_value: Mapping[str, Any],
                          observations: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(observations)
    planner = ActiveInspectionPlanner(RuntimeMap.from_mapping(map_value), ActiveInspectionPolicy.from_mapping(policy_value))
    for row in rows:
        planner.process(row)
    return planner.result(object_identity(rows))


def evaluate_plan(plan: Mapping[str, Any], truth_map: Mapping[str, Any]) -> dict[str, Any]:
    """Offline truth evaluation. This function is not called by the runtime."""
    truth = []
    for obj in truth_map.get("objects", []):
        class_name = obj.get("class_name", obj.get("asset_id"))
        if class_name in ALLOWED_CLASSES:
            pose = obj.get("pose", obj)
            truth.append((str(obj.get("object_id", len(truth))), class_name,
                          float(pose.get("east_m", pose.get("x_m", 0))), float(pose.get("north_m", pose.get("y_m", 0)))))
    matches, correct = [], 0
    for track in plan.get("tracks", []):
        nearest = min(truth, key=lambda t: _distance((track["east_m"], track["north_m"]), (t[2], t[3])), default=None)
        if nearest and _distance((track["east_m"], track["north_m"]), (nearest[2], nearest[3])) <= 3:
            matches.append(nearest[0])
            correct += int(track["class_name"] == nearest[1])
    recall = len(set(matches)) / len(truth) if truth else 1
    accuracy = correct / len(matches) if matches else 0
    distance = sum(max(0, len(d.get("transit_cells", [])) - 1) for d in plan.get("decisions", [])) * float(truth_map.get("resolution_m", 1))
    return {"schema_version": 1, "evaluation_scope": "offline_truth_only", "target_count": len(truth),
            "matched_target_count": len(set(matches)), "target_recall": round(recall, 8),
            "class_accuracy": round(accuracy, 8), "flight_distance_m": round(distance, 8),
            "acceptance": {"recall_at_least_95_percent": recall >= .95,
                           "class_accuracy_at_least_90_percent": accuracy >= .90}}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"observation line {number} must be an object")
            rows.append(row)
    return rows
