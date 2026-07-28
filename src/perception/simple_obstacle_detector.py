import math

from src.planner.obstacle_config import build_obstacle_map, get_resolution_altitude, get_start_goal


def normalize_angle_deg(angle_deg):
    """Return an angle in the range [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


class SimpleObstacleDetector:
    """Map-based prototype for a simple forward obstacle detector.

    This is not real LiDAR, depth, or camera perception. It uses the known
    obstacle map to emulate what a short-range onboard detector might report.
    """

    def __init__(
        self,
        obstacle_config,
        obstacle_map=None,
        detection_range_m=4.0,
        warning_distance_m=2.0,
        danger_distance_m=1.0,
        detection_fov_deg=90.0,
        use_inflated_cells=True,
        use_raw_cells=True,
        flight_altitude_m=None,
    ):
        self.obstacle_config = obstacle_config
        # Detection range is the outer sensor range. Obstacles farther away are
        # ignored by this prototype detector.
        self.detection_range_m = float(detection_range_m)
        # Warning distance means "close enough to require caution", but not
        # yet the closest emergency band.
        self.warning_distance_m = float(warning_distance_m)
        # Danger distance means "very close". Later stages can use this for
        # emergency stop, landing, or local replanning.
        self.danger_distance_m = float(danger_distance_m)
        self.detection_fov_deg = float(detection_fov_deg)
        self.use_inflated_cells = bool(use_inflated_cells)
        self.use_raw_cells = bool(use_raw_cells)
        self.resolution_m, config_altitude_m = get_resolution_altitude(obstacle_config)

        if obstacle_map is None:
            start_cell, goal_cell = get_start_goal(obstacle_config)
            obstacle_map = build_obstacle_map(
                obstacle_config,
                flight_altitude_m=(
                    config_altitude_m if flight_altitude_m is None else flight_altitude_m
                ),
                start_cell=start_cell,
                goal_cell=goal_cell,
            )
        self.obstacle_map = obstacle_map
        self.obstacle_types_by_name = self._obstacle_types_by_name(obstacle_config)
        self.sensor_cells = self._build_sensor_cells()

    def _obstacle_types_by_name(self, obstacle_config):
        types_by_name = {}
        for obstacle in obstacle_config.get("obstacles", []):
            name = obstacle.get("name", "<unnamed>")
            types_by_name[name] = obstacle.get("visual_category") or obstacle.get("type", "")
        return types_by_name

    def _names_for_cell(self, layer, cell):
        if layer == "raw":
            name_map = self.obstacle_map.get("raw_obstacle_cell_to_name", {})
        else:
            name_map = self.obstacle_map.get("inflated_obstacle_cell_to_name", {})
        names_text = name_map.get(cell, "")
        names = [name for name in names_text.split(",") if name]
        return names or ["unknown"]

    def _type_text_for_names(self, names):
        types = []
        for name in names:
            obstacle_type = self.obstacle_types_by_name.get(name, "")
            if obstacle_type and obstacle_type not in types:
                types.append(obstacle_type)
        return ",".join(types)

    def _cell_to_local_center(self, cell):
        grid_x, grid_y = cell
        return (
            (grid_x + 0.5) * self.resolution_m,
            (grid_y + 0.5) * self.resolution_m,
        )

    def _build_sensor_cells(self):
        cells = []
        seen = set()

        if self.use_raw_cells:
            for cell in sorted(self.obstacle_map.get("raw_obstacle_cells", set())):
                names = self._names_for_cell("raw", cell)
                east_m, north_m = self._cell_to_local_center(cell)
                cells.append(
                    {
                        "obstacle_layer": "raw",
                        "grid_x": int(cell[0]),
                        "grid_y": int(cell[1]),
                        "obstacle_east_m": east_m,
                        "obstacle_north_m": north_m,
                        "obstacle_name": ",".join(names),
                        "obstacle_type": self._type_text_for_names(names),
                    }
                )
                seen.add(("raw", cell))

        if self.use_inflated_cells:
            for cell in sorted(self.obstacle_map.get("inflated_blocking_cells", set())):
                key = ("inflated", cell)
                if key in seen:
                    continue
                names = self._names_for_cell("inflated", cell)
                east_m, north_m = self._cell_to_local_center(cell)
                cells.append(
                    {
                        "obstacle_layer": "inflated",
                        "grid_x": int(cell[0]),
                        "grid_y": int(cell[1]),
                        "obstacle_east_m": east_m,
                        "obstacle_north_m": north_m,
                        "obstacle_name": ",".join(names),
                        "obstacle_type": self._type_text_for_names(names),
                    }
                )

        return cells

    def _inside_fov(self, relative_bearing_deg, yaw_deg):
        # PX4 local yaw is treated as a compass-style heading in local NED:
        # 0 deg points north, +90 deg points east. A bearing is computed with
        # atan2(east_delta, north_delta), so it uses the same convention.
        #
        # If yaw is missing, the detector uses 360-degree mode. This is useful
        # during startup or when attitude telemetry is temporarily unavailable.
        if yaw_deg is None:
            return True
        return abs(relative_bearing_deg) <= self.detection_fov_deg / 2.0

    def classify_risk(self, nearest_distance_m):
        """Classify risk from the nearest detected obstacle distance.

        clear: no obstacle inside detection range.
        detected: obstacle is inside sensor range but farther than warning.
        warning: obstacle is within warning distance but farther than danger.
        danger: obstacle is within danger distance.
        """
        distance_m = safe_float(nearest_distance_m)
        if distance_m is None:
            return "clear"
        if distance_m <= self.danger_distance_m:
            return "danger"
        if distance_m <= self.warning_distance_m:
            return "warning"
        if distance_m <= self.detection_range_m:
            return "detected"
        return "clear"

    def detect(self, local_north_m, local_east_m, yaw_deg=None, altitude_m=None):
        local_north_m = safe_float(local_north_m)
        local_east_m = safe_float(local_east_m)
        yaw_deg = safe_float(yaw_deg)
        _ = altitude_m

        if local_north_m is None or local_east_m is None:
            return {
                "detected_obstacle": False,
                "risk_level": "clear",
                "closest_obstacle_name": "",
                "closest_obstacle_distance_m": None,
                "closest_obstacle_bearing_deg": None,
                "closest_obstacle_in_detection_range": False,
                "closest_obstacle_in_fov": False,
                "nearest_obstacle_name": "",
                "nearest_obstacle_layer": "",
                "nearest_obstacle_distance_m": None,
                "nearest_obstacle_bearing_deg": None,
                "detected_obstacle_count": 0,
                "warning_distance_m": self.warning_distance_m,
                "danger_distance_m": self.danger_distance_m,
                "detected": False,
                "closest_obstacle": None,
                "nearest_obstacle": None,
                "detected_obstacles": [],
            }

        detected_obstacles = []
        closest_obstacle = None
        for cell_info in self.sensor_cells:
            east_delta = cell_info["obstacle_east_m"] - local_east_m
            north_delta = cell_info["obstacle_north_m"] - local_north_m
            distance_m = math.hypot(east_delta, north_delta)

            absolute_bearing_deg = math.degrees(math.atan2(east_delta, north_delta))
            if yaw_deg is None:
                relative_bearing_deg = None
            else:
                relative_bearing_deg = normalize_angle_deg(absolute_bearing_deg - yaw_deg)

            in_detection_range = distance_m <= self.detection_range_m
            in_fov = self._inside_fov(relative_bearing_deg, yaw_deg)
            obstacle_state = {
                **cell_info,
                "distance_m": distance_m,
                "bearing_deg_relative": relative_bearing_deg,
                "in_detection_range": in_detection_range,
                "in_fov": in_fov,
                "detected": bool(in_detection_range and in_fov),
            }
            if closest_obstacle is None or distance_m < closest_obstacle["distance_m"]:
                closest_obstacle = obstacle_state

            if not in_detection_range:
                continue

            if not in_fov:
                continue

            detected_obstacles.append(obstacle_state)

        detected_obstacles.sort(key=lambda item: item["distance_m"])
        nearest = detected_obstacles[0] if detected_obstacles else None
        risk_level = self.classify_risk(nearest["distance_m"] if nearest else None)

        return {
            "detected_obstacle": bool(detected_obstacles),
            "risk_level": risk_level,
            "closest_obstacle_name": (
                closest_obstacle["obstacle_name"] if closest_obstacle else ""
            ),
            "closest_obstacle_distance_m": (
                closest_obstacle["distance_m"] if closest_obstacle else None
            ),
            "closest_obstacle_bearing_deg": (
                closest_obstacle["bearing_deg_relative"] if closest_obstacle else None
            ),
            "closest_obstacle_in_detection_range": (
                bool(closest_obstacle["in_detection_range"]) if closest_obstacle else False
            ),
            "closest_obstacle_in_fov": (
                bool(closest_obstacle["in_fov"]) if closest_obstacle else False
            ),
            "nearest_obstacle_name": nearest["obstacle_name"] if nearest else "",
            "nearest_obstacle_layer": nearest["obstacle_layer"] if nearest else "",
            "nearest_obstacle_distance_m": nearest["distance_m"] if nearest else None,
            "nearest_obstacle_bearing_deg": (
                nearest["bearing_deg_relative"] if nearest else None
            ),
            "detected_obstacle_count": len(detected_obstacles),
            "warning_distance_m": self.warning_distance_m,
            "danger_distance_m": self.danger_distance_m,
            "detected": bool(detected_obstacles),
            "closest_obstacle": closest_obstacle,
            "nearest_obstacle": nearest,
            "detected_obstacles": detected_obstacles,
        }
