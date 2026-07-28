"""Argument parser construction for the standalone flight runner."""

import argparse
from pathlib import Path

from src.flight.flight_defaults import (
    CONNECTION_TIMEOUT_S,
    DEFAULT_SYSTEM_ADDRESS,
    LANDING_TIMEOUT_S,
    LOGGER_SHUTDOWN_TIMEOUT_S,
    MAX_HORIZONTAL_SPEED_M_S,
    MIN_RISK_SPEED_M_S,
    POSITION_READY_TIMEOUT_S,
    REACHED_HORIZONTAL_ERROR_M,
    RETURN_SPEED_SCALE,
    TELEMETRY_TIMEOUT_S,
    TURN_SETTLE_S,
)


def build_argument_parser():
    """Build the flight runner parser without resolving the selected map."""
    parser = argparse.ArgumentParser(
        description="Plan an A* grid path and optionally fly it in PX4 SITL."
    )
    parser.add_argument(
        "--system-address",
        default=DEFAULT_SYSTEM_ADDRESS,
        help=f"MAVSDK system address. Default: {DEFAULT_SYSTEM_ADDRESS}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and save a preview without connecting to PX4 or flying.",
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Print a short task summary instead of full obstacle-cell diagnostics.",
    )
    parser.add_argument(
        "--allow-diagonal",
        action="store_true",
        help="Allow diagonal moves in the A* grid planner.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Meters per grid cell. Default: 1.0",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=None,
        help="Target altitude above takeoff in meters. Overrides obstacle config altitude_m.",
    )
    map_source = parser.add_mutually_exclusive_group()
    map_source.add_argument(
        "--obstacle-config",
        type=Path,
        help=(
            "JSON obstacle config override. When omitted, use the obstacle config "
            "paired with the map selected by `python main.py map`."
        ),
    )
    map_source.add_argument(
        "--use-built-in-grid",
        action="store_true",
        help="Use the legacy built-in 10 x 10 grid instead of the selected test map.",
    )
    parser.add_argument(
        "--return-home",
        action="store_true",
        help=(
            "After reaching the A* goal, fly the reversed A* waypoint path "
            "back to the start before landing. This does not use PX4 RTL."
        ),
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=MAX_HORIZONTAL_SPEED_M_S,
        help=f"Maximum horizontal A* speed in m/s. Default: {MAX_HORIZONTAL_SPEED_M_S}",
    )
    parser.add_argument(
        "--waypoint-acceptance",
        type=float,
        default=REACHED_HORIZONTAL_ERROR_M,
        help=f"Horizontal waypoint acceptance radius in meters. Default: {REACHED_HORIZONTAL_ERROR_M}",
    )
    parser.add_argument(
        "--waypoint-timeout",
        default="auto",
        help=(
            "Per-waypoint timeout in seconds, or 'auto' to estimate from waypoint "
            "distance, speed, return speed scale, and perception risk action. Default: auto"
        ),
    )
    parser.add_argument(
        "--turn-settle",
        type=float,
        default=TURN_SETTLE_S,
        help=f"Seconds to hover after reaching each waypoint. Default: {TURN_SETTLE_S}",
    )
    parser.add_argument(
        "--return-speed-scale",
        type=float,
        default=RETURN_SPEED_SCALE,
        help=f"Return route speed multiplier. Default: {RETURN_SPEED_SCALE}",
    )
    parser.add_argument(
        "--enable-perception",
        action="store_true",
        help="Enable the selected local perception source.",
    )
    parser.add_argument(
        "--perception-source",
        choices=["map_baseline", "gazebo_lidar_2d", "replay"],
        default="map_baseline",
        help="Perception input source. Default: map_baseline",
    )
    parser.add_argument(
        "--sensor-topic",
        default="auto",
        help="Gazebo LiDAR topic or 'auto'. Default: auto",
    )
    parser.add_argument(
        "--sensor-replay",
        type=Path,
        help="JSONL scan recording used by replay perception.",
    )
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        help="Optional deterministic sensor-noise and outage manifest.",
    )
    parser.add_argument(
        "--sensor-startup-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the first healthy sensor frame. Default: 5",
    )
    parser.add_argument(
        "--sensor-stale-after",
        type=float,
        default=0.5,
        help="Maximum LiDAR frame age before fail-safe landing. Default: 0.5",
    )
    parser.add_argument(
        "--risk-model",
        default="geometric",
        help="Risk model: geometric or an ONNX model path. Default: geometric",
    )
    parser.add_argument(
        "--risk-fusion",
        choices=["ml_only", "safety_max"],
        default="safety_max",
        help="How ONNX risk combines with geometry. Default: safety_max",
    )
    parser.add_argument(
        "--equipment-model",
        type=Path,
        help="Optional YOLO/ONNX equipment model used by semantic research tasks.",
    )
    parser.add_argument(
        "--detection-range",
        type=float,
        default=4.0,
        help="Perception detection range in meters. Default: 4.0",
    )
    parser.add_argument(
        "--detection-fov",
        type=float,
        default=90.0,
        help="Forward perception field of view in degrees. Default: 90",
    )
    parser.add_argument(
        "--warning-distance",
        type=float,
        default=2.0,
        help="Distance in meters for warning-level perception risk. Default: 2.0",
    )
    parser.add_argument(
        "--danger-distance",
        type=float,
        default=1.0,
        help="Distance in meters for danger-level perception risk. Default: 1.0",
    )
    parser.add_argument(
        "--risk-action",
        choices=["log_only", "slow_down", "stop_and_land"],
        default="log_only",
        help="Optional behavior for perception risk. Default: log_only",
    )
    parser.add_argument(
        "--min-risk-speed",
        type=float,
        default=MIN_RISK_SPEED_M_S,
        help=(
            "Minimum horizontal commanded speed in m/s when --risk-action slow_down "
            f"is reducing speed. Default: {MIN_RISK_SPEED_M_S}"
        ),
    )
    parser.add_argument(
        "--perception-use-inflated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inflated planning cells for simulated perception. Default: true",
    )
    parser.add_argument(
        "--perception-use-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use raw physical obstacle cells for simulated perception. Default: true",
    )
    parser.add_argument(
        "--enable-local-replan",
        action="store_true",
        help=(
            "Attempt local A* replans when perception risk reaches the configured "
            "threshold."
        ),
    )
    parser.add_argument(
        "--replan-mode",
        choices=["log_only", "active"],
        default="log_only",
        help="Local replanning mode. Default: log_only",
    )
    parser.add_argument(
        "--replan-risk-level",
        choices=["detected", "warning", "danger"],
        default="danger",
        help="Minimum perception risk level that triggers a local replan attempt. Default: danger",
    )
    parser.add_argument(
        "--replan-cooldown",
        type=float,
        default=5.0,
        help="Minimum seconds between local replan attempts. Default: 5.0",
    )
    parser.add_argument(
        "--dynamic-obstacle-inflation",
        type=int,
        default=1,
        help="Grid-cell inflation applied to perception-derived dynamic obstacles. Default: 1",
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=5,
        help="Maximum local replan attempts per flight. Default: 5",
    )
    parser.add_argument(
        "--connection-timeout",
        type=float,
        default=CONNECTION_TIMEOUT_S,
        help=f"Seconds to wait for a PX4 connection. Default: {CONNECTION_TIMEOUT_S:g}",
    )
    parser.add_argument(
        "--position-ready-timeout",
        type=float,
        default=POSITION_READY_TIMEOUT_S,
        help=f"Seconds to wait for PX4 local-position readiness. Default: {POSITION_READY_TIMEOUT_S:g}",
    )
    parser.add_argument(
        "--telemetry-timeout",
        type=float,
        default=TELEMETRY_TIMEOUT_S,
        help=f"Maximum age of critical telemetry during flight. Default: {TELEMETRY_TIMEOUT_S:g}",
    )
    parser.add_argument(
        "--landing-timeout",
        type=float,
        default=LANDING_TIMEOUT_S,
        help=f"Seconds to wait for confirmed landing. Default: {LANDING_TIMEOUT_S:g}",
    )
    parser.add_argument(
        "--logger-shutdown-timeout",
        type=float,
        default=LOGGER_SHUTDOWN_TIMEOUT_S,
        help=f"Seconds to wait for telemetry logger shutdown. Default: {LOGGER_SHUTDOWN_TIMEOUT_S:g}",
    )
    return parser
