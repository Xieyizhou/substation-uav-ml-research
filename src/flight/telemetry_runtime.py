"""MAVSDK telemetry subscriptions and structured CSV logging."""

import asyncio
import csv
from datetime import datetime, timezone

from src.flight.async_runtime import cancel_tasks
from src.flight.flight_config import LOG_DIR
from src.flight.flight_state import (
    local_position,
    local_velocity,
    target_errors,
    update_latest,
)
from src.flight.live_telemetry import (
    configured_snapshot_path, snapshot_record, write_snapshot,
)
from src.flight.perception_response import current_perception_detection
from src.flight.replanning_controller import reset_replan_event_fields
from src.logging.flight_logger import (
    TELEMETRY_CSV_HEADER,
    build_telemetry_log_row,
    perception_csv_values,
    replan_csv_values,
)


LOGGER_SHUTDOWN_TIMEOUT_S = 10.0


def configure_logger_timeout(timeout_s):
    global LOGGER_SHUTDOWN_TIMEOUT_S
    LOGGER_SHUTDOWN_TIMEOUT_S = timeout_s


async def watch_connection_state(drone, latest):
    async for state in drone.core.connection_state():
        update_latest(latest, "connected", bool(state.is_connected))


async def watch_position_velocity_ned(drone, latest):
    async for position_velocity in drone.telemetry.position_velocity_ned():
        update_latest(latest, "position_velocity", position_velocity)


async def watch_attitude(drone, latest):
    async for attitude in drone.telemetry.attitude_euler():
        update_latest(latest, "attitude", attitude)


async def watch_battery(drone, latest):
    async for battery in drone.telemetry.battery():
        update_latest(latest, "battery", battery)


async def watch_flight_mode(drone, latest):
    async for flight_mode in drone.telemetry.flight_mode():
        update_latest(latest, "flight_mode", flight_mode)


async def watch_armed(drone, latest):
    async for armed in drone.telemetry.armed():
        update_latest(latest, "armed", armed)


async def watch_in_air(drone, latest):
    async for in_air in drone.telemetry.in_air():
        update_latest(latest, "in_air", in_air)


async def log_telemetry(
    drone,
    stop_event,
    log_path,
    latest,
    phase_state,
    target_state,
    planner_info,
    perception_config,
    replan_config,
    replan_state,
    perception_detector=None,
):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = configured_snapshot_path()
    start_time = datetime.now(timezone.utc)
    watcher_tasks = [
        asyncio.create_task(watch_connection_state(drone, latest), name="connection-state"),
        asyncio.create_task(
            watch_position_velocity_ned(drone, latest), name="position-velocity"
        ),
        asyncio.create_task(watch_attitude(drone, latest), name="attitude"),
        asyncio.create_task(watch_battery(drone, latest), name="battery"),
        asyncio.create_task(watch_flight_mode(drone, latest), name="flight-mode"),
        asyncio.create_task(watch_armed(drone, latest), name="armed"),
        asyncio.create_task(watch_in_air(drone, latest), name="in-air"),
    ]
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(TELEMETRY_CSV_HEADER)
        try:
            while not stop_event.is_set():
                for task in watcher_tasks:
                    if not task.done():
                        continue
                    if task.cancelled():
                        raise RuntimeError(
                            f"Telemetry watcher {task.get_name()} was cancelled unexpectedly"
                        )
                    error = task.exception()
                    if error is not None:
                        raise RuntimeError(
                            f"Telemetry watcher {task.get_name()} failed"
                        ) from error
                    raise RuntimeError(
                        f"Telemetry watcher {task.get_name()} stopped unexpectedly"
                    )
                now = datetime.now(timezone.utc)
                position = local_position(latest)
                velocity = local_velocity(latest)
                attitude = latest["attitude"]
                battery = latest["battery"]
                target = target_state.copy()
                error = target_errors(position, target)
                detection = current_perception_detection(
                    perception_config,
                    perception_detector,
                    position,
                    attitude,
                    timestamp_utc=now.isoformat(),
                    elapsed_s=round((now - start_time).total_seconds(), 3),
                    replan_config=replan_config,
                    velocity=velocity,
                )
                writer.writerow(
                    build_telemetry_log_row(
                        now,
                        start_time,
                        phase_state,
                        target,
                        position,
                        velocity,
                        attitude,
                        battery,
                        latest["flight_mode"],
                        latest["armed"],
                        planner_info,
                        error,
                        perception_csv_values(perception_config, detection),
                        replan_csv_values(replan_state),
                    )
                )
                write_snapshot(
                    snapshot_path,
                    snapshot_record(
                        now, start_time, phase_state, target, position, velocity,
                        attitude, latest,
                    ),
                )
                if replan_state.get("replan_triggered") or replan_state.get(
                    "replan_route_replaced"
                ):
                    reset_replan_event_fields(replan_state)
                log_file.flush()
                await asyncio.sleep(0.2)
        finally:
            pending_watchers = await cancel_tasks(
                watcher_tasks, LOGGER_SHUTDOWN_TIMEOUT_S
            )
            if pending_watchers:
                names = ", ".join(task.get_name() for task in pending_watchers)
                raise TimeoutError(
                    f"Telemetry watchers did not stop before timeout: {names}"
                )
