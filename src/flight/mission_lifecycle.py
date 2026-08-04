"""Connection, logging, mission supervision, and final cleanup."""

import asyncio
import contextlib

from src.flight.async_runtime import cancel_tasks
from src.flight.mission_events import MissionEventWriter
from src.flight.perception_response import DangerObstacleDetected
from src.flight.replanning_controller import empty_replan_state


async def execute_flight(
    system_address,
    waypoints,
    planner_info,
    perception_config,
    replan_config,
    settings,
    services,
    perception_detector=None,
    return_home=False,
    visual_mission_events=None,
):
    drone = services["system_factory"]()
    log_path = services["make_log_path"]()
    latest = {
        "connected": None,
        "position_velocity": None,
        "attitude": None,
        "battery": None,
        "flight_mode": None,
        "armed": None,
        "in_air": None,
        "updated_at": {},
    }
    replan_state = empty_replan_state()
    replan_state["replan_mode"] = replan_config.get("mode", "log_only")
    phase_state = {"phase": "connecting", "route_direction": "none"}
    event_writer = (
        MissionEventWriter(visual_mission_events)
        if visual_mission_events is not None
        else None
    )
    if event_writer is not None:
        phase_state["_event_publisher"] = event_writer
        event_writer.publish(
            "mission_started",
            phase=phase_state["phase"],
            route_direction=phase_state["route_direction"],
        )
    target_state = {"name": "", "north_m": 0.0, "east_m": 0.0, "down_m": 0.0}
    stop_logging = asyncio.Event()
    telemetry_task = None
    mission_task = None
    pending_error = None
    landing_confirmed = None
    status_path = services["write_run_status"](log_path, "starting", "connecting")
    print(f"Run status: {status_path}")
    try:
        if perception_detector is not None and hasattr(perception_detector, "start"):
            print(
                "Starting perception source: "
                f"{perception_config.get('source', 'map_baseline')}"
            )
            await perception_detector.start()
            await perception_detector.wait_ready(
                perception_config.get("sensor_startup_timeout_s", 5.0)
            )
        print(f"Connecting to PX4 SITL with MAVSDK at {system_address}...")
        try:
            await asyncio.wait_for(
                drone.connect(system_address=system_address),
                timeout=settings.connection_timeout_s,
            )
        except asyncio.TimeoutError as error:
            raise TimeoutError(
                f"Timed out waiting {settings.connection_timeout_s:g}s "
                "for MAVSDK connection startup"
            ) from error
        await services["wait_for_connection"](drone, settings.connection_timeout_s)
        await services["wait_for_position_ready"](
            drone, settings.position_ready_timeout_s
        )
        print(f"Starting telemetry log: {log_path}")
        telemetry_task = asyncio.create_task(
            services["log_telemetry"](
                drone,
                stop_logging,
                log_path,
                latest,
                phase_state,
                target_state,
                planner_info,
                perception_config,
                replan_config,
                replan_state,
                perception_detector,
            ),
            name="telemetry-logger",
        )
        services["write_run_status"](log_path, "running", phase_state["phase"])
        mission_runner = services.get(
            "mission_runner", services["fly_astar_waypoints"]
        )
        mission_task = asyncio.create_task(
            mission_runner(
                drone,
                latest,
                phase_state,
                target_state,
                waypoints,
                perception_config,
                perception_detector,
                replan_config,
                replan_state,
                return_home=return_home,
            ),
            name="flight-mission",
        )
        done, _ = await asyncio.wait(
            {mission_task, telemetry_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if telemetry_task in done:
            if not mission_task.done():
                mission_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await mission_task
            logger_error = telemetry_task.exception()
            if logger_error is not None:
                raise RuntimeError("Telemetry logger failed during flight") from logger_error
            raise RuntimeError("Telemetry logger stopped unexpectedly during flight")
        await mission_task
        landing_confirmed = phase_state["phase"] == "landed"
        if not landing_confirmed:
            raise RuntimeError("Mission ended without confirmed landing")
        services["write_run_status"](
            log_path, "completed", phase_state["phase"], landing_confirmed=True
        )
        if event_writer is not None:
            event_writer.publish(
                "mission_completed",
                status="completed",
                phase=phase_state["phase"],
                landing_confirmed=True,
            )
    except Exception as error:
        print(f"Flight error: {error}")
        pending_error = error
        phase_name = (
            "landing_after_danger"
            if isinstance(error, DangerObstacleDetected)
            else "landing_after_error"
        )
        if phase_state["phase"] == "landed":
            landing_confirmed = True
        elif telemetry_task is not None:
            landing_confirmed = await services["attempt_safe_landing"](
                drone, latest, phase_state, phase_name
            )
        services["write_run_status"](
            log_path,
            "failed",
            phase_state["phase"],
            message=f"{type(error).__name__}: {error}",
            landing_confirmed=landing_confirmed,
        )
        if event_writer is not None:
            event_writer.publish(
                "mission_failed",
                status="failed",
                phase=phase_state["phase"],
                landing_confirmed=landing_confirmed,
                failure_type=type(error).__name__,
                message=str(error),
            )
    finally:
        if telemetry_task is not None:
            print("Stopping telemetry logging...")
            stop_logging.set()
            try:
                await asyncio.wait_for(
                    asyncio.shield(telemetry_task),
                    timeout=settings.logger_shutdown_timeout_s,
                )
            except Exception as error:
                if not telemetry_task.done():
                    await cancel_tasks(
                        [telemetry_task], settings.logger_shutdown_timeout_s
                    )
                if pending_error is None:
                    pending_error = RuntimeError(
                        "Telemetry logger did not shut down cleanly"
                    )
                    pending_error.__cause__ = error
                    services["write_run_status"](
                        log_path,
                        "failed",
                        phase_state["phase"],
                        message=f"{type(error).__name__}: {error}",
                        landing_confirmed=landing_confirmed,
                    )
            if log_path.exists():
                print(f"Telemetry log saved to {log_path}")
        if perception_detector is not None and hasattr(perception_detector, "stop"):
            try:
                await perception_detector.stop()
            except Exception as error:
                if pending_error is None:
                    pending_error = RuntimeError(
                        f"Perception source did not stop cleanly: {error}"
                    )
    if pending_error is not None:
        raise pending_error
    print("Done.")
