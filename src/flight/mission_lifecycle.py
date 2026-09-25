"""Connection, logging, mission supervision, and final cleanup."""

import asyncio
import contextlib

from src.flight.async_runtime import cancel_tasks
from src.flight.active_inspection_trial import wait_for_trial_budget
from src.flight.dynamic_blocker import coordinate_dynamic_blocker
from src.flight.mavsdk_connection import connect_mavsdk
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
    visual_runtime=None,
):
    drone = None
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
    blocker_task = None
    trial_task = None
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
        drone = await connect_mavsdk(
            services["system_factory"],
            system_address,
            settings.connection_timeout_s,
            services["wait_for_connection"],
            services["close_system"],
            attempts=services.get("connection_attempts", 2),
            retry_delay_s=services.get("connection_retry_delay_s", 2.0),
        )
        await services["wait_for_position_ready"](
            drone, settings.position_ready_timeout_s
        )
        if visual_runtime is not None:
            await visual_runtime.start(latest, phase_state, replan_config, replan_state)
            await visual_runtime.wait_ready(perception_config.get("sensor_startup_timeout_s", 5.0))
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
        semantic_trial = replan_config.get("semantic_trial")
        if semantic_trial is not None:
            trial_task = asyncio.create_task(
                wait_for_trial_budget(latest, semantic_trial["airborne_budget_s"]),
                name="active-inspection-trial-budget",
            )
            if event_writer is not None:
                event_writer.publish(
                    "trial_started",
                    trial_id=semantic_trial["trial_id"],
                    budget_s=semantic_trial["airborne_budget_s"],
                    trial_identity=semantic_trial["artifact_identity"],
                )
        dynamic_scenario = replan_config.get("dynamic_scenario")
        if dynamic_scenario is not None:
            coordinator = services.get(
                "dynamic_blocker_coordinator", coordinate_dynamic_blocker
            )
            blocker_task = asyncio.create_task(
                coordinator(latest, phase_state, dynamic_scenario),
                name="dynamic-blocker-coordinator",
            )
        while not mission_task.done():
            watched = {mission_task, telemetry_task}
            if blocker_task is not None and not blocker_task.done():
                watched.add(blocker_task)
            if trial_task is not None and not trial_task.done():
                watched.add(trial_task)
            done, _ = await asyncio.wait(
                watched, return_when=asyncio.FIRST_COMPLETED
            )
            if telemetry_task in done:
                logger_error = telemetry_task.exception()
                if logger_error is not None:
                    raise RuntimeError(
                        "Telemetry logger failed during flight"
                    ) from logger_error
                raise RuntimeError("Telemetry logger stopped unexpectedly during flight")
            if trial_task is not None and trial_task in done:
                replan_state["semantic_trial_stop_requested"] = True
                replan_state["semantic_mission_complete"] = True
                if event_writer is not None:
                    event_writer.publish(
                        "trial_budget_reached",
                        budget_s=semantic_trial["airborne_budget_s"],
                    )
                await cancel_tasks(
                    [mission_task], settings.logger_shutdown_timeout_s
                )
                mission_task = None
                landing_confirmed = await services["attempt_safe_landing"](
                    drone, latest, phase_state, "landing_after_trial"
                )
                if not landing_confirmed:
                    raise RuntimeError("active inspection trial landing was not confirmed")
                if event_writer is not None:
                    event_writer.publish(
                        "trial_completed",
                        reason="airborne_budget",
                        landing_confirmed=True,
                    )
                break
            if blocker_task is not None and blocker_task in done:
                blocker_error = blocker_task.exception()
                if blocker_error is not None:
                    raise RuntimeError(
                        "Dynamic blocker coordinator failed during flight"
                    ) from blocker_error
        if mission_task is not None:
            await mission_task
        if blocker_task is not None:
            await blocker_task
        landing_confirmed = phase_state["phase"] == "landed"
        if not landing_confirmed:
            raise RuntimeError("Mission ended without confirmed landing")
        services["write_run_status"](
            log_path, "completed", phase_state["phase"], landing_confirmed=True
        )
        if event_writer is not None:
            event_writer.publish(
                "landing_confirmed",
                phase=phase_state["phase"],
                landing_confirmed=True,
            )
            event_writer.publish(
                "mission_completed",
                status="completed",
                phase=phase_state["phase"],
                landing_confirmed=True,
            )
    except (Exception, asyncio.CancelledError) as error:
        print(f"Flight error: {error}")
        pending_error = error
        # Stop route commands before landing; keep telemetry alive for confirmation.
        active_controllers = [task for task in (mission_task, blocker_task, trial_task)
                              if task is not None and not task.done()]
        await cancel_tasks(active_controllers, settings.logger_shutdown_timeout_s)
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
            if landing_confirmed:
                event_writer.publish(
                    "landing_confirmed", phase=phase_state["phase"],
                    landing_confirmed=True, recovery=True,
                )
            event_writer.publish(
                "mission_failed",
                status="failed",
                phase=phase_state["phase"],
                landing_confirmed=landing_confirmed,
                failure_type=type(error).__name__,
                message=str(error),
            )
    finally:
        if mission_task is not None and not mission_task.done():
            await cancel_tasks([mission_task], settings.logger_shutdown_timeout_s)
        if blocker_task is not None and not blocker_task.done():
            await cancel_tasks([blocker_task], settings.logger_shutdown_timeout_s)
        if trial_task is not None and not trial_task.done():
            await cancel_tasks([trial_task], settings.logger_shutdown_timeout_s)
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
        if visual_runtime is not None:
            try:
                await visual_runtime.stop()
            except Exception as error:
                if pending_error is None:
                    pending_error = RuntimeError(f"Visual runtime did not stop cleanly: {error}")
        close_system = services.get("close_system")
        if close_system is not None and drone is not None:
            try:
                close_system(drone)
            except Exception as error:
                if pending_error is None:
                    pending_error = RuntimeError(
                        f"MAVSDK system did not stop cleanly: {error}"
                    )
    if pending_error is not None:
        raise pending_error
    print("Done.")
