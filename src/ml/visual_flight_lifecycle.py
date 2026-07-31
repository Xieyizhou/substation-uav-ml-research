"""Translate flight lifecycle JSONL into visual phases and recorder stop signals."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


def load_mission_events(path):
    if path is None or not Path(path).is_file():
        return ()
    events = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed mission event: {error}"
                ) from error
    return tuple(events)


def append_live_phase_event(output_directory, mission_phase):
    allowed = {
        "cruise_distant",
        "approach",
        "close_inspection",
        "target_transition",
        "other",
    }
    if mission_phase not in allowed:
        raise ValueError(f"unsupported mission phase: {mission_phase}")
    output = Path(output_directory)
    timestamp = _latest_simulation_timestamp(output / "live_status.json")
    if timestamp is None:
        raise RuntimeError("recording has not received an RGB frame")
    event = {
        "simulation_timestamp": timestamp,
        "mission_phase": mission_phase,
        "event_source": "manual",
    }
    with (output / "mission_events.jsonl").open(
        "a", encoding="utf-8"
    ) as destination:
        destination.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def write_live_status(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def visual_phase_for_flight_event(event):
    if event.get("event_type") == "phase_changed":
        phase = event.get("phase")
        if phase == "outbound_to_goal":
            return "cruise_distant"
        if phase == "goal_hover":
            return "close_inspection"
        if phase == "return_to_start":
            return "target_transition"
    if (
        event.get("event_type") == "waypoint_started"
        and event.get("route_direction") == "outbound"
        and event.get("is_final_waypoint") is True
    ):
        return "approach"
    return None


def _latest_simulation_timestamp(live_status_path):
    try:
        status = json.loads(Path(live_status_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    value = status.get("last_simulation_timestamp")
    return None if value is None else float(value)


def _append_visual_event(path, event, mission_phase, timestamp):
    record = {
        "simulation_timestamp": timestamp,
        "mission_phase": mission_phase,
        "event_source": "flight_lifecycle",
        "flight_event_sequence": event["event_sequence"],
        "flight_event_type": event["event_type"],
        "flight_phase": event.get("phase"),
    }
    with Path(path).open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(record, sort_keys=True) + "\n")
        destination.flush()
    return record


def _validate_event(event, previous_sequence):
    if event.get("flight_mission_event_schema_version") != 1:
        raise ValueError("unsupported flight mission event schema")
    sequence = event.get("event_sequence")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence != previous_sequence + 1
    ):
        raise ValueError("flight mission events are not contiguous")
    return sequence


async def monitor_flight_lifecycle(
    flight_events_path,
    live_status_path,
    visual_events_path,
    stop_event,
    outcome,
    *,
    post_landing_drain_s=1.0,
    poll_interval_s=0.05,
):
    if post_landing_drain_s < 0:
        raise ValueError("post_landing_drain_s must be non-negative")
    offset = 0
    previous_sequence = 0
    emitted_phases = set()
    pending = []
    while True:
        path = Path(flight_events_path)
        if path.is_file():
            with path.open(encoding="utf-8") as source:
                source.seek(offset)
                lines = source.readlines()
                offset = source.tell()
            for line in lines:
                if not line.strip():
                    continue
                event = json.loads(line)
                previous_sequence = _validate_event(event, previous_sequence)
                mission_phase = visual_phase_for_flight_event(event)
                if mission_phase and mission_phase not in emitted_phases:
                    pending.append((event, mission_phase))
                    emitted_phases.add(mission_phase)
                event_type = event.get("event_type")
                if event_type == "mission_completed":
                    if (
                        event.get("status") != "completed"
                        or event.get("phase") != "landed"
                        or event.get("landing_confirmed") is not True
                    ):
                        raise ValueError("invalid completed flight mission event")
                    outcome.update(event)
                elif event_type == "mission_failed":
                    outcome.update(event)
            timestamp = _latest_simulation_timestamp(live_status_path)
            if timestamp is not None:
                for event, mission_phase in pending:
                    _append_visual_event(
                        visual_events_path,
                        event,
                        mission_phase,
                        timestamp,
                    )
                pending.clear()
            if outcome.get("event_type") == "mission_failed":
                stop_event.set()
                return
            if outcome.get("event_type") == "mission_completed" and not pending:
                await asyncio.sleep(post_landing_drain_s)
                stop_event.set()
                return
        await asyncio.sleep(poll_interval_s)
