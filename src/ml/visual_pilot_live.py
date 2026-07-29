"""Live Gazebo collection orchestration for one visual pilot recording."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile

from src.ml.gazebo_visual_truth import GazeboTruthSource
from src.ml.visual_pilot import write_pilot_recording
from src.sensors.gazebo_camera import GazeboCameraSource
from src.sensors.gazebo_visual_transport import (
    inspect_visual_sources,
    load_research_visual_configuration,
)


def prepare_pilot_output_directory(output_directory):
    """Create an empty directory without mixing separate recording attempts."""
    output = Path(output_directory)
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"pilot output path is not a directory: {output}")
        if next(output.iterdir(), None) is not None:
            raise ValueError(
                f"pilot output directory is not empty: {output}; "
                "choose a new recording directory"
            )
    else:
        output.mkdir(parents=True)
    return output


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
    status_path = output / "live_status.json"
    if not status_path.is_file():
        raise RuntimeError("pilot recording has no live RGB status")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    timestamp = status.get("last_simulation_timestamp")
    if timestamp is None:
        raise RuntimeError("pilot recording has not received an RGB frame")
    event = {
        "simulation_timestamp": float(timestamp),
        "mission_phase": mission_phase,
    }
    events_path = output / "mission_events.jsonl"
    with events_path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(event, sort_keys=True) + "\n")
    return event


async def probe_live_visual_sources(*, timeout_s):
    inspected = await inspect_visual_sources(timeout_s=timeout_s)
    configured = load_research_visual_configuration()
    with tempfile.TemporaryDirectory() as temporary:
        camera = GazeboCameraSource(temporary, topic=inspected["rgb"]["topic"])
        truth_source = GazeboTruthSource(
            width=configured["rgb"]["width"],
            height=configured["rgb"]["height"],
            topic=inspected["truth"]["topic"],
        )
        await asyncio.gather(camera.start(), truth_source.start())
        camera_events = camera.events(timeout_s=timeout_s)
        truth_events = truth_source.events(timeout_s=timeout_s)
        camera_task = asyncio.create_task(anext(camera_events))
        truth_task = asyncio.create_task(anext(truth_events))
        try:
            camera_event, truth_event = await asyncio.wait_for(
                asyncio.gather(camera_task, truth_task),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as error:
            raise TimeoutError(
                f"Gazebo visual probe timed out after {timeout_s:g}s"
            ) from error
        finally:
            for task in (camera_task, truth_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(camera_task, truth_task, return_exceptions=True)
            await camera_events.aclose()
            await truth_events.aclose()
            await camera.stop()
            await truth_source.stop()
    if not camera_event.valid:
        raise RuntimeError(f"invalid Gazebo RGB probe: {camera_event.invalid_reason}")
    if not truth_event.truth.valid:
        raise RuntimeError(
            "invalid Gazebo truth probe: "
            + "; ".join(truth_event.truth.invalid_reasons)
        )
    return {
        "inspection": inspected,
        "configured": configured,
        "rgb_frame": camera_event.frame.to_record(),
        "truth_frame": truth_event.truth.to_record(),
    }


async def record_live_visual_pilot(
    output_directory,
    *,
    recording_id,
    source_timeout_s,
    duration_s=None,
    frame_limit=None,
    maximum_skew_ms=33.334,
    expected_rate_tolerance_fraction=None,
    jitter_p95_limit_ms=None,
    receive_stall_limit_ms=None,
    mission_events_path=None,
):
    if source_timeout_s <= 0:
        raise ValueError("source_timeout_s must be positive")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if frame_limit is not None and frame_limit <= 0:
        raise ValueError("frame_limit must be positive")
    if maximum_skew_ms < 0:
        raise ValueError("maximum_skew_ms must be non-negative")
    output = prepare_pilot_output_directory(output_directory)
    inspected = await inspect_visual_sources(timeout_s=source_timeout_s)
    configured = load_research_visual_configuration()
    if (
        configured["rgb"]["width"] != configured["truth"]["width"]
        or configured["rgb"]["height"] != configured["truth"]["height"]
    ):
        raise RuntimeError("configured RGB and truth image dimensions differ")
    camera = GazeboCameraSource(
        output_directory, topic=inspected["rgb"]["topic"]
    )
    truth_source = GazeboTruthSource(
        width=configured["rgb"]["width"],
        height=configured["rgb"]["height"],
        topic=inspected["truth"]["topic"],
    )
    frames = []
    truths = []
    invalid_frames = []
    stop = asyncio.Event()
    source_timeout_count = 0
    complete = False
    live_status_path = output / "live_status.json"

    async def collect_camera():
        async for event in camera.events(timeout_s=source_timeout_s):
            if event.valid:
                frames.append(event.frame)
                if len(frames) == 1 or len(frames) % 100 == 0:
                    print(
                        "Visual pilot progress: "
                        f"{len(frames)} PNG frames, "
                        f"simulation time {event.frame.capture_timestamp:.3f}s",
                        flush=True,
                    )
                live_status_path.write_text(
                    json.dumps(
                        {
                            "recording_id": recording_id,
                            "accepted_frame_count": len(frames),
                            "last_simulation_timestamp": (
                                event.frame.capture_timestamp
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if frame_limit is not None and len(frames) >= frame_limit:
                    stop.set()
                    return
            else:
                invalid_frames.append(
                    {
                        "receive_index": event.receive_index,
                        "reason": event.invalid_reason,
                    }
                )

    async def collect_truth():
        async for event in truth_source.events(timeout_s=source_timeout_s):
            truths.append(event.truth)

    async def duration_stop():
        await asyncio.sleep(duration_s)
        stop.set()

    await asyncio.gather(camera.start(), truth_source.start())
    camera_task = asyncio.create_task(collect_camera(), name="visual-pilot-rgb")
    truth_task = asyncio.create_task(collect_truth(), name="visual-pilot-truth")
    stop_task = asyncio.create_task(stop.wait(), name="visual-pilot-stop")
    tasks = {camera_task, truth_task, stop_task}
    timer_task = None
    if duration_s is not None:
        timer_task = asyncio.create_task(
            duration_stop(), name="visual-pilot-duration"
        )
        tasks.add(timer_task)
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        source_failure = next(
            (
                task
                for task in (camera_task, truth_task)
                if task in done and task.exception() is not None
            ),
            None,
        )
        if source_failure is not None:
            error = source_failure.exception()
            if isinstance(error, TimeoutError):
                source_timeout_count += 1
            raise error
        complete = stop.is_set() or stop_task in done or (
            timer_task is not None and timer_task in done
        )
        if complete:
            if not camera_task.done():
                camera_task.cancel()
                await asyncio.gather(camera_task, return_exceptions=True)
            await asyncio.sleep(maximum_skew_ms / 1000.0)
    except asyncio.CancelledError:
        complete = bool(frames)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await camera.stop()
        await truth_source.stop()
        summary = write_pilot_recording(
            output_directory,
            frames,
            truths,
            recording_id=recording_id,
            mission_events=load_mission_events(
                mission_events_path or output / "mission_events.jsonl"
            ),
            payload_root=output_directory,
            invalid_frames=invalid_frames,
            source_timeout_count=source_timeout_count,
            complete=complete,
            maximum_skew_ms=maximum_skew_ms,
            expected_rate_tolerance_fraction=expected_rate_tolerance_fraction,
            jitter_p95_limit_ms=jitter_p95_limit_ms,
            receive_stall_limit_ms=receive_stall_limit_ms,
        )
        live_status_path.unlink(missing_ok=True)
    return {
        "inspection": inspected,
        "configured": configured,
        "summary": summary,
    }
