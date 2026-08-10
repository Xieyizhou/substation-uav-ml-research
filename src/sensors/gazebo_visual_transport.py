"""Gazebo Transport discovery and timestamp helpers for visual sources."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import re
from xml.etree import ElementTree


RGB_CONFIGURED_TOPIC = "research_camera/image"
TRUTH_CONFIGURED_TOPIC = "research_camera/boxes"
RGB_MESSAGE_TYPE = "gz.msgs.Image"
TRUTH_MESSAGE_TYPE = "gz.msgs.AnnotatedAxisAligned2DBox_V"
GAZEBO_SIM_CLOCK = "gazebo_sim_time"
# A 1920x1080 BGRA frame expands to about 11 MiB when Gazebo emits its
# payload as base64 JSON on one line. asyncio's 64 KiB default cannot carry it.
GAZEBO_JSON_STREAM_LIMIT_BYTES = 16 * 1024 * 1024
RESEARCH_MODEL = (
    Path(__file__).resolve().parents[2]
    / "simulation/models/x500_research/model.sdf"
)


def transport_environment():
    environment = os.environ.copy()
    environment.setdefault("GZ_IP", "127.0.0.1")
    return environment


async def stop_stream_process(process, *, timeout_s=2.0):
    """Terminate a streaming subprocess while draining its pipe buffers."""
    if process is None:
        return
    if process.returncode is None:
        process.terminate()
    communicate = getattr(process, "communicate", None)
    wait_for_exit = communicate if communicate is not None else process.wait
    try:
        await asyncio.wait_for(wait_for_exit(), timeout=timeout_s)
    except asyncio.TimeoutError:
        if process.returncode is None:
            process.kill()
        await asyncio.wait_for(wait_for_exit(), timeout=timeout_s)


async def _run_gz(*arguments, timeout_s):
    process = await asyncio.create_subprocess_exec(
        "gz",
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=transport_environment(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise TimeoutError(f"Gazebo command timed out after {timeout_s:g}s")
    if process.returncode:
        detail = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Gazebo command failed: {detail or process.returncode}")
    return stdout.decode(errors="replace")


def _topic_candidates(topics, configured_topic):
    suffix = "/" + configured_topic.strip("/")
    return sorted(
        topic
        for topic in topics
        if topic.strip("/") == configured_topic.strip("/") or topic.endswith(suffix)
    )


async def discover_configured_topic(configured_topic, *, timeout_s=5.0):
    output = await _run_gz("topic", "-l", timeout_s=timeout_s)
    topics = [line.strip() for line in output.splitlines() if line.strip()]
    candidates = _topic_candidates(topics, configured_topic)
    if not candidates:
        raise RuntimeError(
            f"configured Gazebo topic {configured_topic!r} was not discovered"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"configured Gazebo topic {configured_topic!r} is ambiguous: "
            + ", ".join(candidates)
        )
    return candidates[0]


async def inspect_topic(topic, expected_type, *, timeout_s=5.0):
    output = await _run_gz("topic", "-i", "-t", topic, timeout_s=timeout_s)
    message_types = sorted(set(re.findall(r"gz\.msgs\.[A-Za-z0-9_]+", output)))
    if expected_type not in message_types:
        observed = ", ".join(message_types) or "unknown"
        raise RuntimeError(
            f"Gazebo topic {topic!r} has message type {observed}; "
            f"expected {expected_type}"
        )
    return {
        "topic": topic,
        "message_type": expected_type,
        "inspection_output": output.strip(),
    }


async def inspect_visual_sources(*, timeout_s=5.0):
    rgb_topic, truth_topic = await asyncio.gather(
        discover_configured_topic(RGB_CONFIGURED_TOPIC, timeout_s=timeout_s),
        discover_configured_topic(TRUTH_CONFIGURED_TOPIC, timeout_s=timeout_s),
    )
    rgb, truth = await asyncio.gather(
        inspect_topic(rgb_topic, RGB_MESSAGE_TYPE, timeout_s=timeout_s),
        inspect_topic(truth_topic, TRUTH_MESSAGE_TYPE, timeout_s=timeout_s),
    )
    return {
        "transport": "gz topic JSON subprocess",
        "clock_domain": GAZEBO_SIM_CLOCK,
        "rgb": rgb,
        "truth": truth,
    }


def message_timestamp(message):
    header = message.get("header")
    if not isinstance(header, dict):
        raise ValueError("Gazebo message is missing header")
    stamp = header.get("stamp")
    if not isinstance(stamp, dict):
        raise ValueError("Gazebo message is missing header.stamp")
    seconds = stamp.get("sec", stamp.get("seconds"))
    nanoseconds = stamp.get("nsec", stamp.get("nanoseconds", 0))
    if seconds is None:
        raise ValueError("Gazebo message timestamp is missing seconds")
    timestamp = float(seconds) + float(nanoseconds) / 1_000_000_000.0
    if timestamp < 0:
        raise ValueError("Gazebo simulation timestamp must be non-negative")
    return timestamp


def header_sequence(message):
    header = message.get("header") or {}
    for item in header.get("data") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("key", "")).lower() not in {"seq", "sequence"}:
            continue
        values = item.get("value") or item.get("values") or []
        if values:
            try:
                value = int(values[0])
            except (TypeError, ValueError):
                return None
            return value if value >= 0 else None
    return None


def load_research_visual_configuration(path=RESEARCH_MODEL):
    root = ElementTree.parse(path).getroot()
    sensors = {
        sensor.get("name"): sensor
        for sensor in root.findall(".//sensor")
        if sensor.get("name") in {"research_rgb", "research_boxes"}
    }
    if set(sensors) != {"research_rgb", "research_boxes"}:
        raise ValueError("x500_research must configure RGB and bounding-box sensors")

    def values(sensor):
        topic = sensor.findtext("topic")
        width = sensor.findtext("camera/image/width")
        height = sensor.findtext("camera/image/height")
        rate = sensor.findtext("update_rate")
        if not topic or not width or not height or not rate:
            raise ValueError("research visual sensor configuration is incomplete")
        result = {
            "sensor_name": sensor.get("name"),
            "configured_topic": topic,
            "width": int(width),
            "height": int(height),
            "configured_update_rate_hz": float(rate),
        }
        if sensor.get("name") == "research_boxes":
            box_type = sensor.findtext("camera/box_type")
            if box_type != "full_2d":
                raise ValueError(
                    "research bounding-box sensor must use Gazebo full_2d mode"
                )
            result["box_type"] = box_type
        return result

    return {
        "rgb": values(sensors["research_rgb"]),
        "truth": values(sensors["research_boxes"]),
    }
