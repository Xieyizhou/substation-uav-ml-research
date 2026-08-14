"""Progress-aware outer timeout for sequential study flights."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import time

from src.study.flight_budget import (
    MAX_PROGRESS_EXTENSION_S,
    PROGRESS_STALL_WINDOW_S,
)
from src.vision.collection.process import CollectionProcessError, wait_process


LOG_STALL_WINDOW_S = 15.0
PROGRESS_STEP_M = 0.25


@dataclass
class FlightProgress:
    target_key: tuple[str, str, str]
    horizontal_error_m: float


class FlightProgressWatchdog:
    def __init__(self, now_s, soft_timeout_s):
        self.soft_deadline_s = now_s + soft_timeout_s
        self.absolute_deadline_s = self.soft_deadline_s + min(
            soft_timeout_s, MAX_PROGRESS_EXTENSION_S
        )
        self.last_activity_s = now_s
        self.last_progress_s = now_s
        self.target_key = None
        self.best_error_m = None

    def observe_activity(self, now_s):
        self.last_activity_s = now_s

    def observe_progress(self, progress, now_s):
        if progress.target_key != self.target_key:
            self.target_key = progress.target_key
            self.best_error_m = progress.horizontal_error_m
            self.last_progress_s = now_s
        elif progress.horizontal_error_m <= self.best_error_m - PROGRESS_STEP_M:
            self.best_error_m = progress.horizontal_error_m
            self.last_progress_s = now_s

    def should_continue(self, now_s):
        if now_s >= self.absolute_deadline_s:
            return False
        if now_s < self.soft_deadline_s:
            return True
        return (
            now_s - self.last_activity_s <= LOG_STALL_WINDOW_S
            and now_s - self.last_progress_s <= PROGRESS_STALL_WINDOW_S
        )

    def stop_reason(self, now_s):
        if now_s >= self.absolute_deadline_s:
            return "absolute flight limit reached"
        if now_s - self.last_activity_s > LOG_STALL_WINDOW_S:
            return "flight telemetry log stopped updating"
        return "flight made no 0.25 m waypoint progress for 45s"


class FlightProgressReader:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = self.path.open("r", encoding="utf-8", newline="")
        self.fields = next(csv.reader([self.handle.readline()]))

    def close(self):
        self.handle.close()

    def read_latest(self):
        latest = None
        while True:
            offset = self.handle.tell()
            line = self.handle.readline()
            if not line:
                break
            if not line.endswith("\n"):
                self.handle.seek(offset)
                break
            values = next(csv.reader([line]))
            if len(values) == len(self.fields):
                latest = dict(zip(self.fields, values))
        if latest is None or not latest.get("target_name"):
            return None
        try:
            error_m = float(latest["horizontal_error_m"])
        except (KeyError, TypeError, ValueError):
            return None
        return FlightProgress(
            (
                latest.get("phase", ""), latest.get("route_direction", ""),
                latest["target_name"],
            ),
            error_m,
        )


def _new_log(before_logs):
    log_dir = (
        next(iter(before_logs)).parent
        if before_logs else Path.cwd() / "data/logs"
    )
    candidates = set(log_dir.glob("astar_*.csv")) - before_logs
    return next(iter(candidates)) if len(candidates) == 1 else None


def wait_for_study_flight(
    managed, timeout_s, *, before_logs, allow_progress_extension
):
    if not allow_progress_extension:
        return wait_process(managed, timeout_s)
    watchdog = FlightProgressWatchdog(time.monotonic(), timeout_s)
    reader = None
    size = None
    try:
        while watchdog.should_continue(time.monotonic()):
            code = managed.process.poll()
            if code is not None:
                if code:
                    raise CollectionProcessError(
                        f"{managed.name} exited with code {code}; "
                        f"inspect {managed.log_path}"
                    )
                return None
            if reader is None:
                path = _new_log(before_logs)
                reader = FlightProgressReader(path) if path else None
            if reader is not None:
                current_size = reader.path.stat().st_size
                if current_size != size:
                    size = current_size
                    watchdog.observe_activity(time.monotonic())
                progress = reader.read_latest()
                if progress is not None:
                    watchdog.observe_progress(progress, time.monotonic())
            time.sleep(0.5)
        raise CollectionProcessError(
            f"{managed.name} stopped by progress watchdog: "
            f"{watchdog.stop_reason(time.monotonic())}; inspect {managed.log_path}"
        )
    finally:
        if reader is not None:
            reader.close()
        managed.close_log()
