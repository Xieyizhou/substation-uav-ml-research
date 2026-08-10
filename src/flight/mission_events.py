"""Machine-readable flight lifecycle events for external recorders."""

from __future__ import annotations

import json
import time
from pathlib import Path


class MissionEventWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        self.sequence = 0

    def publish(self, event_type, **details):
        self.sequence += 1
        event = {
            "flight_mission_event_schema_version": 1,
            "event_sequence": self.sequence,
            "event_type": event_type,
            "host_monotonic_ns": time.monotonic_ns(),
            **details,
        }
        with self.path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(event, sort_keys=True) + "\n")
            destination.flush()
        return event
