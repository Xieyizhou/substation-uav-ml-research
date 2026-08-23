import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.vision.run_hard_example_flight import _events, _wait_event


class HardExampleFlightTests(unittest.TestCase):
    def test_wait_event_accepts_event_and_event_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(json.dumps({"event_type":"takeoff_completed"}) + "\n")
            process = Mock(); process.poll.return_value = None
            _wait_event(path, "takeoff_completed", process, .1)

    def test_invalid_json_is_not_treated_as_an_event(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"; path.write_text("partial")
            self.assertEqual(_events(path), [])


if __name__ == "__main__":
    unittest.main()
