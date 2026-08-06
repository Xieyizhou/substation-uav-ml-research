import subprocess
import sys
from pathlib import Path
import tempfile
import time
import unittest

from src.vision.collection.process import (
    _process_group_exists,
    start_process,
    stop_process,
)


class ProcessGroupCleanupTests(unittest.TestCase):
    def test_stop_process_removes_descendants_after_group_leader_exits(self):
        with tempfile.TemporaryDirectory() as temporary:
            managed = start_process(
                "process tree",
                [
                    sys.executable,
                    "-c",
                    (
                        "import subprocess,sys,time; "
                        "subprocess.Popen([sys.executable,'-c',"
                        "'import signal,time; signal.signal(signal.SIGINT, "
                        "signal.SIG_IGN); time.sleep(60)']); "
                        "time.sleep(0.2)"
                    ),
                ],
                Path(temporary) / "tree.log",
            )
            process_group_id = managed.process.pid
            time.sleep(0.4)
            self.assertIsNotNone(managed.process.poll())
            self.assertTrue(_process_group_exists(process_group_id))
            stop_process(managed, grace_s=0.2, kill_grace_s=1.0)
            self.assertFalse(_process_group_exists(process_group_id))

    def test_stop_process_accepts_already_finished_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            managed = start_process(
                "short process",
                [sys.executable, "-c", "pass"],
                Path(temporary) / "short.log",
            )
            managed.process.wait(timeout=2.0)
            stop_process(managed, grace_s=0.1, kill_grace_s=0.1)
            self.assertTrue(managed.log_handle.closed)


if __name__ == "__main__":
    unittest.main()
