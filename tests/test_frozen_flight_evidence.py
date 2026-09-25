import json
from pathlib import Path
import tempfile
import unittest

from src.ml.artifacts import file_sha256
from src.sandbox.evidence_archive import freeze_evidence
from src.sandbox.frozen_flight_evidence import BASE, POSITIVE, STOPPED, STOP_PROOF, verify_flight_archive


class FrozenFlightEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "controller.py"
        self.source.write_text("historical controller")
        self.rows = {}
        units = []
        landed = dict(landing_confirmed=True, final_armed=False, owned_processes_exited=True)
        for case, n in [("no_path", 1)] + [(c, n) for c in ("baseline", "west", "later") for n in (1, 2)]:
            prefix = "lowload-repeat-v1" if case in ("no_path", "baseline") or (case, n) == ("west", 1) else "startup-repeat-v2"
            directory = f"{prefix}-{case}-{n}"
            units.append(dict(case=case, repeat=n, directory=directory))
            self.put(directory + "/completion.json", status="expected_safe_no_path_rejection_verified" if case == "no_path" else "sensor_stop_replan_resume_goal_verified")
            self.put(directory + "/protocol.json")
            self.put(directory + "/runtime/receipt.json", **landed)
            self.put(directory + "/evidence-replay.json", status="raw_scan_pose_maps_replayed")
            if case != "no_path":
                self.put(directory + "/map-registration.json")
                self.put(directory + "/runtime/vision/completion.json", status="live_complete_not_flight_certified")
        self.put("startup-repeat-v2/protocol.json", units=units)
        self.put("startup-repeat-v2/completion.json", status="repeat_campaign_verified", sandbox_integration_allowed=True,
                 not_run=[], units=[dict(u, passed=True) for u in units])
        for name in ("lowload-vehicle-v1", "envelope-aware-scene-v1", POSITIVE, STOPPED):
            self.put(name + "/protocol.json")
        self.put(POSITIVE + "/completion.json", status="visual_stop_replan_resume_standoff_verified")
        self.put(POSITIVE + "/runtime/receipt.json", **landed)
        self.put(STOPPED + "/runtime/receipt.json", **landed, error="Sandbox requested controlled stop and landing")
        self.rows[STOP_PROOF] = dict(state="controlled_visual_stop_verified", run=str(self.root / BASE / STOPPED))

    def put(self, name, **values):
        self.rows[BASE + name] = values

    def verify(self):
        paths = {}
        for name, values in self.rows.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(values, inputs={str(self.source): file_sha256(self.source)})))
            paths[name] = path
        output = self.root / "archive.zip"
        result = freeze_evidence(paths, output, project_root=self.root)
        # Changing the live implementation cannot change historical facts.
        self.source.write_text("new controller, not yet flight-qualified")
        return verify_flight_archive(output, result["manifest_sha256"])

    def test_all_historical_cases_verified_without_certifying_current_runtime(self):
        report = self.verify()
        self.assertTrue(report["historical_verified"])
        self.assertEqual(report["records"], 50)
        self.assertFalse(report["current_runtime_certified"])

    def test_duplicate_repeat_matrix_rejected(self):
        rows = self.rows[BASE + "startup-repeat-v2/protocol.json"]["units"]
        rows[-1] = rows[0]
        with self.assertRaisesRegex(ValueError, "matrix"):
            self.verify()

    def test_incomplete_landing_and_wrong_stop_provenance_rejected(self):
        self.rows[BASE + STOPPED + "/runtime/receipt.json"]["final_armed"] = True
        with self.assertRaisesRegex(ValueError, "landing"):
            self.verify()

    def test_wrong_stop_run_rejected(self):
        self.rows[STOP_PROOF]["run"] = "/some/other/run"
        with self.assertRaisesRegex(ValueError, "visual or stop"):
            self.verify()

    def test_missing_replay_rejected(self):
        self.rows[BASE + "lowload-repeat-v1-baseline-1/evidence-replay.json"]["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "raw sensor"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
