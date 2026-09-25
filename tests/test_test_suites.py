import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_tests import modules_for


class TestSuiteSelectionTests(unittest.TestCase):
    def test_new_tests_are_core_and_manifest_cannot_hide_missing_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests/test_new.py").touch()
            (root / "tests/test_historical.py").touch()
            path = root / "config/sandbox/test_suites.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"research_modules": ["test_historical"]}))
            self.assertEqual(modules_for("core", root), ["test_new"])
            self.assertEqual(modules_for("research", root), ["test_historical"])
            self.assertEqual(len(modules_for("all", root)), 2)
            path.write_text(json.dumps({"research_modules": ["test_missing"]}))
            with self.assertRaisesRegex(ValueError, "missing"):
                modules_for("core", root)
