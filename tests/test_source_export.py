"""Clean-source bundles preserve current code and omit local research state."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

from scripts.export_source import export_source


class SourceExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.write(".gitignore", ".venv/\ndata/research/\noutputs/*\n.env\n")
        self.write("README.md", "tracked old content")
        subprocess.run(["git", "add", "README.md", ".gitignore"], cwd=self.root, check=True)
        self.write("README.md", "current content")
        self.write("src/new_module.py", "VALUE = 7\n")

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return path

    def test_current_and_untracked_code_without_local_data(self):
        for name in (".env", ".venv/secret", "data/research/model.json", "outputs/job.json"):
            self.write(name, "local-only")
        script = self.write("scripts/run.sh", "#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        self.write("archive/research_scripts/vision/old.py", "OLD = True\n")
        report = export_source(self.root, self.root / "source.zip")
        with zipfile.ZipFile(report["output"]) as bundle:
            self.assertEqual(bundle.read("README.md"), b"current content")
            self.assertIn("src/new_module.py", bundle.namelist())
            self.assertIn("archive/research_scripts/vision/old.py", bundle.namelist())
            self.assertEqual(bundle.getinfo("scripts/run.sh").external_attr >> 16 & 0o777, 0o755)
            self.assertFalse(any(name.startswith((".git/", ".venv/", "data/research/", "outputs/"))
                                 or name == ".env" for name in bundle.namelist()))
            manifest = json.loads(bundle.read("SOURCE_MANIFEST.json"))
            self.assertEqual(len(manifest), report["files"])
            for row in manifest:
                self.assertEqual(hashlib.sha256(bundle.read(row["path"])).hexdigest(), row["sha256"])

    def test_symlink_is_not_dereferenced_and_existing_bundle_is_preserved(self):
        (self.root / "src/private.py").symlink_to("../.git/config")
        output = self.root / "source.zip"
        export_source(self.root, output)
        before = output.read_bytes()
        with zipfile.ZipFile(output) as bundle:
            self.assertNotIn("src/private.py", bundle.namelist())
        with self.assertRaises(FileExistsError):
            export_source(self.root, output)
        self.assertEqual(output.read_bytes(), before)
