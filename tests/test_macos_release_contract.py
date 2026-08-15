"""Release-boundary checks for the macOS preview artifact."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MacOSReleaseContractTests(unittest.TestCase):
    def test_packager_creates_archive_and_checksum(self):
        script = (ROOT / "scripts/package_macos_release.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("ditto -c -k", script)
        self.assertIn("shasum -a 256", script)
        self.assertIn("CFBundleShortVersionString", script)
        self.assertIn("config/sandbox/version.json", script)
        self.assertIn("dist/releases", script)
        self.assertNotIn("data/research", script)
        self.assertNotIn("models/", script)

    def test_release_is_manual_and_marked_prerelease(self):
        workflow = (
            ROOT / ".github/workflows/macos-preview-release.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertIn("--prerelease", workflow)
        self.assertIn("contents: write", workflow)
        self.assertNotIn("APPLE_", workflow)


if __name__ == "__main__":
    unittest.main()
