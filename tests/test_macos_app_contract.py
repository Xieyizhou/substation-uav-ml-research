"""Static contract checks for the native macOS Sandbox shell."""

from pathlib import Path
import plistlib
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "apps/macos/SandboxApp"


class MacOSAppContractTests(unittest.TestCase):
    def test_bundle_metadata_matches_executable(self):
        with (APP_ROOT / "Resources/Info.plist").open("rb") as handle:
            value = plistlib.load(handle)
        self.assertEqual(value["CFBundleExecutable"], "UAVSandboxApp")
        self.assertEqual(
            value["CFBundleIdentifier"],
            "io.github.xieyizhou.uav-research-sandbox",
        )
        self.assertEqual(value["CFBundlePackageType"], "APPL")

    def test_native_shell_keeps_loopback_boundary(self):
        model = (
            APP_ROOT / "Sources/UAVSandboxApp/SandboxAppModel.swift"
        ).read_text(encoding="utf-8")
        web_view = (
            APP_ROOT / "Sources/UAVSandboxApp/SandboxWebView.swift"
        ).read_text(encoding="utf-8")
        self.assertIn('"--host", "127.0.0.1"', (
            APP_ROOT / "Sources/SandboxAppCore/ProjectConfiguration.swift"
        ).read_text(encoding="utf-8"))
        self.assertIn('http://127.0.0.1:', model)
        self.assertIn('"127.0.0.1", "localhost", "::1"', web_view)

    def test_packaging_script_targets_generated_dist_only(self):
        script = (ROOT / "scripts/build_macos_app.sh").read_text(encoding="utf-8")
        self.assertIn('dist/UAV Research Sandbox.app', script)
        self.assertNotIn("data/research", script)
        self.assertNotIn("models/", script)


if __name__ == "__main__":
    unittest.main()
