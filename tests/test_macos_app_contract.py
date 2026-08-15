"""Static contract checks for the native macOS Sandbox shell."""

from pathlib import Path
import plistlib
import struct
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
        self.assertEqual(value["CFBundleIconFile"], "AppIcon")
        self.assertEqual(value["CFBundleShortVersionString"], "0.2.1")

    def test_bundle_version_matches_shared_manifest(self):
        import json

        version = json.loads(
            (ROOT / "config/sandbox/version.json").read_text(encoding="utf-8")
        )
        with (APP_ROOT / "Resources/Info.plist").open("rb") as handle:
            plist = plistlib.load(handle)
        self.assertEqual(plist["CFBundleShortVersionString"], version["macos_app_version"])

    def test_icon_master_is_square_1024_png(self):
        payload = (APP_ROOT / "Resources/AppIcon.png").read_bytes()
        self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", payload[16:24])
        self.assertEqual((width, height), (1024, 1024))

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
        self.assertIn("scripts/create_icns.py", script)

    def test_status_home_reads_only_loopback_endpoints(self):
        model = (
            APP_ROOT / "Sources/UAVSandboxApp/SandboxStatusModel.swift"
        ).read_text(encoding="utf-8")
        for endpoint in (
            "profile", "version", "setup", "runtime", "doctor", "storage", "operator"
        ):
            self.assertIn(f'"api/{endpoint}"', model)
        self.assertNotIn("operator/start", model)
        self.assertNotIn("operator/stop", model)


if __name__ == "__main__":
    unittest.main()
