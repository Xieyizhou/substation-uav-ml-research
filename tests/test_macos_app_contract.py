"""Static contract checks for the native macOS Sandbox shell."""

from pathlib import Path
import json
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
        self.assertEqual(value["CFBundleShortVersionString"], "0.6.0")

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

    def test_native_workbench_import_and_quit_are_controlled(self):
        model = (APP_ROOT / "Sources/UAVSandboxApp/SandboxAppModel.swift").read_text()
        web_view = (APP_ROOT / "Sources/UAVSandboxApp/SandboxWebView.swift").read_text()
        delegate = (APP_ROOT / "Sources/UAVSandboxApp/AppDelegate.swift").read_text()
        build = (ROOT / "scripts/build_macos_app.sh").read_text()
        self.assertIn('"class_map": sourceToCanonical', model)
        self.assertIn("statusCode == 202", model)
        self.assertIn('host == "import-yolo"', web_view)
        self.assertIn("four distinct source class IDs", model)
        self.assertIn("Keep Task Running", delegate)
        self.assertIn("Stop Task and Quit", delegate)
        self.assertIn("AppDelegate.swift", build)

    def test_native_service_and_jobs_share_homebrew_aware_path(self):
        project = (
            APP_ROOT / "Sources/SandboxAppCore/ProjectConfiguration.swift"
        ).read_text(encoding="utf-8")
        model = (
            APP_ROOT / "Sources/UAVSandboxApp/SandboxAppModel.swift"
        ).read_text(encoding="utf-8")
        self.assertIn('"/opt/homebrew/bin"', project)
        self.assertIn('"/usr/local/bin"', project)
        self.assertEqual(model.count("project.runtimeEnvironment()"), 2)
        self.assertIn("child.environment = project.runtimeEnvironment()", model)

    def test_demo_runtime_does_not_require_repository_or_python(self):
        project = (
            APP_ROOT / "Sources/SandboxAppCore/ProjectConfiguration.swift"
        ).read_text(encoding="utf-8")
        model = (
            APP_ROOT / "Sources/UAVSandboxApp/SandboxAppModel.swift"
        ).read_text(encoding="utf-8")
        build = (ROOT / "scripts/build_macos_app.sh").read_text(encoding="utf-8")
        self.assertIn("self != .demo", project)
        self.assertIn("profile == .demo", model)
        self.assertIn("StandaloneDemo.swift", build)
        self.assertIn("StandaloneDemoView.swift", build)

    def test_advanced_profiles_use_persisted_verified_runtime(self):
        project = (
            APP_ROOT / "Sources/SandboxAppCore/ProjectConfiguration.swift"
        ).read_text(encoding="utf-8")
        model = (
            APP_ROOT / "Sources/UAVSandboxApp/SandboxAppModel.swift"
        ).read_text(encoding="utf-8")
        launcher = (ROOT / "scripts/flight/start_px4_substation.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("VerifiedRuntimeProfile", project)
        for variable in (
            "UAV_SANDBOX_PYTHON",
            "UAV_SANDBOX_GZ_EXECUTABLE",
            "PX4_ROOT",
            "UAV_SANDBOX_OPENCV_PREFIX",
            "UAV_SANDBOX_QT_PREFIX",
        ):
            self.assertIn(variable, project)
        self.assertIn("runtimeForLaunch", model)
        self.assertNotIn("killall", launcher)

    def test_runtime_compatibility_manifest_is_explicit(self):
        manifest = json.loads(
            (ROOT / "config/sandbox/runtime_compatibility.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["python"]["minimum_version"], "3.11")
        self.assertEqual(manifest["gazebo"]["tested_sim_major_versions"], [8])
        self.assertEqual(manifest["opencv"]["preferred_formula"], "opencv@4")
        self.assertEqual(manifest["qt"]["preferred_formula"], "qt@5")

    def test_packaging_script_targets_generated_dist_only(self):
        script = (ROOT / "scripts/build_macos_app.sh").read_text(encoding="utf-8")
        self.assertIn('dist/UAV Research Sandbox.app', script)
        self.assertNotIn("data/research", script)
        self.assertNotIn("models/", script)
        self.assertIn("scripts/create_icns.py", script)

    def test_ci_packaging_version_comes_from_shared_manifest(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        packaging = workflow.split("- name: Validate preview packaging", 1)[1]
        self.assertIn("config/sandbox/version.json", packaging)
        self.assertIn('["macos_app_version"]', packaging)
        self.assertIn('package_macos_release.sh "$version"', packaging)
        self.assertNotRegex(packaging, r"package_macos_release\.sh [0-9]+\.[0-9]+\.[0-9]+")

    def test_advanced_runtime_candidate_picker_covers_every_component(self):
        discovery = (
            APP_ROOT / "Sources/SandboxAppCore/RuntimeCandidateDiscovery.swift"
        ).read_text(encoding="utf-8")
        view = (
            APP_ROOT / "Sources/UAVSandboxApp/RuntimeCandidateView.swift"
        ).read_text(encoding="utf-8")
        build = (ROOT / "scripts/build_macos_app.sh").read_text(encoding="utf-8")
        for component in ("python", "px4", "gazebo", "opencv", "qt"):
            self.assertIn(f'"{component}"', discovery)
        for label in (
            "Python executable…",
            "PX4 checkout…",
            "Gazebo executable…",
            "OpenCV 4 prefix…",
            "Qt 5 prefix…",
        ):
            self.assertIn(label, view)
        self.assertIn("RuntimeCandidateDiscovery.swift", build)
        self.assertIn("RuntimeCandidateView.swift", build)

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
