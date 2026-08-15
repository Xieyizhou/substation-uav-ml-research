"""Release-boundary checks for macOS preview and Beta artifacts."""

from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


class MacOSReleaseContractTests(unittest.TestCase):
    def test_packager_creates_dmg_archive_manifest_and_checksums(self):
        script = (ROOT / "scripts/package_macos_release.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("ditto -c -k", script)
        self.assertIn("hdiutil create", script)
        self.assertIn("shasum -a 256", script)
        self.assertIn("macos_release_manifest.py", script)
        self.assertIn("MACOS_PREVIEW_INSTALL.txt", script)
        self.assertIn("CFBundleShortVersionString", script)
        self.assertIn("config/sandbox/version.json", script)
        self.assertIn("dist/releases", script)
        self.assertNotIn("data/research", script)
        self.assertNotIn("models/", script)

    def test_manifest_declares_preview_boundary(self):
        script = (ROOT / "scripts/macos_release_manifest.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"developer_preview"', script)
        self.assertIn('"standalone_demo_included"', script)
        self.assertIn('"advanced_profiles_require_project_repository"', script)
        self.assertIn('"advanced_profiles_require_python_environment"', script)
        self.assertIn('"external_simulator_toolchain"', script)
        self.assertIn('"ad_hoc"', script)

    def test_manifest_verifier_rejects_changed_artifact(self):
        tool = ROOT / "scripts/macos_release_manifest.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            release_root = Path(temporary_directory)
            prefix = "UAV-Research-Sandbox-v0.5.0-macos-arm64"
            archive = release_root / f"{prefix}.zip"
            disk_image = release_root / f"{prefix}.dmg"
            manifest = release_root / "release.json"
            checksums = release_root / "SHA256SUMS"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "UAV Research Sandbox.app/Contents/Info.plist",
                    "preview",
                )
            disk_image.write_bytes(b"disk-image")
            subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "create",
                    "--version",
                    "0.5.0",
                    "--architecture",
                    "arm64",
                    "--source-commit-sha",
                    "a" * 40,
                    "--tracked-worktree-clean",
                    "false",
                    "--output",
                    str(manifest),
                    str(archive),
                    str(disk_image),
                ],
                check=True,
            )
            manifest_hash = self._sha256(manifest)
            archive_hash = self._sha256(archive)
            disk_image_hash = self._sha256(disk_image)
            checksums.write_text(
                f"{archive_hash}  {archive.name}\n"
                f"{disk_image_hash}  {disk_image.name}\n"
                f"{manifest_hash}  {manifest.name}\n"
            )
            self._run_verify(tool, manifest, checksums, check=True)
            disk_image.write_bytes(b"changed")
            failed = self._run_verify(tool, manifest, checksums)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("artifact identity mismatch", failed.stderr)

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _run_verify(tool, manifest, checksums, check=False):
        return subprocess.run(
            [
                sys.executable,
                str(tool),
                "verify",
                "--manifest",
                str(manifest),
                "--checksums",
                str(checksums),
            ],
            check=check,
            capture_output=True,
            text=True,
        )

    def test_release_is_manual_and_marked_prerelease(self):
        workflow = (
            ROOT / ".github/workflows/macos-preview-release.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertIn("--prerelease", workflow)
        self.assertIn("contents: write", workflow)
        self.assertIn("*.dmg", workflow)
        self.assertIn("*-release.json", workflow)
        self.assertIn("*-SHA256SUMS", workflow)
        self.assertNotIn("APPLE_", workflow)

    def test_unsigned_beta_is_verifiable_and_requires_clean_sources(self):
        script = (ROOT / "scripts/package_macos_release.sh").read_text(
            encoding="utf-8"
        )
        wrapper = (ROOT / "scripts/package_macos_beta.sh").read_text(
            encoding="utf-8"
        )
        workflow = (
            ROOT / ".github/workflows/macos-beta-release.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("--unsigned-beta", wrapper)
        self.assertIn("unsigned_beta", script)
        self.assertIn("Beta packaging requires a clean tracked worktree", script)
        self.assertIn("MACOS_UNSIGNED_BETA_INSTALL.txt", script)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("package_macos_beta.sh", workflow)
        self.assertIn("codesign --verify", workflow)
        self.assertNotIn("MACOS_DEVELOPER_ID", workflow)
        self.assertNotIn("spctl --assess", workflow)
        self.assertIn("--prerelease", workflow)

    def test_notarized_packager_requires_developer_id_and_profile(self):
        script = (ROOT / "scripts/package_macos_release.sh").read_text(
            encoding="utf-8"
        )
        build = (ROOT / "scripts/build_macos_app.sh").read_text(encoding="utf-8")
        wrapper = (ROOT / "scripts/package_macos_notarized_beta.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--notarize", wrapper)
        self.assertIn("MACOS_CODESIGN_IDENTITY", script)
        self.assertIn("MACOS_NOTARY_PROFILE", script)
        self.assertIn("notarytool submit", script)
        self.assertIn("stapler staple", script)
        self.assertIn("--options runtime", build)

        environment = os.environ.copy()
        environment.pop("MACOS_CODESIGN_IDENTITY", None)
        environment.pop("MACOS_NOTARY_PROFILE", None)
        failed = subprocess.run(
            [str(ROOT / "scripts/package_macos_notarized_beta.sh"), "0.5.0"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("MACOS_CODESIGN_IDENTITY", failed.stderr)

    def test_notarized_manifest_requires_signed_and_stapled_identity(self):
        tool = ROOT / "scripts/macos_release_manifest.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            release_root = Path(temporary_directory)
            prefix = "UAV-Research-Sandbox-v0.5.0-macos-arm64"
            archive = release_root / f"{prefix}.zip"
            disk_image = release_root / f"{prefix}.dmg"
            manifest = release_root / f"{prefix}-release.json"
            checksums = release_root / f"{prefix}-SHA256SUMS"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "UAV Research Sandbox.app/Contents/Info.plist", "beta"
                )
            disk_image.write_bytes(b"disk-image")
            subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "create",
                    "--version",
                    "0.5.0",
                    "--architecture",
                    "arm64",
                    "--distribution-tier",
                    "notarized_beta",
                    "--signing",
                    "developer_id",
                    "--notarization",
                    "stapled",
                    "--source-commit-sha",
                    "b" * 40,
                    "--tracked-worktree-clean",
                    "true",
                    "--output",
                    str(manifest),
                    str(archive),
                    str(disk_image),
                ],
                check=True,
            )
            checksums.write_text(
                f"{self._sha256(archive)}  {archive.name}\n"
                f"{self._sha256(disk_image)}  {disk_image.name}\n"
                f"{self._sha256(manifest)}  {manifest.name}\n"
            )
            self._run_verify(tool, manifest, checksums, check=True)
            payload = json.loads(manifest.read_text())
            payload["signing"] = "ad_hoc"
            payload["notarization"] = "not_requested"
            manifest.write_text(json.dumps(payload))
            failed = self._run_verify(tool, manifest, checksums)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Developer ID signed and notarized", failed.stderr)

    def test_notarized_workflow_is_manual_and_cleans_credentials(self):
        workflow = (
            ROOT / ".github/workflows/macos-notarized-beta-release.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("pull_request:", workflow)
        for secret in (
            "MACOS_DEVELOPER_ID_P12_BASE64",
            "MACOS_DEVELOPER_ID_P12_PASSWORD",
            "MACOS_DEVELOPER_ID_APPLICATION",
            "APP_STORE_CONNECT_KEY_P8_BASE64",
            "APP_STORE_CONNECT_KEY_ID",
            "APP_STORE_CONNECT_ISSUER_ID",
        ):
            self.assertIn(secret, workflow)
        self.assertIn("spctl --assess", workflow)
        self.assertIn("stapler validate", workflow)
        self.assertIn("security delete-keychain", workflow)
        self.assertIn("--prerelease", workflow)


if __name__ == "__main__":
    unittest.main()
