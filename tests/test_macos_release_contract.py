"""Release-boundary checks for the macOS preview artifact."""

from pathlib import Path
import hashlib
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
            prefix = "UAV-Research-Sandbox-v0.4.0-macos-arm64"
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
                    "0.4.0",
                    "--architecture",
                    "arm64",
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


if __name__ == "__main__":
    unittest.main()
