#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
version=${1:-0.2.0}
bundle_plist="$project_root/apps/macos/SandboxApp/Resources/Info.plist"

case "$version" in
  *[!0-9.]*|.*|*.)
    echo "version must contain dot-separated numbers" >&2
    exit 2
    ;;
esac

bundle_version=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$bundle_plist")
if [ "$version" != "$bundle_version" ]; then
  echo "release version $version does not match bundle version $bundle_version" >&2
  exit 2
fi

"$project_root/scripts/build_macos_app.sh" release

release_root="$project_root/dist/releases"
app="$project_root/dist/UAV Research Sandbox.app"
archive="$release_root/UAV-Research-Sandbox-v${version}-macos-arm64.zip"
checksum="$archive.sha256"

mkdir -p "$release_root"
rm -f "$archive" "$checksum"
ditto -c -k --sequesterRsrc --keepParent "$app" "$archive"
(
  cd "$release_root"
  shasum -a 256 "$(basename "$archive")" > "$(basename "$checksum")"
)

echo "Created $archive"
echo "Created $checksum"
