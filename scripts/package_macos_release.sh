#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
version_manifest="$project_root/config/sandbox/version.json"
manifest_version=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["macos_app_version"])' "$version_manifest")
version=${1:-$manifest_version}
bundle_plist="$project_root/apps/macos/SandboxApp/Resources/Info.plist"

case "$version" in
  *[!0-9.]*|.*|*.)
    echo "version must contain dot-separated numbers" >&2
    exit 2
    ;;
esac

bundle_version=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$bundle_plist")
if [ "$manifest_version" != "$bundle_version" ]; then
  echo "version manifest $manifest_version does not match bundle version $bundle_version" >&2
  exit 2
fi
if [ "$version" != "$bundle_version" ]; then
  echo "release version $version does not match bundle version $bundle_version" >&2
  exit 2
fi

"$project_root/scripts/build_macos_app.sh" release

release_root="$project_root/dist/releases"
app="$project_root/dist/UAV Research Sandbox.app"
architecture=$(uname -m)
release_name="UAV-Research-Sandbox-v${version}-macos-${architecture}"
archive="$release_root/${release_name}.zip"
dmg="$release_root/${release_name}.dmg"
manifest="$release_root/${release_name}-release.json"
checksums="$release_root/${release_name}-SHA256SUMS"

mkdir -p "$release_root"
staging=$(mktemp -d "$release_root/.macos-preview.XXXXXX")
trap 'rm -rf "$staging"' EXIT HUP INT TERM

rm -f "$archive" "$dmg" "$manifest" "$checksums"
ditto -c -k --sequesterRsrc --keepParent "$app" "$archive"
ditto "$app" "$staging/UAV Research Sandbox.app"
ln -s /Applications "$staging/Applications"
cp "$project_root/docs/MACOS_PREVIEW_INSTALL.txt" "$staging/README.txt"
hdiutil create -quiet -volname "UAV Research Sandbox" \
  -srcfolder "$staging" -format UDZO "$dmg"

/usr/bin/python3 "$project_root/scripts/macos_release_manifest.py" create \
  --version "$version" \
  --architecture "$architecture" \
  --output "$manifest" \
  "$archive" "$dmg"

(
  cd "$release_root"
  shasum -a 256 \
    "$(basename "$archive")" \
    "$(basename "$dmg")" \
    "$(basename "$manifest")" > "$(basename "$checksums")"
)

/usr/bin/python3 "$project_root/scripts/macos_release_manifest.py" verify \
  --manifest "$manifest" \
  --checksums "$checksums"

echo "Created $archive"
echo "Created $dmg"
echo "Created $manifest"
echo "Created $checksums"
