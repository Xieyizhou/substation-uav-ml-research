#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
notarize=false
unsigned_beta=false
case "${1:-}" in
  --unsigned-beta) unsigned_beta=true; shift ;;
  --notarize) notarize=true; shift ;;
esac
version_manifest="$project_root/config/sandbox/version.json"
manifest_version=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["macos_app_version"])' "$version_manifest")
version=${1:-$manifest_version}
bundle_plist="$project_root/apps/macos/SandboxApp/Resources/Info.plist"

distribution_tier=developer_preview
signing=ad_hoc
notarization=not_requested
if [ "$unsigned_beta" = true ]; then
  distribution_tier=unsigned_beta
fi
if [ "$notarize" = true ]; then
  : "${MACOS_CODESIGN_IDENTITY:?MACOS_CODESIGN_IDENTITY is required for a Beta package}"
  : "${MACOS_NOTARY_PROFILE:?MACOS_NOTARY_PROFILE is required for a Beta package}"
  if [ "$MACOS_CODESIGN_IDENTITY" = "-" ]; then
    echo "Beta packaging requires a Developer ID Application identity" >&2
    exit 2
  fi
  command -v xcrun >/dev/null 2>&1 || {
    echo "xcrun is required for notarization" >&2
    exit 2
  }
  distribution_tier=notarized_beta
  signing=developer_id
  notarization=stapled
fi

source_commit_sha=$(git -C "$project_root" rev-parse HEAD)
tracked_worktree_clean=true
if [ -n "$(git -C "$project_root" status --porcelain --untracked-files=no)" ]; then
  tracked_worktree_clean=false
fi
if { [ "$unsigned_beta" = true ] || [ "$notarize" = true ]; } && \
   [ "$tracked_worktree_clean" != true ]; then
  echo "Beta packaging requires a clean tracked worktree" >&2
  exit 2
fi

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
submission="$release_root/.${release_name}-notary-submission.zip"

mkdir -p "$release_root"
staging=$(mktemp -d "$release_root/.macos-preview.XXXXXX")
trap 'rm -rf "$staging"; rm -f "$submission"' EXIT HUP INT TERM

rm -f "$archive" "$dmg" "$manifest" "$checksums" "$submission"

if [ "$notarize" = true ]; then
  codesign --verify --deep --strict --verbose=2 "$app"
  ditto -c -k --sequesterRsrc --keepParent "$app" "$submission"
  xcrun notarytool submit "$submission" \
    --keychain-profile "$MACOS_NOTARY_PROFILE" --wait
  xcrun stapler staple "$app"
  xcrun stapler validate "$app"
  rm -f "$submission"
fi

ditto -c -k --sequesterRsrc --keepParent "$app" "$archive"
ditto "$app" "$staging/UAV Research Sandbox.app"
ln -s /Applications "$staging/Applications"
install_readme="$project_root/docs/MACOS_PREVIEW_INSTALL.txt"
if [ "$unsigned_beta" = true ]; then
  install_readme="$project_root/docs/MACOS_UNSIGNED_BETA_INSTALL.txt"
elif [ "$notarize" = true ]; then
  install_readme="$project_root/docs/MACOS_NOTARIZED_BETA_INSTALL.txt"
fi
cp "$install_readme" "$staging/README.txt"
hdiutil create -quiet -volname "UAV Research Sandbox" \
  -srcfolder "$staging" -format UDZO "$dmg"

if [ "$notarize" = true ]; then
  codesign --force --sign "$MACOS_CODESIGN_IDENTITY" --timestamp "$dmg"
  xcrun notarytool submit "$dmg" \
    --keychain-profile "$MACOS_NOTARY_PROFILE" --wait
  xcrun stapler staple "$dmg"
  xcrun stapler validate "$dmg"
fi

/usr/bin/python3 "$project_root/scripts/macos_release_manifest.py" create \
  --version "$version" \
  --architecture "$architecture" \
  --distribution-tier "$distribution_tier" \
  --signing "$signing" \
  --notarization "$notarization" \
  --source-commit-sha "$source_commit_sha" \
  --tracked-worktree-clean "$tracked_worktree_clean" \
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
