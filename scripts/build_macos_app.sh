#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
package_root="$project_root/apps/macos/SandboxApp"
configuration=${1:-release}

case "$configuration" in
  debug|release) ;;
  *)
    echo "usage: $0 [debug|release]" >&2
    exit 2
    ;;
esac

build_root="$package_root/.build/$configuration"
binary="$build_root/UAVSandboxApp"

build_icon() {
  source_icon="$package_root/Resources/AppIcon.png"
  iconset="$build_root/AppIcon.iconset"
  icon_file="$build_root/AppIcon.icns"
  if [ ! -f "$source_icon" ]; then
    echo "App icon source is missing: $source_icon" >&2
    exit 1
  fi
  rm -rf "$iconset"
  mkdir -p "$iconset"
  for size in 16 32 128 256 512; do
    double=$((size * 2))
    sips -z "$size" "$size" "$source_icon" \
      --out "$iconset/icon_${size}x${size}.png" >/dev/null
    sips -z "$double" "$double" "$source_icon" \
      --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
  done
  python3 "$project_root/scripts/create_icns.py" "$iconset" "$icon_file"
}

build_with_command_line_tools() {
  sdk=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk
  if [ ! -d "$sdk" ]; then
    sdk=$(xcrun --sdk macosx --show-sdk-path)
  fi
  architecture=$(uname -m)
  target="$architecture-apple-macosx13.0"
  optimization=-O
  if [ "$configuration" = debug ]; then
    optimization=-Onone
  fi
  mkdir -p "$build_root/module-cache"
  swiftc -emit-library -static -parse-as-library -swift-version 5 \
    "$optimization" -sdk "$sdk" -target "$target" \
    -module-cache-path "$build_root/module-cache" \
    -module-name SandboxAppCore \
    -emit-module-path "$build_root/SandboxAppCore.swiftmodule" \
    -o "$build_root/libSandboxAppCore.a" \
    "$package_root/Sources/SandboxAppCore/ProjectConfiguration.swift" \
    "$package_root/Sources/SandboxAppCore/ProcessExecution.swift" \
    "$package_root/Sources/SandboxAppCore/LocalWebDocument.swift" \
    "$package_root/Sources/SandboxAppCore/RuntimeModels.swift" \
    "$package_root/Sources/SandboxAppCore/RuntimeCandidateModels.swift" \
    "$package_root/Sources/SandboxAppCore/RuntimeProfileStore.swift" \
    "$package_root/Sources/SandboxAppCore/RuntimeInspection.swift" \
    "$package_root/Sources/SandboxAppCore/RuntimeCompatibilityManager.swift" \
    "$package_root/Sources/SandboxAppCore/RuntimeCandidateDiscovery.swift" \
    "$package_root/Sources/SandboxAppCore/RuntimeConflict.swift" \
    "$package_root/Sources/SandboxAppCore/StandaloneDemoModels.swift" \
    "$package_root/Sources/SandboxAppCore/StandaloneDemo.swift"
  swiftc -parse-as-library -swift-version 5 "$optimization" \
    -sdk "$sdk" -target "$target" \
    -module-cache-path "$build_root/module-cache" \
    -I "$build_root" -L "$build_root" -lSandboxAppCore \
    -o "$binary" \
    "$package_root/Sources/UAVSandboxApp/UAVSandboxApp.swift" \
    "$package_root/Sources/UAVSandboxApp/AppDelegate.swift" \
    "$package_root/Sources/UAVSandboxApp/SandboxAppFailures.swift" \
    "$package_root/Sources/UAVSandboxApp/SandboxAppState.swift" \
    "$package_root/Sources/UAVSandboxApp/SandboxAppModel.swift" \
    "$package_root/Sources/UAVSandboxApp/SandboxRuntimeModel.swift" \
    "$package_root/Sources/UAVSandboxApp/SandboxStatusModels.swift" \
    "$package_root/Sources/UAVSandboxApp/SandboxStatusModel.swift" \
    "$package_root/Sources/UAVSandboxApp/NativeDashboardView.swift" \
    "$package_root/Sources/UAVSandboxApp/StandaloneDemoView.swift" \
    "$package_root/Sources/UAVSandboxApp/RuntimeCompatibilityView.swift" \
    "$package_root/Sources/UAVSandboxApp/RuntimeCandidateView.swift" \
    "$package_root/Sources/UAVSandboxApp/ContentView.swift" \
    "$package_root/Sources/UAVSandboxApp/SandboxWebView.swift"
}

if xcodebuild -version >/dev/null 2>&1; then
  swift build --package-path "$package_root" --configuration "$configuration"
else
  echo "Full Xcode not selected; using the Command Line Tools build path."
  build_with_command_line_tools
fi

if [ ! -x "$binary" ]; then
  echo "SwiftPM executable was not produced at $binary" >&2
  exit 1
fi

build_icon

destination="$project_root/dist/UAV Research Sandbox.app"
contents="$destination/Contents"
rm -rf "$destination"
mkdir -p "$contents/MacOS" "$contents/Resources"
cp "$binary" "$contents/MacOS/UAVSandboxApp"
cp "$package_root/Resources/Info.plist" "$contents/Info.plist"
cp "$build_root/AppIcon.icns" "$contents/Resources/AppIcon.icns"

if command -v codesign >/dev/null 2>&1; then
  signing_identity=${MACOS_CODESIGN_IDENTITY:--}
  if [ "$signing_identity" = "-" ]; then
    codesign --force --sign - --timestamp=none "$destination"
  else
    codesign --force --sign "$signing_identity" \
      --options runtime --timestamp "$destination"
  fi
fi

echo "Built $destination"
