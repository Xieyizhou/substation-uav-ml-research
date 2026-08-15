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
    "$package_root/Sources/SandboxAppCore/ProcessExecution.swift"
  swiftc -parse-as-library -swift-version 5 "$optimization" \
    -sdk "$sdk" -target "$target" \
    -module-cache-path "$build_root/module-cache" \
    -I "$build_root" -L "$build_root" -lSandboxAppCore \
    -o "$binary" \
    "$package_root/Sources/UAVSandboxApp/UAVSandboxApp.swift" \
    "$package_root/Sources/UAVSandboxApp/SandboxAppModel.swift" \
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

destination="$project_root/dist/UAV Research Sandbox.app"
contents="$destination/Contents"
rm -rf "$destination"
mkdir -p "$contents/MacOS" "$contents/Resources"
cp "$binary" "$contents/MacOS/UAVSandboxApp"
cp "$package_root/Resources/Info.plist" "$contents/Info.plist"

if command -v codesign >/dev/null 2>&1; then
  codesign --force --sign - --timestamp=none "$destination"
fi

echo "Built $destination"
