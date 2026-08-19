#!/bin/bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
package_root="$project_root/apps/macos/SandboxApp"
developer_dir="${DEVELOPER_DIR:-$(xcode-select -p 2>/dev/null || true)}"

if [[ -z "$developer_dir" || "$developer_dir" == */CommandLineTools ]]; then
    echo "macOS App tests require the full Xcode toolchain, not Command Line Tools alone." >&2
    echo "Install Xcode, then run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
    exit 2
fi

if [[ ! -x "$developer_dir/usr/bin/xcodebuild" && ! -x "$developer_dir/../usr/bin/xcodebuild" ]]; then
    echo "DEVELOPER_DIR does not identify a complete Xcode installation: $developer_dir" >&2
    exit 2
fi

export DEVELOPER_DIR="$developer_dir"
swift_path="$(xcrun --find swift)"
sdk_path="$(xcrun --sdk macosx --show-sdk-path)"
sdk_version="$(xcrun --sdk macosx --show-sdk-version)"

echo "Xcode: $(xcodebuild -version | tr '\n' ' ')"
echo "Swift: $($swift_path --version | head -1)"
echo "macOS SDK: $sdk_version ($sdk_path)"

exec "$swift_path" test --package-path "$package_root"
