# macOS Sandbox App

The native macOS shell starts and supervises the existing local Sandbox HTTP
service. It does not duplicate flight, simulator, collection, or ML logic.
Those workflows remain in the Python application and keep their existing
allowlists, single-job lock, timeouts, output budgets, receipts, and recovery
rules.

## Current scope

The first native milestone provides:

- repository selection and validation;
- Demo, Development, and Formal profile selection;
- `.venv/bin/python` discovery with a `PATH` fallback;
- profile bootstrap before launch;
- loopback-only service health checks;
- connection to an already-running local service without taking ownership;
- an embedded `WKWebView` restricted to loopback navigation;
- bounded service logs; and
- graceful interrupt, terminate, and kill fallback for a service started by
  the App.

PX4, Gazebo, datasets, model weights, and the Python environment are not
bundled. Development and Formal profiles continue to use the repository's
installed dependencies. Demo Profile remains the portable first-run path.

## Build

Run from the repository root:

```bash
swift test --package-path apps/macos/SandboxApp
./scripts/build_macos_app.sh release
open "dist/UAV Research Sandbox.app"
```

The build script uses SwiftPM when full Xcode is selected. With Command Line
Tools only, it uses the installed macOS SDK directly. SwiftPM tests still
require a matching compiler and SDK; installing a stable Xcode release is the
recommended development setup. The script creates an ad-hoc-signed local
application, and the generated `dist/` directory is ignored by Git.

If the App cannot infer the repository, choose it from the start screen. The
folder must contain `main.py` and `src/sandbox`. For scripted launches, the
initial folder can be supplied to the executable with:

```bash
UAV_SANDBOX_PROJECT_ROOT="$PWD" \
  "dist/UAV Research Sandbox.app/Contents/MacOS/UAVSandboxApp"
```

## Distribution boundary

The local `.app` is not yet a public installer. GitHub distribution requires a
stable application icon, Developer ID signing, hardened-runtime entitlements,
notarization, release archives, and a clean-machine installation gate. A later
milestone may bundle the dependency-free Demo source and runtime, while the
full simulator remains an explicitly detected external toolchain.
