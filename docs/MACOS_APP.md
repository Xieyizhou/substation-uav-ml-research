# macOS Sandbox App

The native macOS shell starts and supervises the existing local Sandbox HTTP
service. It does not duplicate flight, simulator, collection, or ML logic.
Those workflows remain in the Python application and keep their existing
allowlists, single-job lock, timeouts, output budgets, receipts, and recovery
rules.

## Current scope

The native application provides:

- repository selection and validation;
- Demo, Development, and Formal profile selection;
- `.venv/bin/python` discovery with a `PATH` fallback;
- profile bootstrap before launch;
- loopback-only service health checks;
- connection to an already-running local service without taking ownership;
- an embedded `WKWebView` restricted to loopback navigation;
- a native read-only status page for profile, environment, runtime, storage,
  and managed-job health;
- a stable application icon and versioned bundle metadata;
- bounded service logs; and
- graceful interrupt, terminate, and kill fallback for a service started by
  the App.

PX4, Gazebo, datasets, model weights, and the Python environment are not
bundled. Development and Formal profiles continue to use the repository's
installed dependencies. Demo Profile remains the portable first-run path.

After the local service starts, **Get started** presents profile-aware setup
status. It detects the project interpreter, PX4 checkout and SITL build,
Gazebo, MAVSDK, and simulation worlds. Installation remains an explicit user
operation: the App exposes copyable commands and official documentation, then
rechecks the environment without running a package manager itself.

The App version, Sandbox product milestone, operator API, and gate schema are
separate compatibility identities. Their shared source is
`config/sandbox/version.json`; the native status page displays both the App and
Sandbox versions.

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

The status page polls five read-only loopback endpoints every five seconds. It
does not receive the operator token and cannot start or stop flight jobs. Use
the **Workbench** section for the existing, guarded workflow controls.

If the App cannot infer the repository, choose it from the start screen. The
folder must contain `main.py` and `src/sandbox`. For scripted launches, the
initial folder can be supplied to the executable with:

```bash
UAV_SANDBOX_PROJECT_ROOT="$PWD" \
  "dist/UAV Research Sandbox.app/Contents/MacOS/UAVSandboxApp"
```

## Preview archive

Create a versioned ZIP and matching checksum from the repository root:

```bash
./scripts/package_macos_release.sh 0.2.1
cd dist/releases
shasum -a 256 -c UAV-Research-Sandbox-v0.2.1-macos-arm64.zip.sha256
```

After an App-managed Development flight smoke completes, bind its job receipt,
flight summary, source commit, cleanup status, and versions into one local gate:

```bash
.venv/bin/python main.py sandbox --profile development development-app-gate \
  --workflow-receipt outputs/sandbox/operator/jobs/JOB_ID/workflow_receipt.json \
  --flight-summary outputs/sandbox/flight_smoke/RUN_ID/summary.json \
  --output outputs/sandbox/development-app-gate/latest
```

The gate fails if the mission was not completed with confirmed landing, a
simulator process remains, the tracked worktree changed, the scenario is blind,
or PNG dataset payloads were produced.

The manual **macOS preview release** GitHub Actions workflow runs Swift tests,
builds the application, verifies its ad-hoc signature and checksum, and
publishes both files as a GitHub prerelease. It never packages datasets, model
weights, simulator installations, or local experiment outputs.

## Distribution boundary

The preview archive is not a standalone public installer. It is ad-hoc signed,
is intended for Apple Silicon development machines, and still uses a selected
repository plus its Python environment. Public distribution still requires a
Developer ID certificate, hardened-runtime entitlements, notarization, and a
clean-machine Gatekeeper test. A later milestone may bundle the
dependency-free Demo source and runtime; the full simulator remains an
explicitly detected external toolchain.
