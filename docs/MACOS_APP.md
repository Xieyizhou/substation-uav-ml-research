# macOS Sandbox App

The native macOS application includes a small deterministic Demo runtime and
starts the existing local Sandbox HTTP service for Development and Formal. It
does not duplicate flight, simulator, collection, or training logic. Those
workflows remain in the Python application and keep their existing allowlists,
single-job lock, timeouts, output budgets, receipts, and recovery rules.

## Current scope

The native application provides:

- a repository-free, native Demo classifier with a deterministic saved artifact;
- repository selection and validation for Development and Formal;
- Demo, Development, and Formal profile selection;
- verified runtime discovery for Python, PX4, Gazebo, OpenCV, and Qt;
- an inspected candidate list for every runtime component, with explicit selection;
- persistent absolute runtime paths with lightweight launch-time revalidation;
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
installed dependencies. Demo Profile is compiled into the App and writes its
latest deterministic result under the user's Application Support directory.

After the local service starts, **Get started** presents profile-aware setup
status. It detects the project interpreter, PX4 checkout and SITL build,
Gazebo, MAVSDK, and simulation worlds. Installation remains an explicit user
operation: the App exposes copyable commands and official documentation, then
rechecks the environment without running a package manager itself.

## Runtime compatibility

Standalone Demo never invokes the Runtime Compatibility Manager. Development
and Formal use this sequence before starting the Python service:

```text
Discover → Inspect → Validate → Select → Persist → Revalidate
```

The project virtual environment has first priority. A saved and previously
verified Python is considered next, followed by compatible candidates found in
known command locations. Python older than 3.11 or missing `mavsdk` is not
selected. ML-only packages such as PyTorch and ONNX remain optional for the
basic Development runtime.

PX4 inspection verifies the Git checkout, required SITL/Gazebo source
structure, current commit, and available tag description. Gazebo inspection
runs the selected absolute `gz` executable and verifies its Sim major version;
finding a command named `gz` is not sufficient. OpenCV prefers `opencv@4`, and
Qt prefers `qt@5`.

The version-controlled policy is
`config/sandbox/runtime_compatibility.json`. A verified local selection is
stored at:

```text
~/Library/Application Support/UAV Research Sandbox/runtime-profile.json
```

The file records the project, component paths, versions and identities, result,
and validation time. Development permits an untested compatible version with a
warning; Formal blocks untested runtimes. Missing, unsupported, or changed
runtime identities block both profiles until the user selects **Check runtime**
or chooses another inspected candidate. **Candidates…** shows repository,
manual, saved, environment, `PATH`, default-location, and Homebrew discoveries;
each row reports its version and compatibility result. Manual selection also
supports OpenCV and Qt prefixes. **Use automatic priority** clears all manual
overrides and reruns deterministic discovery.

The selected paths are passed to the service and flight launcher through a
controlled environment. The verified tool directories lead `PATH`, and the
launcher also receives explicit Python, Gazebo, PX4, OpenCV, and Qt paths. The
App never changes a PX4 branch, upgrades Homebrew packages, modifies shell
profiles, or terminates simulator processes it does not own. A likely external
PX4/Gazebo collision blocks launch with the process identity instead.

Multiple Python, PX4, or Gazebo installations may therefore coexist. Changing
or removing a saved executable, changing PX4 `HEAD`, or replacing a versioned
dependency produces `changed_since_validation` rather than a silent fallback.

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

The start screen and candidate sheet provide manual selectors for the Python
executable, PX4 root, `gz` executable, OpenCV prefix, and Qt prefix. These
choices are inspected before they are saved; selecting a path never installs,
upgrades, removes, or rewrites that runtime.

## Preview installer

Create a versioned DMG, ZIP, release manifest, and checksum list from the
repository root:

```bash
./scripts/package_macos_release.sh 0.6.0
cd dist/releases
shasum -a 256 -c UAV-Research-Sandbox-v0.6.0-macos-*-SHA256SUMS
```

The DMG presents the App beside an Applications shortcut and includes a short
installation note. The ZIP is retained for automated or scripted installation.
The JSON release manifest records the artifact hashes, architecture, signing
tier, included standalone Demo, and advanced-profile dependencies so downstream
tooling cannot mistake the preview for a standalone simulator.

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
builds the application, verifies its ad-hoc signature and release checksums,
and publishes the DMG, ZIP, manifest, and checksum list as a GitHub prerelease.
It never packages datasets, model weights, simulator installations, or local
experiment outputs.

## Unsigned Beta installer

The current Beta is intended for repeatable GitHub distribution before Apple
Developer credentials are available. It is built only from a clean tracked
worktree, records the exact source commit, creates a versioned ZIP and DMG, and
binds their byte counts and SHA256 identities into a verified release manifest:

```bash
./scripts/package_macos_beta.sh 0.6.0
cd dist/releases
shasum -a 256 -c UAV-Research-Sandbox-v0.6.0-macos-*-SHA256SUMS
```

The manual **macOS unsigned Beta release** workflow repeats Swift tests and
release verification on a fresh GitHub macOS runner before publishing a
prerelease. The App is ad-hoc signed so it can execute locally, but it has no
Developer ID identity and is not Apple-notarized. The included installation
note directs users to macOS **Open Anyway** if Gatekeeper blocks first launch;
it never asks users to disable Gatekeeper.

This is a reproducible build procedure and a cryptographically verifiable
artifact identity, not a claim that ZIP or DMG bytes are bit-for-bit identical
across SDK versions. The source commit, architecture, App version, build tier,
artifact hashes, and external-runtime boundary are sufficient to reproduce and
audit the same release inputs.

## Future notarized Beta pipeline

The notarized Beta uses the same bundle boundary, but requires a Developer ID Application
certificate, hardened-runtime signing, Apple notarization, and stapled tickets
for both the App and DMG. The release command deliberately fails before
building when either credential is absent:

```bash
MACOS_CODESIGN_IDENTITY="Developer ID Application: Example (TEAMID)" \
MACOS_NOTARY_PROFILE="uav-sandbox-notary" \
  ./scripts/package_macos_notarized_beta.sh 0.6.0
```

Create the named notary profile with `xcrun notarytool store-credentials` first.
The manual **macOS notarized Beta release** workflow performs the same checks on
GitHub's macOS runner and publishes a prerelease only after `codesign`,
Gatekeeper, stapler, manifest, and checksum verification pass. Configure these
repository secrets:

- `MACOS_DEVELOPER_ID_P12_BASE64`
- `MACOS_DEVELOPER_ID_P12_PASSWORD`
- `MACOS_DEVELOPER_ID_APPLICATION`
- `APP_STORE_CONNECT_KEY_P8_BASE64`
- `APP_STORE_CONNECT_KEY_ID`
- `APP_STORE_CONNECT_ISSUER_ID`

The workflow deletes its temporary keychain and credential files even when a
release step fails. The Beta manifest must declare `developer_id` signing and
`stapled` notarization; an ad-hoc artifact cannot pass as Beta.

## Distribution boundary

The preview and unsigned Beta are ad-hoc signed; only the latter requires clean
sources and records a Beta provenance contract. The future notarized Beta adds
Developer ID and Apple trust without changing the product boundary: Demo is
standalone, while Development and Formal still require a selected repository,
Python environment, and external simulator toolchain. A public stable release
should additionally pass a clean-machine install and flight smoke on each
supported macOS/architecture combination.
