# Read-only research inspector

The local research inspector presents collection state, environment checks,
bounded logs, runtime observations, scenario progress, and paginated recording
frames. It is an observation interface. Existing command-line services and
scientific contracts remain the source of truth.

Run the local checks from the repository root without installing additional
dependencies:

```bash
python main.py sandbox doctor
python main.py sandbox status
python main.py sandbox serve
```

The web server can also be started directly:

```bash
python3 -m src.inspection.app
```

Then open `http://127.0.0.1:8765`. Use `--project-root`, `--host`, or `--port`
to override local defaults. The default project configuration inspects
`data/research/visual_collection_v2` and falls back to the tracked version 2
plan when the local plan is absent.

## Safety boundary

- The server implements GET queries only and returns 405 for POST requests.
- Approved collection, recording, log, and PNG paths are resolved centrally.
- Browser input cannot select arbitrary paths or non-PNG payloads.
- Logs are selected by scenario and fixed kind, escaped, and bounded to at most
  1,000 lines per request.
- Frame pages read ordered manifests and load PNG payloads only on demand.
- Blind scenario detail, logs, recordings, frames, and predictions are not
  exposed. Aggregate blind scenario counts remain visible.
- Process observations return only the process role and PID. Command lines and
  environment data are never returned to the browser.
- Missing programs and processes are reported; nothing is installed, started,
  stopped, retried, validated, or materialized.

The inspector does not replace collection status, validation, training,
evaluation, replay, or flight-control commands.

## Flight smoke gate

Use a flight smoke run to verify one non-blind route before recording dataset
frames:

```bash
python main.py sandbox flight-smoke \
  --scenario-id development-01-reactor_centered_v2-3004
```

The command starts the existing PX4 and Gazebo workflow, flies the complete
route, confirms landing, and writes bounded diagnostic output under
`outputs/sandbox/flight_smoke`. It does not launch the visual recorder and does
not create a recording receipt or dataset identity. Blind scenarios are
rejected.

The recommended local workflow is:

1. Run `sandbox doctor` and `sandbox status`.
2. Pass a flight smoke run without recording frames.
3. Run one collection scenario and validate its recording.
4. Expand to a small multi-scenario gate before a full collection batch.
