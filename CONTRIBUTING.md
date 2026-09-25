# Contributing

Thank you for your interest in this project.

This repository is a simulation-first UAV autonomy research platform built
around PX4 SITL, Gazebo, MAVSDK-Python, A* path planning, live/replayed LiDAR,
geometric and learned-risk interfaces, and local replanning.

## Development Setup

Use Python 3.12+ for the full core suite. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
```

Run commands from the repository root.

## Project Structure

- `main.py`: unified command-line entry point
- `src/cli/`: command routing and user-facing CLI logic
- `src/flight/`: flight execution and mission runtime
- `src/planner/`: A* planning and obstacle-map conversion
- `src/perception/`: simulated perception and risk states
- `src/sensors/`: live/replay sensor sources and timestamped contracts
- `src/ml/`: datasets, evaluation, ONNX, and semantic-perception research
- `src/vision/`: camera collection, visual training, evaluation, and replay
- `src/maps/`: map and destination management
- `src/logging/`: telemetry analysis, reports, plots, and comparisons
- `scripts/flight/experiments/`: official experiment launchers
- `simulation/worlds/`: Gazebo environments
- `tests/`: offline regression tests

## Contribution Scope

Keep changes focused and reviewable.

When modifying flight, planning, perception, or replanning behavior:

- describe the behavioral change clearly;
- add or update regression tests where practical;
- avoid unrelated refactors in the same pull request;
- preserve existing safety checks unless the change explicitly replaces them.

Do not modify an external PX4-Autopilot checkout as part of changes to this repository.

Do not commit generated logs, runtime outputs, caches, local environments, PID files, or IDE settings.

### AI Agent Prompt Policy

Do not commit or push AI coding-agent prompts, system/developer prompts, local
agent instructions, or tool-specific agent configuration files. This includes
`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.codex/`, `.agents/`,
`.cursor/`, `.continue/`, `.claude/`, `.gemini/`, `.codebuddy/`,
`.kiro/`, `.qoder/`, `.windsurf/`, MCP/OpenCode configuration, GitHub
Copilot instruction/prompt files, and files named like `*.instruction.md`,
`*.prompt.md`, `*agent-prompt*`, or `*system-prompt*`.

Generated code-review graph data under `.code-review-graph/` is also local
tooling state and must not be committed or pushed.

Keep these files local and untracked. Before committing, run the portable core
test suite; the repository hygiene test fails when a prohibited agent prompt
path is already tracked. Before any push, review `git status` and the commits
that are ahead of the remote branch.

## Validation

Run the portable core suite (no local research data or ML stack required):

```bash
python scripts/run_tests.py --suite core --report outputs/core-tests.json
```

For research changes, also run the relevant research tests in the environment
described in [VALIDATION.md](docs/VALIDATION.md). The full discovery suite needs
the original research artifacts; it is not the clean-clone entry point.

Compile Python sources:

```bash
python -m compileall -q src main.py scripts/run_tests.py scripts/export_source.py
# With Python 3.12+ for historical research scripts:
python -m compileall -q src scripts archive main.py
```

Check shell-script syntax:

```bash
find scripts -type f -name "*.sh" -print0 | xargs -0 -n1 bash -n
```

Check patch formatting:

```bash
git diff --check
```

PX4 and Gazebo flight validation must be performed locally. Passing offline tests does not claim real-hardware validation.

## Source and licensing

Keep raw datasets, downloaded external images, weights and generated evidence
out of Git. Add provenance and license details for any new third-party material
to [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Do not change frozen receipts
or script bytes to make historical verification pass.

## Pull Requests

A pull request should include:

- a clear summary of the change;
- the motivation for the change;
- files or modules affected;
- validation commands run;
- known limitations or follow-up work.

Keep generated output files out of pull requests unless they are small, intentional demonstration artifacts.
