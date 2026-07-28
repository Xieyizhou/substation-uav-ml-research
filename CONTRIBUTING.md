# Contributing

Thank you for your interest in this project.

This repository is a simulation-first UAV autonomy demo built around PX4 SITL, Gazebo, MAVSDK-Python, A* path planning, simulated perception, and local replanning.

## Development Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands from the repository root.

## Project Structure

- `main.py`: unified command-line entry point
- `src/cli/`: command routing and user-facing CLI logic
- `src/flight/`: flight execution and mission runtime
- `src/planner/`: A* planning and obstacle-map conversion
- `src/perception/`: simulated perception and risk states
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

Keep these files local and untracked. Before committing, run the full offline
test suite; the repository hygiene test fails when a prohibited agent prompt
path is already tracked. Before any push, review `git status` and the commits
that are ahead of the remote branch.

## Validation

Run the offline test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Compile Python sources:

```bash
python -m compileall -q src scripts main.py
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

## Pull Requests

A pull request should include:

- a clear summary of the change;
- the motivation for the change;
- files or modules affected;
- validation commands run;
- known limitations or follow-up work.

Keep generated output files out of pull requests unless they are small, intentional demonstration artifacts.
