# Unified Command Reference

Run all public commands from the repository root with the project virtual
environment active:

```bash
source .venv/bin/activate
python main.py --help
```

`main.py` is the supported user entry point. Files under `scripts/` are now
internal launchers, small single-purpose task examples, map builders, or release
utilities rather than a second public command system.

## Daily workflow

| Goal | Command |
| --- | --- |
| Select a map | `python main.py map` |
| Start PX4/Gazebo with the selected map | `python main.py map start` |
| Select a destination | `python main.py point` |
| List compact tasks | `python main.py task list` |
| Preview the selected route | `python main.py task run preview_route` |
| Fly to the selected point | `python main.py task run fly_to_point` |
| Fly out and return | `python main.py task run fly_round_trip` |
| Fly with perception | `python main.py task run fly_with_perception` |
| Fly with active replanning | `python main.py task run fly_with_replan` |

Advanced task overrides go after `--`:

```bash
python main.py task run fly_to_point -- --max-speed 0.6
```

## Maps and points

```bash
python main.py map list
python main.py map current
python main.py map use complex
python main.py map preview --return-home
python main.py map start complex
python main.py map start complex --vehicle-model x500_research
python main.py map generate

python main.py point list
python main.py point current
python main.py point use center
python main.py point --map extreme list
```

Changing or regenerating a map is refused while a project-managed flight or PX4
session is running. Changing a point is refused only during a managed flight.

To diagnose launcher paths and dependencies without copying a world or starting
PX4, run:

```bash
python main.py map check
```

The launcher derives the project root from its own location, so it works from
any current directory and ignores a stale `PROJECT_ROOT` environment variable.

## Advanced A* controls

```bash
python main.py astar preview --return-home --altitude 1.5
python main.py astar fly --return-home --max-speed 0.8
python main.py astar analyze
```

The current map and point are used automatically. `astar analyze` remains a
short compatibility alias; `report analyze` is the canonical report command.
All flight-runner options remain available through `astar preview` and
`astar fly`.

## Official experiments

PX4/Gazebo must already be running:

```bash
python main.py experiment list
python main.py experiment run static
python main.py experiment run perception
python main.py experiment run replan-log
python main.py experiment run active-replan
python main.py experiment run-all --trials 3
```

## Reports

```bash
python main.py report analyze
python main.py report analyze --log data/logs/astar_example.csv
python main.py report summarize
python main.py report compare --mode both --min-runs-per-stage 3
python main.py report validate-active --latest 3
```

## Sensors, datasets, models, and studies

```bash
python main.py sensor list
python main.py sensor check --source gazebo_lidar_2d
python main.py sensor record --output data/research/scans/run.jsonl
python main.py sensor replay --input data/research/scans/run.jsonl

python main.py data world \
  --source simulation/worlds/substation_complex.sdf \
  --output outputs/research/worlds/complex_2001.sdf \
  --map complex --seed 2001
python main.py data collect \
  --input data/research/scans/run.jsonl \
  --output data/research/lidar_v2 \
  --map complex --target center --seed 2001
python main.py data validate --dataset data/research/lidar_v2
python main.py data summarize --dataset data/research/lidar_v2

python main.py model train \
  --dataset data/research/lidar_v2/samples.jsonl \
  --output models/lidar/risk_v2
python main.py model inspect --package models/lidar/risk_v2
python main.py model predict \
  --model models/lidar/risk_v2 \
  --dataset data/research/lidar_v2/samples.jsonl \
  --output outputs/research/predictions/risk_v2.jsonl
python main.py model evaluate \
  --predictions outputs/research/predictions/risk_v2.jsonl
python main.py model benchmark \
  --predictions outputs/research/predictions/risk_v2.jsonl
python main.py model protocol --config config/perception/research_protocol.json

python main.py study create --name risk-cnn-v2 \
  --candidate models/lidar/risk_v2
python main.py study run STUDY_ID --tier replay
python main.py study run STUDY_ID --tier closed-loop
python main.py study run STUDY_ID --tier formal
python main.py study resume STUDY_ID
python main.py study status STUDY_ID
python main.py study compare STUDY_ID
python main.py study promote STUDY_ID
```

Optional training and inference dependencies are pinned in
`requirements-ml.txt`. Raw datasets and model weights are ignored by Git.
See `docs/ML_RESEARCH_PLATFORM.md` for data isolation, safety, and benchmark
requirements.

Study state lives in ignored `outputs/research/registry.sqlite`. `study run`
is idempotent: it preserves completed runs, writes a machine-readable worker
queue, and imports any matching result JSON. `study resume` resets only
interrupted/failed work. Promotion remains blocked until formal paired results
show no safety regression and at least one quality or latency improvement.

## Checks

```bash
python main.py check environment
python main.py check perception
python main.py check replan
python main.py check maps
python main.py check tests
python main.py check all
```

`check all` combines the dependency check, perception smoke test, and complete
offline regression suite. It does not start PX4 or fly the vehicle.
