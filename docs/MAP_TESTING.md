# Substation Test Maps

The map catalog keeps each Gazebo world paired with its PX4 spawn pose and A*
obstacle configuration. This prevents a flight from using obstacle data for a
different world.

## Available maps

| ID | Difficulty | Size | Intended use |
| --- | --- | --- | --- |
| `training` | 1/5 | 16 x 16 m | First flight, smoke tests, basic return-home |
| `simple` | 2/5 | 20 x 20 m | Classic baseline and regression comparison |
| `medium` | 3/5 | 24 x 24 m | Mixed equipment and crossing corridors |
| `complex` | 4/5 | 28 x 28 m | Perception and local-replan experiments |
| `extreme` | 5/5 | 32 x 32 m | Long-route and active-replan stress tests |

## Interactive workflow

Stop PX4 first, then run:

```bash
source .venv/bin/activate
python main.py map
```

Choose a number, then start the selected world:

```bash
python main.py map start
```

In a second terminal, the official runners automatically use the selected
map's obstacle config:

```bash
python main.py experiment run static
```

## Command-line workflow

```bash
python main.py map list
python main.py map current
python main.py map use medium
python main.py map preview --return-home
python main.py map start
```

You can select and start in one command:

```bash
python main.py map start extreme
```

`use` and `start` refuse to change the selection while the project-managed PX4
launcher is alive. Stop its terminal with `Ctrl+C` before switching maps.

## Destination point workflow

Each map has five validated destination presets: `top_right`, `center`, `left`,
`bottom`, and `right`. The menu and labels are in English:

```bash
python main.py point
python main.py point list
python main.py point current
python main.py point use center
```

Selections are saved separately for each map. You can change the destination
while PX4/Gazebo is running, but not while a managed flight is in progress.
The next preview, direct flight, or formal experiment automatically uses it.
The red Gazebo `goal_marker` is moved immediately when the running world exposes
its pose service. A runtime SDF copy also applies the selection on every map
start, so the source world files remain deterministic.

If switching is blocked, inspect the latest `astar_*.status.json`. A terminal
status with `landing_confirmed: true` should now be followed by bounded MAVSDK
cleanup and automatic PID removal, even if a telemetry subscription is slow to
cancel.

To inspect another map's points without changing the current map:

```bash
python main.py point --map extreme list
```

## Compact flight tasks

The compact task layer keeps each user-facing program small while reusing the
tested flight engine:

```bash
python main.py task
python main.py task list
python main.py task run preview_route
python main.py task run fly_to_point
python main.py task run fly_round_trip
python main.py task run fly_with_perception
python main.py task run fly_with_replan
```

Equivalent single-purpose programs live under `scripts/flight/tasks/`.
Advanced flight flags may be appended after `--`, for example:

```bash
python main.py task run fly_to_point -- --max-speed 0.6
```

## Direct A* commands

When `--obstacle-config` is omitted, `main.py astar ...` automatically uses the
current catalog selection:

```bash
python main.py astar preview --return-home
python main.py astar fly --return-home --altitude 1.5
python main.py astar analyze
```

An explicit `--obstacle-config` still overrides the selected map for advanced
offline analysis. Do not use that override for live flight unless it matches the
running Gazebo world.

Advanced commands also follow the current map selection:

```bash
python main.py astar preview --return-home
python main.py check perception
```

For an offline one-off override, pass `--obstacle-config PATH`. The old built-in
10 x 10 A* grid is still available only when requested explicitly:

```bash
python main.py astar preview --use-built-in-grid
```

## Regenerating the generated maps

The four generated worlds and obstacle configs are deterministic:

```bash
python main.py map generate
```

Map generation is refused while a managed flight or PX4 session is running.
After regeneration, run `python main.py check maps` before starting PX4.
