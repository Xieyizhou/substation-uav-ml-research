# Simulation Assets

## Substation World v3

`simulation/worlds/substation_simple.sdf` is a simplified geometric substation world for local planning and PX4 SITL control. Version 3 keeps the world lightweight and beginner-friendly while making it more recognizable as an electrical substation.

The world uses only basic SDF geometry:

- green visual-only start pad at east=0.5, north=0.5
- orange visual-only goal pad at east=16.5, north=16.5
- visual-only gravel floor and service corridor strips
- two transformer-like assemblies with simple collision boxes, bases, fins, and top bushings
- three blue/teal cabinet assemblies with simple collision boxes, bases, and front panels
- grounded gantry poles with high visual-only crossbeams, busbars, and insulators
- low boundary walls that leave the southwest takeoff zone clear

The scene is still intentionally simplified and is not photorealistic. Visual details are used to make the world readable in Gazebo, while collision geometry and the A* obstacle config remain simple for planning and control experiments.

The takeoff zone is intentionally clear from approximately east=-1..3 and north=-1..3. Do not place collision obstacles in that area.

## Coordinate Convention

Gazebo and the A* grid use the same horizontal convention:

```text
Gazebo x = east_m
Gazebo y = north_m
Gazebo z = up_m

A* grid x = east
A* grid y = north
```

The flight code converts A* grid cells into local NED waypoints:

```text
local east_m = (grid x + 0.5) * resolution_m
local north_m = (grid y + 0.5) * resolution_m
local down_m = -altitude_m
```

This cell-center rule keeps the planner, Gazebo world, and MAVSDK local NED targets aligned. Examples with `resolution_m = 1.0`:

```text
start_cell [0, 0] -> east=0.5, north=0.5
goal_cell [16, 16] -> east=16.5, north=16.5
```

For a 20 m x 20 m world, the visible `substation_floor` is centered at Gazebo `x=10`, `y=10` and covers `x=0..20`, `y=0..20`. Obstacle rectangles in `config/substation_obstacles.json` use inclusive grid cell indices.

The visible substation map is the custom 20m x 20m floor at `x=0..20`, `y=0..20`. The default Gazebo ground visual is hidden to avoid confusing it with the A* grid. The `ground_plane` model remains as a physics-only collision plane.

A subtle visual-only 1 m grid is drawn on the custom floor and aligns with the A* grid. Small visual-only origin and axis markers help debug orientation:

- white origin marker near `x=0`, `y=0`
- blue east-axis marker near `y=0`
- green north-axis marker near `x=0`

## Shared Obstacle Map

`config/substation_obstacles.json` describes the same world concept as a 2D grid map for A*. The planner is still 2D A*, but the obstacle loader now separates raw physical footprints from inflated planning obstacles.

Each obstacle has horizontal grid footprint fields plus:

```json
"visual_category": "transformer",
"z_min_m": 0.0,
"z_max_m": 1.8
```

The horizontal `x_min`, `x_max`, `y_min`, and `y_max` fields are raw physical footprint cells that correspond to Gazebo objects. They are not inflated keepouts.

The default flight settings are:

```json
"altitude_m": 1.5,
"vertical_safety_margin_m": 0.3,
"horizontal_inflation_cells": 1
```

An obstacle blocks the path only when the selected flight altitude overlaps the obstacle vertical range with the safety margin:

```text
flight_altitude_m + vertical_safety_margin_m >= z_min_m
flight_altitude_m - vertical_safety_margin_m <= z_max_m
```

Blocking obstacles are inflated horizontally by `horizontal_inflation_cells` before A* runs. Inflation is clipped to the grid boundary, and the configured start and goal cells are kept free with a warning if inflation would cover them. A* uses the inflated blocking cells; Gazebo objects correspond to the raw physical footprint cells.

Obstacle entries correspond to the major collision objects in the SDF world:

- `transformer_1`
- `transformer_2`
- `cabinet_1`
- `cabinet_2`
- `cabinet_3`
- `switchgear_south`
- `pole_west_1`
- `pole_west_2`
- `pole_east_1`
- `pole_east_2`

The previous obstacle map over-blocked the internal area after inflation, so A* tended to route around the outer boundary. The current layout uses compact raw transformer/cabinet footprints, individual one-cell pole footprints, and a compact `switchgear_south` cabinet that guides the route into an internal inspection corridor. Low boundary walls remain in the Gazebo world as context, but they are not planner obstacles at the default 1.5 m flight altitude. Visual-only floor markings, start markers, and goal markers are not included as blocking obstacles.

The JSON is not a full physics export from Gazebo. It is a beginner-friendly planning map that keeps the start area clear, leaves the goal cell free, and creates a non-straight route around multiple substation obstacle groups.

The A* map uses separate local keepout cells around each pole rather than one continuous gantry wall. This keeps the visible gantry realistic while preserving inspection corridors through the middle of the map. The configured start remains `[0, 0]`, the goal remains `[16, 16]`, and the goal marker remains at east=16.5, north=16.5.

The dry-run preview now draws both layers:

- light gray cells: raw physical footprints, useful for comparing against Gazebo objects
- dark hatched cells: inflated planning obstacles used by A*
- blue/orange/red path layers: outbound path, simplified waypoints, and return route

`python main.py astar preview` validates the height-aware obstacle config before previewing or flying. It checks that the start and goal are inside the map, start/goal are not blocked, the nearby start zone cells are free, obstacle footprints and vertical ranges are sane, the goal is not too close to inflated obstacles, and a path exists. If the goal is adjacent to inflated blocking cells, it prints nearby safer candidate goal cells but does not automatically change the configured goal.

## Analysis Improvements

`python main.py report analyze` can overlay the raw and inflated obstacle layers on the trajectory plot when given `--obstacle-config config/substation_obstacles.json`. It uses the altitude recorded in the log when available, otherwise it uses `altitude_m` from the obstacle config.

The analysis output:

- separates outbound and return actual trajectories
- avoids drawing misleading long lines across phase or route changes
- overlays planned A* waypoints and labels
- writes a waypoint reached summary to `summary.md`
- checks whether logged local positions entered raw physical footprint cells
- checks whether logged local positions entered inflated safety-buffer cells
- reports blocking versus ignored obstacles in `summary.md`
- writes `collision_points.csv` and `collision_zoom.png` only when raw physical collision or inflated safety-buffer entries exist
- separates raw physical footprint entries from inflated safety-buffer entries
- writes warnings and generated file metadata to `manifest.json`
- writes normalized perception risk metrics when perception columns exist

Interpretation:

- `raw_physical_collision_detected: yes` means logged local XY entered a cell corresponding to a Gazebo object footprint. This is the more serious finding.
- `inflated_safety_buffer_entry_detected: yes` means logged local XY entered the inflated A* keepout area. This is a clearance/safety-buffer violation, not always a physical collision.
- Normalized perception metrics compare sample ratios and percent time in each risk level, which is fairer than raw sample counts when `slow_down` runs last longer than `log_only` runs.

## SDF Z Placement

In SDF, a box or cylinder pose `z` usually places the center of the object, not its bottom. To make an object rest on the Gazebo ground plane at `z=0`, set:

```text
pose z = object height / 2
```

Examples:

- a box with height `0.9` should use pose z `0.45`
- a cylinder with length `4.5` should use pose z `2.25`
- a thin visual floor with height `0.02` should use pose z `0.01`

Floating objects usually mean the pose z or geometry height is wrong. Objects that should sit on the ground should have their bottom at z=0. Visual-only pads and floor markings can be extremely thin and sit just above the ground to avoid z-fighting.

## Manual PX4 Usage

Do not copy files into PX4 automatically from this project. After modifying `substation_simple.sdf`, manually copy it into PX4's world folder:

```bash
cp ~/projects/drone-ai/simulation/worlds/substation_simple.sdf ~/PX4-Autopilot/Tools/simulation/gz/worlds/
```

Then fully restart PX4 and Gazebo:

```bash
pkill -f "px4"
pkill -f "gz sim"
pkill -f "gz"

cd ~/PX4-Autopilot
source .venv/bin/activate
export OpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4"
PX4_GZ_WORLD=substation_simple make px4_sitl gz_x500
```

If a Gazebo simulation is already running, stop it first. `PX4_GZ_WORLD` can be ignored when an existing simulation is already active.

## Troubleshooting PX4 Arming in the Custom World

If PX4 connects but refuses to arm with messages like:

```text
Preflight Fail: Strong magnetic interference
Preflight Fail: no heading reference
```

the Gazebo world usually has a missing or incorrect magnetic field or global reference. `substation_simple.sdf` should keep a world-level `<magnetic_field>` block and a world-level `<spherical_coordinates>` block similar to PX4's default Gazebo world.

After changing either block or changing the world layout, copy the world into PX4 again and fully restart PX4/Gazebo with the commands above.

## Preview the Planning Map

From this project:

```bash
cd ~/projects/drone-ai
source .venv/bin/activate
python main.py astar preview --obstacle-config config/substation_obstacles.json
```

Preview outputs:

```text
outputs/as_preview/grid_path.png
outputs/as_preview/path_preview.json
```

Preview outbound and return path:

```bash
cd ~/projects/drone-ai
source .venv/bin/activate
python main.py astar preview --obstacle-config config/substation_obstacles.json --return-home
```

Height-aware preview at the default low flight altitude:

```bash
cd ~/projects/drone-ai
source .venv/bin/activate
python main.py astar preview --obstacle-config config/substation_obstacles.json --return-home --altitude 1.5
```

Fly the substation A* route:

```bash
cd ~/projects/drone-ai
source .venv/bin/activate
python main.py astar fly --obstacle-config config/substation_obstacles.json --return-home
```

Safer controller test options:

```bash
python main.py astar fly --obstacle-config config/substation_obstacles.json --return-home --altitude 1.5 --max-speed 0.8 --return-speed-scale 0.7 --waypoint-acceptance 0.3
```

Analyze the newest A* flight log with obstacle validation:

```bash
python main.py report analyze --obstacle-config config/substation_obstacles.json
```

Full debug analysis:

```bash
python main.py report analyze --obstacle-config config/substation_obstacles.json --debug-plots
```

Default analysis is for normal experiment review. It writes the core files `summary.md`, `manifest.json`, `traj.png`, `traj_with_obstacles.png` when an obstacle config is available, `error.png`, and `perception_risk_timeline.png` when perception columns exist. Use `--debug-plots` to diagnose altitude, velocity, yaw, waypoint switching, nearest-obstacle distance, and perception count details.

Trajectory plot outputs:

- `traj.png`: clean route-following overview with actual outbound/return path, planned A* waypoints, start, and final/end markers.
- `traj_with_obstacles.png`: obstacle validation view with raw physical footprints, inflated planning cells, and collision/buffer entry points when available.

## Simulated Perception Prototype

The first perception stage does not use real Gazebo LiDAR, depth, camera data, neural networks, or YOLO. It uses `config/substation_obstacles.json` to emulate a simple forward local obstacle detector and logs detections during flight when explicitly enabled.

Test without PX4 or Gazebo:

```bash
python main.py check perception \
  --obstacle-config config/substation_obstacles.json
```

Fly with simulated perception logging:

```bash
python main.py astar fly \
  --obstacle-config config/substation_obstacles.json \
  --return-home \
  --altitude 1.5 \
  --max-speed 0.8 \
  --return-speed-scale 0.7 \
  --waypoint-acceptance 0.3 \
  --enable-perception \
  --detection-range 4.0 \
  --detection-fov 90 \
  --warning-distance 2.0 \
  --danger-distance 1.0 \
  --risk-action log_only
```

Risk levels are `clear`, `detected`, `warning`, and `danger`. Defaults are detection range `4.0 m`, warning distance `2.0 m`, and danger distance `1.0 m`. The default `--risk-action log_only` only records risk. Use `--risk-action slow_down` for a cautious prototype that reduces speed on warning/danger.

Analyze the resulting log with the normal analyzer:

```bash
python main.py report analyze --obstacle-config config/substation_obstacles.json
```

When perception columns are present, normal analysis writes `perception_risk_timeline.png` and a `Perception Summary` section in `summary.md`. The summary includes total duration, risk sample counts and ratios, time and percent time in clear/detected/warning/danger, and nearest-obstacle distance statistics. Re-run with `--debug-plots` to also create `perception_timeline.png` and `detection_count_over_time.png`.

Summarize all analyzed A* experiment runs:

```bash
python main.py report summarize
```

The summary tool scans `outputs/as_*/summary.md` and `outputs/as_*/manifest.json` and writes:

```text
outputs/experiment_summary.csv
outputs/experiment_summary.md
```

The experiment summary groups runs by `risk_action` and compares normalized warning/danger ratios, percent time in warning/danger, obstacle distance statistics, duration, and collision/buffer flags. This is intended for comparing `log_only` and `slow_down` runs with different durations.

Unified commands:

```bash
python main.py astar preview --obstacle-config config/substation_obstacles.json --return-home
python main.py astar fly --obstacle-config config/substation_obstacles.json --return-home
python main.py report analyze --obstacle-config config/substation_obstacles.json
```
