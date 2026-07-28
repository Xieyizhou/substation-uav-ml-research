# UAV Project History

This file archives historical milestones and older workflow notes that used to live in `PROJECT_STATUS.md`.

For the current active A* substation workflow, read `PROJECT_STATUS.md`.

Some historical sections mention old script paths from before the cleanup. The old learning/demo scripts are now preserved under `legacy/learning_scripts/`.

## Current Goal
Run repeatable PX4 SITL A* substation flight experiments with simulated perception response.

## Current Recommended Workflow

The active workflow is the A* substation route using `config/substation_obstacles.json`, `simulation/worlds/substation_simple.sdf`, and the MAVSDK offboard waypoint controller in `scripts/fly_astar_path.py`.

Install project Python dependencies once:
```bash
cd ~/projects/drone-ai
source .venv/bin/activate
pip install -r requirements.txt
```

Start PX4 SITL with the substation world in Terminal A:
```bash
cd ~/projects/drone-ai
bash scripts/start_px4_substation.sh
```

Run the A* substation workflow from this project in Terminal B:
```bash
cd ~/projects/drone-ai
source .venv/bin/activate
python main.py astar preview --obstacle-config config/substation_obstacles.json --return-home --altitude 1.5
python main.py astar fly --obstacle-config config/substation_obstacles.json --return-home --altitude 1.5 --max-speed 0.8 --return-speed-scale 0.7 --waypoint-acceptance 0.3
python main.py astar analyze --obstacle-config config/substation_obstacles.json
python scripts/summarize_experiments.py
```

For perception-response comparison runs:
```bash
python main.py astar fly \
  --obstacle-config config/substation_obstacles.json \
  --return-home \
  --altitude 1.5 \
  --max-speed 1.0 \
  --return-speed-scale 0.8 \
  --waypoint-acceptance 0.4 \
  --enable-perception \
  --risk-action log_only

python main.py astar fly \
  --obstacle-config config/substation_obstacles.json \
  --return-home \
  --altitude 1.5 \
  --max-speed 1.0 \
  --return-speed-scale 0.8 \
  --waypoint-acceptance 0.4 \
  --enable-perception \
  --risk-action slow_down

python main.py astar analyze --obstacle-config config/substation_obstacles.json
python scripts/summarize_experiments.py
```

Current main workflow files:
- `main.py`
- `scripts/start_px4_substation.sh`
- `scripts/takeoff_astar_substation.sh`
- `scripts/run_substation_demo.command`
- `scripts/fly_astar_path.py`
- `scripts/analyze_astar_log.py`
- `scripts/summarize_experiments.py`
- `planners/astar_grid.py`

Generated outputs and logs are preserved. The active analysis and summary scripts do not automatically delete logs or generated output folders.

Legacy learning demo scripts, preserved for reference only:
- `takeoff_and_land.py`
- `scripts/takeoff_land_with_logging.py`
- `scripts/fly_square_offboard.py`
- `scripts/fly_waypoints_local.py`

## Legacy Milestone 1: Basic Takeoff/Landing

The original takeoff/landing milestone below is historical context only. It is no longer the active workflow.

### Completed
- Installed PX4 SITL.
- Installed MAVSDK-Python.
- Can start PX4 and Gazebo.
- Created a basic script to connect to PX4 on udpin://0.0.0.0:14540.
- Script includes connection check, health check, arm, takeoff, hover, and land.
- Added telemetry logging to `takeoff_and_land.py`.
- Telemetry is saved to `data/logs/flight_log.csv`.
- Logged fields include timestamp, latitude, longitude, relative altitude, absolute altitude, flight mode, and armed status when available.

### Historical How to Run
1. Start PX4 SITL and Gazebo.
2. Open a terminal in the project directory:
   ```bash
   cd /Users/xieyizhou/projects/drone-ai
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
4. Run the takeoff and landing script:
   ```bash
   python takeoff_and_land.py
   ```
5. After the script finishes, check the telemetry log:
   ```bash
   cat data/logs/flight_log.csv
   ```

### What Worked
- The script connects to PX4 SITL on `udpin://0.0.0.0:14540`.
- The script waits for connection and healthy global/home position before flying.
- The existing arm, takeoff, hover, and land sequence remains unchanged.
- Telemetry logging runs in the background during flight.
- The updated script passes Python syntax checking with:
  ```bash
  python3 -m py_compile projects/drone-ai/takeoff_and_land.py
  ```

### Historical Next Steps
These early learning steps have been superseded by the active A* substation workflow above.

## Milestone 2: MAVSDK Offboard Square Flight

### What Was Added
- Added `scripts/fly_square_offboard.py`.
- The new script connects to PX4 SITL through MAVSDK, waits for connection and healthy global/home position, arms, takes off, starts Offboard mode, flies a simple NED velocity square, stops Offboard mode, and lands.
- Telemetry is logged at about 5 Hz to a new timestamped CSV file:
  ```text
  data/logs/square_flight_YYYYMMDD_HHMMSS.csv
  ```
- The square flight CSV includes timestamp, elapsed time, flight phase, GPS position, altitude, NED velocity, roll, pitch, yaw, battery percent, flight mode, and armed state.
- Added `scripts/analyze_square_log.py`.
- The analysis script reads a square flight CSV, creates `outputs/<log_file_stem>/`, and saves matplotlib PNG plots for altitude, north/east velocity, yaw, approximate 2D trajectory, and battery percent when available.

### How to Run PX4 First
Terminal A:
```bash
cd ~/PX4-Autopilot
source .venv/bin/activate
export OpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4"
make px4_sitl gz_x500
```

Wait until PX4 SITL and Gazebo are running before starting the Python script.

### How to Run the Square Flight Script
Terminal B or PyCharm:
```bash
cd ~/projects/drone-ai
source .venv/bin/activate
python scripts/fly_square_offboard.py
```

Optional custom MAVSDK system address:
```bash
python scripts/fly_square_offboard.py --system-address udpin://0.0.0.0:14540
```

### How to Run the Analysis Script
Analyze the newest square flight log:
```bash
python scripts/analyze_square_log.py
```

Analyze a specific square flight log:
```bash
python scripts/analyze_square_log.py --log data/logs/square_flight_YYYYMMDD_HHMMSS.csv
```

## Milestone 3: Closed-Loop Local Waypoint Flight

### What Was Added
- Added `scripts/fly_waypoints_local.py`.
- The new script connects to PX4 SITL through MAVSDK, waits for connection and healthy global/home position, arms, takes off, starts Offboard mode, and flies through local NED waypoints using telemetry feedback.
- The waypoint controller reads `position_velocity_ned()` telemetry, computes local position error to each target, sends proportional `VelocityNedYaw` commands, and caps horizontal speed at 1.0 m/s and vertical speed at 0.5 m/s.
- Waypoints are considered reached when horizontal error is below 0.4 m and vertical error is below 0.4 m.
- Telemetry is logged at about 5 Hz to:
  ```text
  data/logs/waypoints_YYYYMMDD_HHMMSS.csv
  ```
- Added `scripts/analyze_waypoints_log.py`.
- The waypoint analysis script reads a waypoint CSV, creates `outputs/wp_YYYYMMDD_HHMMSS/`, saves short-named plots, and writes `summary.md` and `manifest.json`.
- Waypoint analysis cleanup keeps only the 10 newest `outputs/wp_YYYYMMDD_HHMMSS/` folders. It does not delete logs, square-flight folders, or legacy loose files in `outputs/`.

### How to Run PX4 First
Terminal A:
```bash
cd ~/PX4-Autopilot
source .venv/bin/activate
export OpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4"
make px4_sitl gz_x500
```

Wait until PX4 SITL and Gazebo are running before starting the Python script.

### How to Run the Waypoint Flight Script
Terminal B or PyCharm:
```bash
cd ~/projects/drone-ai
source .venv/bin/activate
python scripts/fly_waypoints_local.py
```

Optional custom MAVSDK system address:
```bash
python scripts/fly_waypoints_local.py --system-address udpin://0.0.0.0:14540
```

### How to Analyze the Waypoint Flight
Analyze the newest waypoint log:
```bash
python scripts/analyze_waypoints_log.py
```

Analyze a specific waypoint log:
```bash
python scripts/analyze_waypoints_log.py --log data/logs/waypoints_YYYYMMDD_HHMMSS.csv
```

## Milestone 4: A* Path Planning Flight

### What Was Added
- Added `planners/astar_grid.py`.
- The planner builds a simple 2D grid map, marks obstacle cells, runs A* from a start cell to a goal cell, simplifies the path, and converts grid cells into local NED waypoints.
- The planner remains 2D A*, with a simple 2.5D height-aware obstacle filter before cells are marked blocked.
- Added `scripts/fly_astar_path.py`.
- The A* flight script can run in dry-run preview mode or fly the generated waypoint path in PX4 SITL using the same closed-loop local NED velocity-feedback control style as the waypoint milestone.
- Added `scripts/analyze_astar_log.py`.
- The A* analysis script reads `data/logs/astar_YYYYMMDD_HHMMSS.csv`, creates `outputs/as_YYYYMMDD_HHMMSS/`, saves short-named plots, and writes `summary.md` and `manifest.json`.
- A* analysis no longer deletes old outputs; logs and generated output folders are preserved.
- Added Substation World v3 in `simulation/worlds/substation_simple.sdf`.
- Substation World v3 improves visual recognizability with simplified transformers, cabinets, gantry poles, busbars, insulators, pads, and service corridors.
- Substation World v3 fixes ground alignment by using center-z placement consistently for floor visuals, walls, devices, and poles.
- The substation world is still simplified and intended for planning/control experiments, not photorealistic scene rendering.
- A* grid-to-local conversion now uses the cell-center rule: east=`(grid_x + 0.5) * resolution_m`, north=`(grid_y + 0.5) * resolution_m`.
- `scripts/fly_astar_path.py` validates obstacle configs before previewing or flying and prints the coordinate convention.
- `scripts/analyze_astar_log.py` can overlay obstacle cells, split outbound/return trajectories, add a target timeline, summarize waypoint transitions, and check whether logged positions entered obstacle cells.
- A* analysis now uses `traj.png` as a clean trajectory overview and `traj_with_obstacles.png` as the obstacle validation plot when an obstacle overlay is available.
- `scripts/summarize_experiments.py` collects multiple `outputs/as_*/summary.md` and `manifest.json` analysis results into one benchmark table.
- Experiment summaries are written to `outputs/experiment_summary.csv` and `outputs/experiment_summary.md`.
- Ground/map visual alignment was improved by hiding the default Gazebo ground visual, keeping the physics ground collision, and using a custom 20 m x 20 m visible `substation_floor` centered at x=10, y=10.
- Added a visual-only 1 m floor grid plus origin/east-axis/north-axis markers to make the A* map frame easier to verify in Gazebo.
- The substation obstacle layout was redesigned after the inflated map over-blocked the internal area and encouraged boundary-hugging routes.
- Transformers and cabinets now use compact raw footprints, poles are represented as individual one-cell footprints, and a compact `switchgear_south` object creates an internal inspection-corridor entrance.
- Low boundary walls remain in the Gazebo world for visual context but are no longer planner obstacles at the default 1.5 m flight altitude.
- A* flight defaults now use a slower 0.8 m/s horizontal speed, with `--max-speed`, `--waypoint-acceptance`, `--turn-settle`, and `--return-speed-scale` available for safer testing.
- A* analysis now writes `collision_points.csv` and `collision_zoom.png` and distinguishes hard obstacle-cell entries from near-boundary/coarse-grid clearance warnings.
- `config/substation_obstacles.json` now includes `z_min_m` and `z_max_m` for each obstacle, plus default `altitude_m: 1.5`, `vertical_safety_margin_m: 0.3`, and `horizontal_inflation_cells: 1`.
- Obstacle config footprints are raw physical cells matching Gazebo objects; A* uses a separate inflated blocking-cell layer generated in code.
- Each obstacle now includes a `visual_category` such as `transformer`, `cabinet`, or `pole`, plus config notes explaining raw footprints versus inflation.
- Height-aware planning treats an obstacle as blocking only when `flight_altitude_m + vertical_safety_margin_m >= z_min_m` and `flight_altitude_m - vertical_safety_margin_m <= z_max_m`.
- Blocking obstacle footprints are inflated horizontally by the configured number of grid cells, clipped to the map boundary, while start and goal cells are kept free with a warning if needed.
- Dry-run preview now draws raw physical footprints in light gray and inflated A* planning obstacles in darker hatched gray, making the 2D preview easier to compare with Gazebo.
- Dry-run validation prints map-quality warnings for over-occupied inflated maps, boundary-heavy routes, missing internal corridor usage, goal adjacency, and true clearance violations.
- A* analysis uses the altitude recorded in the log when available, otherwise the obstacle config altitude, and reports blocking versus ignored obstacles.
- A* analysis distinguishes `raw_physical_collision_detected` from `inflated_safety_buffer_entry_detected`; raw footprint entries are more serious, while inflated entries are safety-buffer violations.

### How A* Works
- The grid is a simple artificial 2D map for learning.
- Each grid cell is either free or blocked by an obstacle.
- Obstacle blocking is selected by the 2.5D height filter first; A* still searches only in the 2D grid.
- Gazebo 3D objects correspond to raw physical footprint cells.
- A* plans against inflated blocking cells after height filtering and horizontal safety inflation.
- The goal stays fixed at `[16, 16]`; map layout and safety representation are adjusted around it rather than moving it.
- A* searches from the start cell to the goal cell using a priority queue.
- For non-diagonal planning, the heuristic is Manhattan distance.
- If diagonal planning is enabled with `--allow-diagonal`, the heuristic uses octile-style distance.
- The resulting path is simplified by removing collinear intermediate cells.

### Grid Cells to Local NED Waypoints
- Grid `x` becomes local `east_m` using the cell center.
- Grid `y` becomes local `north_m` using the cell center.
- Flight altitude is represented as negative local NED down:
  ```text
  down_m = -altitude_m
  ```
- With the default 1 meter resolution, grid cell `(8, 8)` becomes approximately:
  ```text
  north_m = 8.5
  east_m = 8.5
  down_m = -1.5
  ```

### Dry-Run Preview
Preview only:
```bash
cd ~/projects/drone-ai
source .venv/bin/activate
python scripts/fly_astar_path.py --dry-run
```

Substation preview with obstacle config and return-home route:
```bash
python scripts/fly_astar_path.py --dry-run --obstacle-config config/substation_obstacles.json --return-home --altitude 1.5
```

Dry-run output:
```text
outputs/as_preview/grid_path.png
outputs/as_preview/path_preview.json
```

### How to Run PX4 First
Terminal A, run PX4:
```bash
cd ~/PX4-Autopilot
source .venv/bin/activate
export OpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4"
make px4_sitl gz_x500
```

Wait until PX4 SITL and Gazebo are running before starting the Python script.

### How to Run the A* Flight
Terminal B or PyCharm, run A* flight:
```bash
cd ~/projects/drone-ai
source .venv/bin/activate
python scripts/fly_astar_path.py
```

Optional flags:
```bash
python scripts/fly_astar_path.py --allow-diagonal
python scripts/fly_astar_path.py --resolution 1.0 --altitude 1.5
python scripts/fly_astar_path.py --system-address udpin://0.0.0.0:14540
python scripts/fly_astar_path.py --max-speed 0.8 --return-speed-scale 0.7 --waypoint-acceptance 0.3
```

### Return-to-Start Behavior
- `--return-home` makes the drone fly the reversed A* waypoint path back to the start before landing.
- This is preferred over PX4 RTL for obstacle-aware testing because PX4 RTL does not know the artificial A* obstacle map.

Preview with return:
```bash
cd ~/projects/drone-ai
source .venv/bin/activate
python scripts/fly_astar_path.py --dry-run --obstacle-config config/substation_obstacles.json --return-home --altitude 1.5
```

Fly with return:
```bash
python scripts/fly_astar_path.py --obstacle-config config/substation_obstacles.json --return-home --altitude 1.5 --max-speed 0.8 --return-speed-scale 0.7 --waypoint-acceptance 0.3
```

Analyze:
```bash
python scripts/analyze_astar_log.py --obstacle-config config/substation_obstacles.json
```

Full debug analysis:
```bash
python scripts/analyze_astar_log.py --obstacle-config config/substation_obstacles.json --debug-plots
```

### How to Analyze the A* Flight
Analyze the newest A* log:
```bash
python scripts/analyze_astar_log.py --obstacle-config config/substation_obstacles.json
```

Analyze a specific A* log:
```bash
python scripts/analyze_astar_log.py --log data/logs/astar_YYYYMMDD_HHMMSS.csv --obstacle-config config/substation_obstacles.json
```

Trajectory plot outputs:
- `traj.png`: clean route-following overview without obstacle or collision overlays.
- `traj_with_obstacles.png`: obstacle validation view with raw footprints, inflated planning cells, and collision/buffer entry points when available.

Default analysis is for normal experiment review and generates the core report files only: `summary.md`, `manifest.json`, `traj.png`, `traj_with_obstacles.png` when an obstacle config is available, `error.png`, and `perception_risk_timeline.png` when perception columns exist. Use `--debug-plots` when diagnosing altitude, velocity, yaw, waypoint switching, nearest obstacle distance, and perception count details. Collision debug files, `collision_points.csv` and `collision_zoom.png`, are generated only when a raw physical collision or inflated safety-buffer entry exists.

Summarize all analyzed A* experiment runs:
```bash
python scripts/summarize_experiments.py
```

Summary outputs:
```text
outputs/experiment_summary.csv
outputs/experiment_summary.md
```

The experiment summary includes normalized perception metrics so runs with different durations can be compared fairly, especially `log_only` versus `slow_down`. It compares warning/danger sample ratios, percent time in warning/danger, obstacle distance statistics, duration, and collision/buffer flags grouped by `risk_action`.

## Perception Stage 1: Simulated Local Obstacle Detection

This is the first perception/AI step, but it is not real camera, LiDAR, depth, or YOLO detection yet.

The new `perception/simple_obstacle_detector.py` module uses the known 2.5D obstacle map to emulate a simple onboard forward obstacle detector. It converts raw physical obstacle cells and inflated planning cells into local cell-center coordinates, checks which cells are within range, and filters them by the drone yaw and field of view when yaw telemetry is available.

This stage is useful for validating the perception logging and analysis pipeline before adding real Gazebo sensor topics. It does not control the drone, replan the route, or stop the vehicle.

Test perception without PX4:
```bash
python scripts/test_simple_perception.py --obstacle-config config/substation_obstacles.json
```

Fly with perception logging:
```bash
python scripts/fly_astar_path.py \
  --obstacle-config config/substation_obstacles.json \
  --return-home \
  --altitude 1.5 \
  --max-speed 0.8 \
  --return-speed-scale 0.7 \
  --waypoint-acceptance 0.3 \
  --enable-perception \
  --detection-range 4.0 \
  --detection-fov 90
```

Analyze perception outputs:
```bash
python scripts/analyze_astar_log.py --obstacle-config config/substation_obstacles.json
```

Perception analysis adds `perception_risk_timeline.png` and normalized perception metrics in `summary.md` when perception columns are present in the log. Re-run with `--debug-plots` to also create `perception_timeline.png` and `detection_count_over_time.png`. `traj_with_obstacles.png` also marks samples where the simulated detector reported an obstacle.

## Perception Stage 1B: Risk-Level Obstacle Detection

The detector still uses the known obstacle map; this is not real LiDAR or camera perception yet. Detection is now classified into four risk levels:

- `clear`: no obstacle detected within the configured detection range.
- `detected`: obstacle is within sensor range but farther than the warning threshold.
- `warning`: nearest obstacle is within the warning threshold but farther than the danger threshold.
- `danger`: nearest obstacle is within the danger threshold.

Default thresholds:
- detection range: 4.0 m
- warning distance: 2.0 m
- danger distance: 1.0 m

The default risk action is `log_only`, so flight behavior does not change unless requested. Prototype options can slow down on warning/danger or stop and land on danger, but this is still not full local replanning.

Waypoint timeout now defaults to `--waypoint-timeout auto`. This was added after a slow_down test timed out before WP05: perception slowdown reduced the commanded horizontal speed, but the previous fixed per-waypoint timeout did not adapt. Auto timeout estimates each leg from the current horizontal distance, max speed, return speed scale, and risk action, then uses a conservative margin. Slow_down mode also has a minimum commanded speed floor with `--min-risk-speed` so warning/danger commands do not become too small to make waypoint progress.

Test perception:
```bash
python scripts/test_simple_perception.py \
  --obstacle-config config/substation_obstacles.json \
  --detection-range 4.0 \
  --detection-fov 90 \
  --warning-distance 2.0 \
  --danger-distance 1.0
```

Fly with perception risk logging:
```bash
python scripts/fly_astar_path.py \
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

Fly with cautious slowdown:
```bash
python scripts/fly_astar_path.py \
  --obstacle-config config/substation_obstacles.json \
  --return-home \
  --altitude 1.5 \
  --max-speed 1.0 \
  --return-speed-scale 0.8 \
  --waypoint-acceptance 0.4 \
  --enable-perception \
  --detection-range 4.0 \
  --detection-fov 90 \
  --warning-distance 1.5 \
  --danger-distance 0.8 \
  --risk-action slow_down \
  --waypoint-timeout auto \
  --min-risk-speed 0.3
```

Analyze:
```bash
python scripts/analyze_astar_log.py --obstacle-config config/substation_obstacles.json
```

Full debug analysis:
```bash
python scripts/analyze_astar_log.py --obstacle-config config/substation_obstacles.json --debug-plots
```

Summarize analyzed experiments:
```bash
python scripts/summarize_experiments.py
```

## Future Perception Stage 2

- Add a Gazebo LiDAR or depth camera sensor to the drone model.
- Discover Gazebo sensor topics with `gz topic -l`.
- Subscribe to real sensor data.
- Replace `SimpleObstacleDetector` with real sensor-based obstacle detection.
- Use detections for local replanning or emergency stop.

Convenience wrapper commands:
```bash
python main.py astar preview --obstacle-config config/substation_obstacles.json --return-home
python main.py astar fly --obstacle-config config/substation_obstacles.json --return-home
python main.py astar analyze --obstacle-config config/substation_obstacles.json
python main.py astar run --obstacle-config config/substation_obstacles.json --return-home
```

## Milestone 5: Substation World v2

### What Was Added
- Added `simulation/worlds/substation_simple.sdf`.
- Added `config/substation_obstacles.json`.
- Added `planners/obstacle_config.py`.
- Updated `scripts/fly_astar_path.py` with:
  ```bash
  --obstacle-config config/substation_obstacles.json
  ```
- Added `simulation/README.md` with manual PX4/Gazebo world usage instructions.

### Why Simple Geometry First
- The substation world is intentionally simple and not photorealistic.
- Version 2 uses two transformer blocks, two cabinet blocks, paired pole rows, high visual-only bus wires, a ground plane, boundary walls, and clear start/goal markers.
- The southwest takeoff area near east=0, north=0 is kept clear of obstacle collisions.
- Simple geometry makes it easier to connect Gazebo objects to a hand-written A* obstacle grid before adding complex assets, YOLO, ROS, or computer vision.

### Shared Obstacle Map
- `config/substation_obstacles.json` defines the planner map for Substation World v2.
- Gazebo `x` and A* grid `x` correspond to local `east_m`.
- Gazebo `y` and A* grid `y` correspond to local `north_m`.
- Rectangular obstacle entries match the major transformer, cabinet, pole-row, and fence keepout areas in `substation_simple.sdf`.
- The config keeps `start_cell: [0, 0]`, `goal_cell: [16, 16]`, `resolution_m: 1.0`, and `altitude_m: 1.5`.
- Obstacles include `z_min_m` and `z_max_m`; low walls below the default flight band are ignored while taller devices and poles block the 2D grid.

### Preview A* Path on the Substation Map
Preview A* only:
```bash
cd ~/projects/drone-ai
source .venv/bin/activate
python scripts/fly_astar_path.py --dry-run --obstacle-config config/substation_obstacles.json
```

Preview output:
```text
outputs/as_preview/grid_path.png
outputs/as_preview/path_preview.json
```

### Manually Copy the World into PX4
Do not copy this automatically. If you want PX4 to load the custom world, copy it manually:
```bash
cp ~/projects/drone-ai/simulation/worlds/substation_simple.sdf ~/PX4-Autopilot/Tools/simulation/gz/worlds/
```

If a Gazebo simulation is already running, stop it first. `PX4_GZ_WORLD` can be ignored when an existing simulation is already active.

### Run PX4 with the Custom World
```bash
pkill -f "px4"
pkill -f "gz sim"
pkill -f "gz"

cd ~/PX4-Autopilot
source .venv/bin/activate
export OpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4"
PX4_GZ_WORLD=substation_simple make px4_sitl gz_x500
```

### Fly A* Path with the Same Obstacle Config
```bash
cd ~/projects/drone-ai
source .venv/bin/activate
python scripts/fly_astar_path.py --obstacle-config config/substation_obstacles.json
```

### Analyze
```bash
python scripts/analyze_astar_log.py --obstacle-config config/substation_obstacles.json
```
