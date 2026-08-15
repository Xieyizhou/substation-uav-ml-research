# Configuration

`substation_obstacles.json` defines the grid map used by the A* planner, the
simulated perception detector, and local replan experiments.

Important conventions:

- Grid `x` maps to local east.
- Grid `y` maps to local north.
- Cell centers convert to local NED waypoints as `(x + 0.5, y + 0.5)`.
- `gazebo_world_origin_m` places the southwest map corner in Gazebo world
  coordinates. The substation uses `[-10, -10, 0]`, centering its 20 x 20 m
  floor on the Gazebo origin while PX4 keeps the same local NED waypoints.
- Raw obstacle rectangles represent physical footprints.
- Height-aware filtering decides which raw obstacles block a given flight altitude.
- Horizontal inflation expands blocking cells into planning keepout cells.

JSON does not support comments, so durable explanation belongs here and in
`docs/EXPERIMENT_PROTOCOL.md`.

`perception/domain_randomization.json` defines deterministic v0.2 scene and
sensor ranges. `python main.py data world` applies equipment pose/scale,
lighting, and unknown-obstacle changes to a generated SDF; its adjacent
scenario JSON drives scan noise, point dropout, stream outages, and
attitude-label jitter. Source worlds are never edited in place.

`perception/research_protocol.json` uses 30 reserved paired scenarios and four
conditions, producing 120 formal runs. Seeds `1001–1030` are rejected by the
dataset collector so formal evidence cannot leak into training.

Machine-readable dataset, model, and study-result contracts live in
`config/schemas/`.

`sandbox/runtime_compatibility.json` is the version-controlled compatibility
policy for the macOS App and local environment check. It records the minimum
and tested Python versions, required Development imports, tested PX4 commit
identities, expected Gazebo Sim major/distribution, and the OpenCV/Qt families.
It does not install or pin external runtimes. Machine-specific verified paths
are stored in the user's Application Support directory and are never tracked.

The coordinated map catalog in `config/maps/catalog.json` also defines five
safe A* destination presets per map. The selected target is stored under
`.runtime/selected_targets.json`; obstacle config files remain deterministic and
continue to keep their original `goal_cell` as the default `top_right` target.
