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

The coordinated map catalog in `config/maps/catalog.json` also defines five
safe A* destination presets per map. The selected target is stored under
`.runtime/selected_targets.json`; obstacle config files remain deterministic and
continue to keep their original `goal_cell` as the default `top_right` target.
