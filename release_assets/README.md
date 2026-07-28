# Optional Local Media

This directory is for local release-media assembly only.

This directory retains optional media tooling inherited from the predecessor
demo. It is not part of the research evidence pipeline. Generated videos such
as `uav_path_planning_demo_preview.mp4` are ignored by Git.

Optional source media for the reusable demo assembler can be placed under `release_assets/raw/`:

- `gazebo_flight_6x.mp4`
- `results_summary.png`
- `github_home.png`

These raw local assets are ignored by git. The generated output is:

```text
release_assets/uav_path_planning_demo_preview.mp4
```

`gazebo_flight_6x.mp4` should be about 14 seconds. Run:

```bash
python scripts/media/make_demo_video.py
```
