# Demo Video Assembly

Use `scripts/media/make_demo_video.py` to assemble a polished black minimalist 16:9 GitHub release demo video from existing local assets. The script does not run PX4, generate plots, modify logs, or change experiment behavior.

Local requirements:

- `ffmpeg`
- Python with `Pillow`

## Expected Assets

- `docs/assets/grid_path.png`: A* grid path preview image.
- `release_assets/raw/gazebo_flight_6x.mp4`: compact PX4/Gazebo flight recording.
- `release_assets/raw/results_summary.png`: result summary screenshot.
- `release_assets/raw/github_home.png`: GitHub repository or release screenshot.

`gazebo_flight_6x.mp4` is expected to be about 14 seconds. If it is longer than 14.5 seconds, the script trims it to 14 seconds. If it is shorter, the script uses the full clip and continues normally.

If any asset is missing, the script prints a warning and substitutes a black placeholder. If `results_summary.png` is missing, it generates a minimalist results card automatically.

## Run

```bash
python scripts/media/make_demo_video.py
```

The default output is:

```text
release_assets/uav_path_planning_demo_preview.mp4
```

The generated MP4 is a local release asset and should not be committed to git. Attach it manually to the GitHub `v0.2.0` release if it is useful.

## Video Structure

- 2.5 seconds: title card.
- 6 seconds: A* path preview.
- About 14 seconds: PX4/Gazebo execution.
- 7.5 seconds: curated results summary.
- 6 seconds: GitHub repository / release ending.
