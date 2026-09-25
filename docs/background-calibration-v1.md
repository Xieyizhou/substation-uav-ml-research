# Isolated background calibration v1

These three development-only scenes do not change the canonical Simple,
Medium or Complex worlds, the map catalog, or the selected map.

- `bg_fence_yard_v1`, seed `790101`: metal pickets, rails and patterned paving.
- `bg_masonry_lane_v1`, seed `790102`: brick-panel walls and mortar contrast.
- `bg_pipe_rack_v1`, seed `790103`: structural uprights and open pipe racks.

No transformer, switchgear, capacitor-bank or reactor models are included.
This inventory is NOT a substitute for synchronized sensor truth or visual
review. All obstacles have conservative collision volumes and matching planning
footprints. Routes use the existing inflated-grid A* and preserve 1.5 m altitude.
Materials are generated in Gazebo, not applied as image augmentation.

## Build and launch

```sh
.venv/bin/python scripts/vision/build_background_calibration_scenes.py
.venv/bin/python scripts/vision/start_background_calibration_scene.py bg_fence_yard_v1
```

The builder refuses to overwrite an existing package. The launcher verifies
package/file identities, uses the existing custom-world launcher with
`x500_research`, and does not select a canonical map or arm a flight.
Stop any existing simulator explicitly before starting one of these scenes.
Use only owned-process cleanup, never global `pkill`.

The transport partition is `substation_background_calibration`; every future
probe, truth recorder and RGB-D collector for this world must use the same
partition and `GZ_IP=127.0.0.1`.

## Evidence boundaries and next step

Generated assets are not yet live-validated. Before collection, perform the
unchanged native 100-pair pre-arm probe and review rendered scene content.
The dedicated adapter below uses the existing collector implementation with the
actual scene ID and development seed. It requires real empty pre-arm truth and
the unchanged RGB-D gate. Its run receipt binds scene, route, package and raw
collection identities. The raw collector's legacy protocol field remains
explicitly recorded for provenance. Do not pass fake canonical map IDs or
manufacture empty truth records.

```sh
.venv/bin/python scripts/vision/run_background_calibration_flight.py \
  --scene-id bg_fence_yard_v1 \
  --output data/research/background_calibration_v1/runs/fence-pilot-v1
```

This adapter is newly implemented and has not yet passed a live pilot. It runs
one scene only; it does not automatically train or launch another scene.

Initial collection is capped at 60 frames per scene, emit stride 30, startup
wait 140 seconds. Require synchronized empty equipment truth BEFORE bbox
filtering, visual review for accidentally device-like structures, safe landing,
and exact/near deduplication against all retained data. A scene needs at least
20 new accepted background frames before expansion. The full training quota
remains 600 background frames. Different seeds do not prove scene independence.

No new samples count toward coverage merely because these assets exist.
No training, formal qualification or model promotion is authorized by this
package. Keep real-domain v3 and MapDraft deferred.
