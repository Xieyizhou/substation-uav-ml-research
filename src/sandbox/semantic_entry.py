"""Run a selected verified model in the bounded, fixed-scene desktop SITL flow."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from src.inspection.config import InspectionConfig
from src.ml.artifacts import file_sha256
from src.sandbox.semantic_qualification import (INTENTS, current_identity, model_directory,
    qualification, run_directory, write_qualification)
from src.sandbox.sitl_assets import install_assets
from src.sandbox.live_replan_stop import check_stop
from src.vision.canonical.plan import write_record


def freeze(config, out, model_id, intent, expected_identity):
    from src.flight.sitl_support import collision_boxes, inside
    from src.flight.tracking_envelope import check_route
    assets = install_assets(config.project_root)
    current = current_identity(config, model_id)
    if current['identity_sha256'] != expected_identity:
        raise ValueError('Current runtime changed after launch request; revalidate')
    if intent == 'mission' and not qualification(config, expected_identity)['qualified']:
        raise ValueError('Current model/runtime needs positive and controlled-stop qualification')
    route = [(1.5,1.5), (3.,1.5)]
    boxes = collision_boxes(assets/'world.sdf')
    if not check_route(route, boxes, map_uncertainty=.2)['passed']:
        raise ValueError('Initial tracking envelope failed')
    for a,b in zip(route,route[1:]):
        for i in range(101):
            point = tuple(x+(y-x)*i/100 for x,y in zip(a,b))
            if any(inside(point, box, 1.8) for box in boxes):
                raise ValueError('Initial static route blocked')
    # A stop can arrive while model dependencies are warming. Never erase it.
    check_stop(out)
    out.mkdir(parents=True, exist_ok=False)
    write_record(out/'protocol.json', dict(schema_version=1, intent=intent,
        simulation_only=True, runtime_identity_sha256=expected_identity, runtime_identity=current['identity'],
        model_id=model_id, goal=route[-1], initial_route=route, static_boxes=boxes,
        history_scope='Frozen scenario provenance only; current qualification is independent',
        waypoint_tolerance_m=.03, stop_speed_m_s=.08, stop_stable_s=1., planning_clearance_m=1.8,
        lidar_emergency_stop_m=1.5, max_replans=1, route_timeout_s=180,
        inputs={row['path']: row['sha256'] for row in current['inputs'].values()}))
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--intent', choices=sorted(INTENTS), required=True)
    parser.add_argument('--identity', required=True)
    args = parser.parse_args()
    config = InspectionConfig.for_profile(args.project_root, profile='development')
    out = run_directory(config.project_root, args.run_id)
    assets = freeze(config, out, args.model_id, args.intent, args.identity)
    model_root = model_directory(config, args.model_id)
    from src.sandbox.visual_runtime_model import visual_runtime_model
    info = visual_runtime_model(model_root)
    from ultralytics import YOLO
    import numpy as np
    from scripts.vision.locked_cpu_threads import locked_threads
    # Finish lazy dependency initialization before MAVSDK/gRPC starts threads.
    with locked_threads(4):
        model = YOLO(str(info['model_path']), task='detect')
        model.predict(source=np.zeros((info['imgsz'],info['imgsz'],3),np.uint8),
                      imgsz=info['imgsz'], rect=False, device='cpu', verbose=False)
        del model
        check_stop(out)
        from src.flight.semantic_mission import run
        asyncio.run(run(config.project_root, config.px4_root, assets, out, model_root))
    # Actual input bytes must still match the preflight identity after execution.
    protocol = json.loads((out/'protocol.json').read_text())
    for path, digest in protocol['inputs'].items():
        if file_sha256(path) != digest:
            raise ValueError('Runtime input changed during flight: '+path)
    runtime = json.loads((out/'runtime/receipt.json').read_text())
    if args.intent == 'qualify-stop':
        write_qualification(out, 'controlled_stop')
    else:
        if runtime['status'] != 'sitl_hover_landed_disarmed':
            raise RuntimeError(runtime['error'] or 'Runtime failed')
        from src.flight.semantic_audit import audit
        audit(out, assets)
        if args.intent == 'qualify-positive':
            write_qualification(out, 'positive')
    print('CURRENT_SITL_TASK_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
