"""Materialize active inspection plan, events, preview, and offline report."""

import json
from pathlib import Path
from typing import Any

from src.planner.active_inspection import evaluate_plan, load_jsonl, run_active_inspection
from src.study.active_inspection_benchmark import acceptance, compare_schedulers


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _svg(plan, map_value) -> str:
    width, height, scale = int(map_value.get("width_cells", 1)), int(map_value.get("height_cells", 1)), 12
    lines = []
    for decision in plan["decisions"]:
        color = "#e4572e" if decision["kind"] == "equipment_observation" else "#2e86ab"
        points = " ".join(f"{c[0]*scale+6},{(height-c[1]-1)*scale+6}" for c in decision["transit_cells"])
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width*scale} {height*scale}"><rect width="100%" height="100%" fill="#f6f1e7"/>{"".join(lines)}</svg>\n'


def materialize_active_inspection(map_path, observations_path, output_path, *, policy_path):
    map_path, observations_path, policy_path, output_path = map(Path, (map_path, observations_path, policy_path, output_path))
    map_value, policy = _json(map_path), _json(policy_path)
    plan = run_active_inspection(map_value, policy, load_jsonl(observations_path))
    report = evaluate_plan(plan, map_value)
    report["scheduler_comparison"] = compare_schedulers(plan, map_value)
    report["acceptance"] = acceptance(report, report["scheduler_comparison"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events, preview, report_path = output_path.with_suffix(".events.jsonl"), output_path.with_suffix(".preview.svg"), output_path.with_suffix(".report.json")
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    events.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in plan["events"]), encoding="utf-8")
    preview.write_text(_svg(plan, map_value), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"plan": str(output_path), "events": str(events), "preview": str(preview), "report": str(report_path)}
