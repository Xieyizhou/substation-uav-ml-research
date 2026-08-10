"""Read-only LiDAR replay and closed-loop gate inspection."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sqlite3

from src.study.comparison import closed_loop_gate


def _relative(root, path):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return None


def _latest_replay(root):
    roots = (
        root.glob("outputs/research/lidar_replay_gate_*/replay_gate.json"),
        root.glob("outputs/sandbox/lidar_replay/*/replay_gate.json"),
    )
    paths = sorted(
        (path for group in roots for path in group),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not paths:
        return {"status": "missing", "passed": False}
    path = paths[0]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        metrics = value["gate_metrics"]
        return {
            "status": "complete" if value.get("passed") else "failed",
            "passed": bool(value.get("passed")),
            "model_id": value.get("model_id"),
            "dataset_id": value.get("dataset_id"),
            "macro_f1": metrics.get("risk_f1"),
            "danger_recall": metrics.get("danger_recall"),
            "inference_p95_ms": metrics.get("inference_p95_ms"),
            "traversability_iou": metrics.get("traversability_iou"),
            "identity": value.get("replay_gate_identity_sha256"),
            "path": _relative(root, path),
        }
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        return {"status": "invalid", "passed": False, "error": str(error)}


def _read_study(root, model_id):
    path = root / "outputs/research/registry.sqlite"
    if not path.is_file():
        return None, []
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        query = """
          SELECT s.* FROM studies s
          WHERE EXISTS (
            SELECT 1 FROM runs r
            WHERE r.study_id=s.study_id AND r.tier='closed-loop'
          )
        """
        values = []
        if model_id:
            query += " AND s.candidate_model=?"
            values.append(model_id)
        query += " ORDER BY s.created_at DESC LIMIT 1"
        study = connection.execute(query, values).fetchone()
        if study is None:
            return None, []
        rows = connection.execute(
            """
            SELECT r.*, m.name, m.value FROM runs r
            LEFT JOIN metrics m ON r.run_id=m.run_id
            WHERE r.study_id=? AND r.tier='closed-loop'
            ORDER BY r.scenario_id, r.condition
            """,
            (study["study_id"],),
        )
        runs = {}
        for row in rows:
            item = runs.setdefault(row["run_id"], dict(row))
            item.setdefault("metrics", {})
            if row["name"] is not None:
                item["metrics"][row["name"]] = row["value"]
        return dict(study), list(runs.values())
    finally:
        connection.close()


def _sum_metric(runs, name):
    return sum(float(run["metrics"].get(name, 0.0)) for run in runs)


def _closed_loop(root, model_id):
    try:
        study, runs = _read_study(root, model_id)
    except sqlite3.Error as error:
        return {"status": "invalid", "passed": False, "error": str(error)}
    if study is None:
        return {"status": "missing", "passed": False, "completed": 0, "total": 0}
    counts = Counter(run["status"] for run in runs)
    completed = counts.get("completed", 0)
    health = [run["metrics"].get("sensor_healthy_ratio") for run in runs]
    latency = [run["metrics"].get("inference_p95_ms") for run in runs]
    health = [value for value in health if value is not None]
    latency = [value for value in latency if value is not None]
    gate = closed_loop_gate(runs)
    return {
        "status": "complete" if completed == len(runs) and runs else "in_progress",
        "passed": gate["passed"],
        "reasons": gate["reasons"],
        "study_id": study["study_id"],
        "model_id": study["candidate_model"],
        "completed": completed,
        "total": len(runs),
        "pending": len(runs) - completed,
        "states": dict(sorted(counts.items())),
        "scenario_count": len({run["scenario_id"] for run in runs}),
        "conditions": sorted({run["condition"] for run in runs}),
        "mission_success": int(_sum_metric(runs, "mission_success")),
        "landing_success": int(_sum_metric(runs, "landing_success")),
        "collision_count": int(_sum_metric(runs, "collision_count")),
        "buffer_entry_count": int(_sum_metric(runs, "buffer_entry_count")),
        "safety_failure_count": int(_sum_metric(runs, "safety_failure_count")),
        "sensor_health_min": min(health) if health else None,
        "inference_p95_max_ms": max(latency) if latency else None,
        "path": _relative(root, root / "outputs/research/study_results" / study["study_id"]),
    }


def lidar_summary(config):
    """Return the latest replay and matching closed-loop candidate state."""
    root = config.project_root
    replay = _latest_replay(root)
    closed = _closed_loop(root, replay.get("model_id"))
    return {
        "status": "complete" if replay["passed"] and closed["passed"] else "incomplete",
        "replay": replay,
        "closed_loop": closed,
    }
