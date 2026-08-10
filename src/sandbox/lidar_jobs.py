"""Allow-listed command resolution for LiDAR sandbox gates."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

from src.sandbox.workflow import artifact_reference


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _latest_replay(root):
    paths = sorted(
        root.glob("outputs/research/lidar_replay_gate_*/replay_gate.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not paths:
        raise ValueError("a completed LiDAR replay gate is required")
    value = _read_json(paths[0])
    if not value.get("passed"):
        raise ValueError("the latest LiDAR replay gate did not pass")
    value["_receipt_path"] = paths[0]
    return value


def _matching_path(root, pattern, key, expected):
    matches = []
    for path in root.glob(pattern):
        try:
            if _read_json(path).get(key) == expected:
                matches.append(path.parent)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    if len(matches) != 1:
        raise ValueError(f"expected one local artifact for {expected}")
    return matches[0]


def _study_id(root, model_id):
    registry = root / "outputs/research/registry.sqlite"
    if not registry.is_file():
        raise ValueError("the research study registry is not present")
    connection = sqlite3.connect(f"file:{registry.resolve()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            """
            SELECT s.study_id FROM studies s
            WHERE s.candidate_model=? AND EXISTS (
              SELECT 1 FROM runs r
              WHERE r.study_id=s.study_id AND r.tier='closed-loop'
                AND r.status!='completed'
            ) ORDER BY s.created_at DESC LIMIT 1
            """,
            (model_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError("the LiDAR closed-loop gate has no pending runs")
    return row[0]


def build_lidar_command(config, action):
    root = config.project_root
    replay = _latest_replay(root)
    model_id, dataset_id = replay["model_id"], replay["dataset_id"]
    if action == "lidar-replay-gate":
        package = _matching_path(
            root, "models/lidar/*/model_manifest.json", "model_id", model_id
        )
        dataset = _matching_path(
            root, "outputs/research/**/dataset_manifest.json", "dataset_id", dataset_id
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = root / "outputs/sandbox/lidar_replay" / stamp
        artifacts = (
            artifact_reference(
                root, "lidar_replay_receipt",
                replay["replay_gate_identity_sha256"],
                replay["_receipt_path"],
            ),
            artifact_reference(root, "lidar_model", model_id, package),
            artifact_reference(root, "lidar_dataset", dataset_id, dataset),
        )
        return (
            action,
            (
                sys.executable, "main.py", "model", "replay-gate",
                "--package", str(package), "--dataset", str(dataset),
                "--output", str(output),
            ),
            600.0,
            artifacts,
            ((output / "replay_gate.json").relative_to(root).as_posix(),),
            False,
        )
    if action == "lidar-closed-loop-next":
        study_id = _study_id(root, model_id)
        return (
            action,
            (
                sys.executable, "main.py", "study", "execute-closed-loop",
                study_id, "--max-runs", "1",
            ),
            540.0,
            (
                artifact_reference(
                    root, "lidar_replay_receipt",
                    replay["replay_gate_identity_sha256"],
                    replay["_receipt_path"],
                ),
                artifact_reference(
                    root, "closed_loop_study", study_id,
                    root / "outputs/research/registry.sqlite",
                ),
            ),
            (),
            True,
        )
    raise ValueError(f"unsupported LiDAR sandbox action: {action}")
