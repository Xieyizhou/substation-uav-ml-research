"""Create and execute a fresh capability challenge for one registered model."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.study.challenge_worker import execute_challenge
from src.study.registry import ResearchRegistry


def run_challenge_gate(project_root, model_id):
    root = Path(project_root).resolve()
    registry_path = root / "outputs/research/registry.sqlite"
    results = root / "outputs/research/study_results"
    registry = ResearchRegistry(registry_path)
    registry.get_model(model_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study_id = registry.create_study(
        f"sandbox-capability-challenge-{stamp}", model_id
    )
    result = execute_challenge(registry_path, study_id, results)
    return {"study_id": study_id, **result}
