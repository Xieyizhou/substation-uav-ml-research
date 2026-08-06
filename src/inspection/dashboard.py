"""Collection dashboard adapter over the established status query service."""

from __future__ import annotations

from collections import Counter
import json

from src.inspection.config import InspectionConfig
from src.inspection.models import DashboardView, ScenarioView
from src.vision.collection.plan import collection_status


BLIND_ROLES = frozenset({"blind", "held_out_test", "test"})


def load_plan(config: InspectionConfig) -> dict:
    try:
        plan = json.loads(config.plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load collection plan: {error}") from error
    if not isinstance(plan.get("scenarios"), list):
        raise ValueError("collection plan has no scenario list")
    return plan


def is_blind(row: dict) -> bool:
    role = str(row.get("dataset_role", "")).lower()
    split = str(row.get("split", "")).lower()
    return role in BLIND_ROLES or split in BLIND_ROLES


def scenario_view(row: dict | None) -> ScenarioView | None:
    if row is None:
        return None
    blind = is_blind(row)
    return ScenarioView(
        scenario_id=str(row.get("scenario_id", "")),
        recording_id=str(row.get("recording_id", "")),
        state=str(row.get("state", "unknown")),
        dataset_role=str(row.get("dataset_role", row.get("split", "unknown"))),
        split=str(row.get("split", "unknown")),
        layout=None if blind else row.get("layout_id", row.get("map_id")),
        route=None if blind else row.get("route_id"),
        seed=None if blind else row.get("seed"),
        target_class=None if blind else row.get("target_class", row.get("target_id")),
    )


def dashboard(config: InspectionConfig) -> DashboardView:
    plan = load_plan(config)
    status = collection_status(plan, config.recordings_root)
    raw = status["recording_states"]
    counts = {
        "complete": raw.get("complete", 0),
        "recording": raw.get("recording", 0),
        "missing": raw.get("missing", 0),
        "partial": raw.get("partial", 0),
        "failed": raw.get("failed", 0),
        "unvalidated": raw.get("unvalidated", 0),
        "invalid": raw.get("invalid_receipt", 0),
    }
    rows = plan["scenarios"]
    role_counts = Counter(
        "blind" if is_blind(row) else str(row.get("dataset_role", row.get("split", "unknown")))
        for row in rows
    )
    current = _active_scenario(rows, config)
    complete = counts["complete"]
    total = len(rows)
    return DashboardView(
        counts, dict(role_counts), complete, total, total - complete,
        round(100.0 * complete / total, 1) if total else 0.0,
        scenario_view(current), scenario_view(status.get("next_scenario")),
    )


def _active_scenario(rows, config):
    for row in rows:
        if (config.recording(str(row["recording_id"])) / "live_status.json").is_file():
            return {**row, "state": "recording"}
    return None
