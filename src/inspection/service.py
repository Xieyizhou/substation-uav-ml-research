"""Application facade used by the local HTTP presentation layer."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

from src.inspection.config import AccessDenied, InspectionConfig
from src.inspection.dashboard import dashboard, is_blind, load_plan
from src.inspection.doctor import run_doctor
from src.inspection.frames import frame_page, frame_path, scenario_progress
from src.inspection.logs import log_tail
from src.inspection.runtime import LocalProcessAdapter, runtime_status


def serialize(value):
    if is_dataclass(value):
        return {key: serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize(item) for item in value]
    return value


class InspectionService:
    def __init__(self, config: InspectionConfig, process_adapter=None, operator=None):
        self.config = config
        self.process_adapter = process_adapter or LocalProcessAdapter()
        self.operator = operator

    def doctor(self):
        return serialize(run_doctor(self.config))

    def dashboard(self):
        return serialize(dashboard(self.config))

    def runtime(self):
        return serialize(runtime_status(self.process_adapter))

    def recordings(self):
        results = []
        for row in load_plan(self.config)["scenarios"]:
            if is_blind(row):
                continue
            recording_id = str(row["recording_id"])
            if not self.config.recording(recording_id).exists():
                continue
            results.append({
                "recording_id": recording_id,
                "scenario_id": row.get("scenario_id"),
                "role": row.get("dataset_role", row.get("split")),
                "target_class": row.get("target_class", row.get("target_id")),
            })
        return results

    def scenarios(self):
        return [
            {
                "scenario_id": row["scenario_id"],
                "dataset_role": row.get("dataset_role", row.get("split")),
                "target_class": row.get("target_class", row.get("target_id")),
            }
            for row in load_plan(self.config)["scenarios"]
            if not is_blind(row)
        ]

    def logs(self, scenario_id, kind, limit):
        row = self._scenario(scenario_id)
        if is_blind(row):
            raise AccessDenied("blind scenario logs are not exposed")
        return serialize(log_tail(self.config, scenario_id, kind, limit))

    def frames(self, recording_id, page, page_size):
        return serialize(frame_page(self.config, recording_id, page, page_size))

    def progress(self, recording_id):
        return serialize(scenario_progress(self.config, recording_id))

    def frame_file(self, recording_id, frame_id):
        return frame_path(self.config, recording_id, frame_id)

    def operator_status(self):
        return self._operator().status()

    def operator_start(self, action, scenario_id=None):
        return self._operator().start(action, scenario_id)

    def operator_stop(self, job_id):
        return self._operator().stop(job_id)

    def operator_log(self, job_id, limit):
        return self._operator().log(job_id, limit)

    def _operator(self):
        if self.operator is None:
            raise AccessDenied("sandbox operator is not configured")
        return self.operator

    def _scenario(self, scenario_id):
        row = next((row for row in load_plan(self.config)["scenarios"]
                    if row.get("scenario_id") == scenario_id), None)
        if row is None:
            raise AccessDenied("scenario is not present in the approved plan")
        return row
