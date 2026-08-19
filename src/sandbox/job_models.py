"""Persistent records for locally managed sandbox jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from src.ml.artifacts import object_sha256


JOB_STATES = frozenset(
    {"preparing", "running", "stopping", "complete", "failed"}
)
TERMINAL_STATES = frozenset({"complete", "failed"})


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_job_id(action):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{action}-{uuid4().hex[:8]}"


@dataclass
class SandboxJob:
    job_id: str
    action: str
    state: str
    created_at: str
    timeout_s: float
    sensitive: bool = False
    scenario_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    pid: int | None = None
    ownership_token: str | None = None
    output_budget_bytes: int = 0
    disk_free_bytes_at_start: int | None = None
    disk_reserve_bytes: int = 0
    output_baseline_bytes: dict[str, int] = field(default_factory=dict)
    exit_code: int | None = None
    error: str | None = None
    failure_code: str | None = None
    failure_retryable: bool | None = None
    stop_requested: bool = False
    recovered: bool = False
    diagnostics: list[str] = field(default_factory=list)
    sandbox_job_schema_version: int = 1

    def __post_init__(self):
        if self.state not in JOB_STATES:
            raise ValueError(f"unsupported sandbox job state: {self.state}")
        if not self.job_id or Path(self.job_id).name != self.job_id:
            raise ValueError("invalid sandbox job identifier")
        if self.output_budget_bytes < 0 or self.disk_reserve_bytes < 0:
            raise ValueError("sandbox output budgets cannot be negative")

    def to_record(self):
        record = asdict(self)
        record["job_identity_sha256"] = object_sha256(record)
        return record

    def to_public_record(self):
        record = self.to_record()
        record.pop("ownership_token", None)
        return record

    @classmethod
    def from_record(cls, value):
        record = dict(value)
        supplied = record.pop("job_identity_sha256", None)
        job = cls(**record)
        if supplied != object_sha256(record):
            raise ValueError("sandbox job identity mismatch")
        return job


class SandboxJobStore:
    def __init__(self, root):
        self.root = Path(root)

    def directory(self, job_id):
        if not job_id or Path(job_id).name != job_id:
            raise ValueError("invalid sandbox job identifier")
        return self.root / job_id

    def log_path(self, job_id):
        return self.directory(job_id) / "job.log"

    def process_result_path(self, job_id):
        return self.directory(job_id) / "process_result.json"

    def ownership_path(self, job_id):
        return self.directory(job_id) / "process.owner"

    def write(self, job):
        directory = self.directory(job.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "job.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(job.to_record(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def read(self, job_id):
        path = self.directory(job_id) / "job.json"
        return SandboxJob.from_record(json.loads(path.read_text(encoding="utf-8")))

    def list(self, limit=50):
        paths = sorted(self.root.glob("*/job.json"), reverse=True)
        jobs = []
        for path in paths[: max(1, min(int(limit), 200))]:
            try:
                jobs.append(
                    SandboxJob.from_record(json.loads(path.read_text(encoding="utf-8")))
                )
            except (OSError, ValueError, json.JSONDecodeError, TypeError):
                continue
        return jobs


def new_preparing_job(action, command, budget, ownership_token):
    return SandboxJob(
        job_id=new_job_id(action), action=command.action, state="preparing",
        created_at=utc_now(), timeout_s=command.timeout_s,
        sensitive=command.sensitive, scenario_id=command.scenario_id,
        ownership_token=ownership_token,
        output_budget_bytes=budget["budget_bytes"],
        disk_free_bytes_at_start=budget["free_bytes"],
        disk_reserve_bytes=budget["reserve_bytes"],
        output_baseline_bytes=budget.get("output_baseline_bytes", {}),
    )
