"""Identity-bound recipes and receipts for managed sandbox workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json
from src.vision.contracts.identity_validation import required_text


WORKFLOW_RECIPE_SCHEMA_VERSION = 1
WORKFLOW_RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WorkflowArtifact:
    kind: str
    identity: str
    path: str

    def __post_init__(self):
        for name in ("kind", "identity", "path"):
            object.__setattr__(self, name, required_text(getattr(self, name), name))


@dataclass(frozen=True)
class WorkflowRecipe:
    job_id: str
    workflow: str
    action: str
    created_at: str
    timeout_s: float
    software_commit_sha: str
    command_identity_sha256: str
    artifacts: tuple[WorkflowArtifact, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    scenario_id: str | None = None
    workflow_recipe_schema_version: int = WORKFLOW_RECIPE_SCHEMA_VERSION

    def __post_init__(self):
        if self.workflow_recipe_schema_version != WORKFLOW_RECIPE_SCHEMA_VERSION:
            raise ValueError("unsupported sandbox workflow recipe schema")
        for name in ("job_id", "workflow", "action", "created_at", "software_commit_sha"):
            object.__setattr__(self, name, required_text(getattr(self, name), name))
        if self.timeout_s <= 0:
            raise ValueError("workflow timeout must be positive")
        if len(self.command_identity_sha256) != 64:
            raise ValueError("invalid workflow command identity")
        if any(Path(path).is_absolute() or ".." in Path(path).parts
               for path in self.expected_outputs):
            raise ValueError("workflow outputs must stay inside the project")

    def identity_record(self):
        return asdict(self)

    @property
    def recipe_identity_sha256(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {**self.identity_record(), "recipe_identity_sha256": self.recipe_identity_sha256}

    @classmethod
    def from_record(cls, value):
        record = dict(value)
        supplied = record.pop("recipe_identity_sha256", None)
        record["artifacts"] = tuple(
            WorkflowArtifact(**item) for item in record.get("artifacts", ())
        )
        record["expected_outputs"] = tuple(record.get("expected_outputs", ()))
        recipe = cls(**record)
        if supplied != recipe.recipe_identity_sha256:
            raise ValueError("sandbox workflow recipe identity mismatch")
        return recipe


def artifact_reference(project_root, kind, identity, path):
    root = Path(project_root).resolve()
    candidate = Path(path).resolve()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("workflow artifact is outside the project") from error
    return WorkflowArtifact(kind, str(identity), relative)


def materialize_workflow_recipe(project_root, store, job, command):
    recipe = WorkflowRecipe(
        job_id=job.job_id,
        workflow=command.workflow,
        action=command.action,
        created_at=job.created_at,
        timeout_s=command.timeout_s,
        software_commit_sha=git_commit(project_root),
        command_identity_sha256=object_sha256(list(command.argv)),
        artifacts=command.artifacts,
        expected_outputs=command.expected_outputs,
        scenario_id=command.scenario_id,
    )
    write_json(store.directory(job.job_id) / "workflow_recipe.json", recipe.to_record())
    return recipe


def _output_records(project_root, expected_outputs):
    root = Path(project_root)
    records = []
    for relative in expected_outputs:
        path = root / relative
        records.append({
            "path": relative,
            "present": path.is_file(),
            "sha256": file_sha256(path) if path.is_file() else None,
        })
    return records


def materialize_workflow_receipt(project_root, store, job, recipe):
    log = store.log_path(job.job_id)
    state = "stopped" if job.stop_requested else job.state
    record = {
        "workflow_receipt_schema_version": WORKFLOW_RECEIPT_SCHEMA_VERSION,
        "recipe_identity_sha256": recipe.recipe_identity_sha256,
        "job_identity_sha256": job.to_record()["job_identity_sha256"],
        "job_id": job.job_id,
        "workflow": recipe.workflow,
        "action": job.action,
        "state": state,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "exit_code": job.exit_code,
        "stop_requested": job.stop_requested,
        "diagnostics": list(job.diagnostics),
        "log_sha256": file_sha256(log) if log.is_file() and not job.sensitive else None,
        "outputs": _output_records(project_root, recipe.expected_outputs),
    }
    record["receipt_identity_sha256"] = object_sha256(record)
    write_json(store.directory(job.job_id) / "workflow_receipt.json", record)
    return record


def inspect_workflow(path, identity_field):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = value.pop(identity_field, None)
    if supplied != object_sha256(value):
        raise ValueError("sandbox workflow identity mismatch")
    value[identity_field] = supplied
    return value
