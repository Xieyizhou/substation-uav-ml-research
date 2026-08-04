"""Static visual benchmark result contract."""

from dataclasses import asdict, dataclass

from src.ml.artifacts import object_sha256
from src.vision.contracts.benchmark_condition import (
    VisualBenchmarkCondition,
    finite_non_negative,
)
from src.vision.contracts.identity_validation import (
    non_negative_int,
    required_text,
    sha256,
)


VISUAL_BENCHMARK_RESULT_SCHEMA_VERSION = 1
TIMING_STAGES = (
    "payload_load_ms", "decode_ms", "preprocess_ms", "backend_call_ms",
    "inference_ms", "postprocess_ms", "end_to_end_ms",
)
FRAME_COUNT_FIELDS = (
    "total", "decoded", "inferred", "skipped", "failed_decode",
    "failed_inference", "labelled",
)
SCHEDULING_FIELDS = (
    "requested_inference_frames", "completed_inference_frames",
    "deadline_misses", "dropped_or_unavailable_frames",
)
RESULT_STATUSES = frozenset({"started", "completed", "failed"})


def _validate_count_mapping(values, expected_fields, name):
    if not isinstance(values, dict) or set(values) != set(expected_fields):
        raise ValueError(f"{name} must contain exactly {list(expected_fields)}")
    result = {}
    for field_name in expected_fields:
        value = values[field_name]
        non_negative_int(value, f"{name}.{field_name}")
        result[field_name] = value
    return result


def _validate_timing_summaries(values):
    if not isinstance(values, dict) or set(values) != set(TIMING_STAGES):
        raise ValueError("timing_summaries must contain every visual timing stage")
    result = {}
    summary_fields = {"count", "min_ms", "max_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms"}
    for stage in TIMING_STAGES:
        summary = values[stage]
        if summary is None:
            result[stage] = None
            continue
        if not isinstance(summary, dict) or set(summary) != summary_fields:
            raise ValueError(f"{stage} summary has invalid fields")
        count = summary["count"]
        non_negative_int(count, f"{stage}.count")
        numeric = {
            field_name: None if summary[field_name] is None else finite_non_negative(summary[field_name], f"{stage}.{field_name}")
            for field_name in summary_fields - {"count"}
        }
        if count == 0 and any(value is not None for value in numeric.values()):
            raise ValueError(f"{stage} cannot have statistics when count is zero")
        if count > 0 and any(value is None for value in numeric.values()):
            raise ValueError(f"{stage} measured summary requires all statistics")
        result[stage] = {"count": count, **numeric}
    return result


@dataclass(frozen=True)
class VisualBenchmarkResult:
    condition_identity_sha256: str
    dataset_identity_sha256: str
    model_identity_sha256: str
    decoder_configuration_id: str
    preprocessing_configuration_id: str
    software_commit_sha: str
    runtime_environment_identity: str
    started: bool
    completion_status: str
    frame_counts: dict[str, int]
    timing_summaries: dict[str, dict | None]
    scheduling_summaries: dict[str, int]
    visual_metrics: dict[str, object]
    resource_metrics: dict[str, object]
    metric_availability: dict[str, bool]
    unavailable_metrics: tuple[str, ...]
    failure_codes: tuple[str, ...]
    raw_result_artifact_manifest_sha256: str | None
    result_schema_version: int = VISUAL_BENCHMARK_RESULT_SCHEMA_VERSION

    def __post_init__(self):
        if self.result_schema_version != VISUAL_BENCHMARK_RESULT_SCHEMA_VERSION:
            raise ValueError(f"unsupported visual result schema {self.result_schema_version}")
        for name in (
            "condition_identity_sha256", "dataset_identity_sha256", "model_identity_sha256",
            "decoder_configuration_id", "preprocessing_configuration_id", "runtime_environment_identity",
        ):
            sha256(getattr(self, name), name)
        object.__setattr__(self, "software_commit_sha", required_text(self.software_commit_sha, "software_commit_sha"))
        if not isinstance(self.started, bool):
            raise ValueError("started must be boolean")
        if self.completion_status not in RESULT_STATUSES:
            raise ValueError("unsupported benchmark completion status")
        if self.completion_status in {"completed", "failed"} and not self.started:
            raise ValueError("a terminal benchmark result must have started")
        frame_counts = _validate_count_mapping(self.frame_counts, FRAME_COUNT_FIELDS, "frame_counts")
        if frame_counts["decoded"] + frame_counts["failed_decode"] > frame_counts["total"]:
            raise ValueError("decoded and failed-decode frames exceed total frames")
        if frame_counts["inferred"] + frame_counts["failed_inference"] > frame_counts["decoded"]:
            raise ValueError("inference outcomes exceed decoded frames")
        if frame_counts["inferred"] + frame_counts["failed_inference"] + frame_counts["skipped"] > frame_counts["decoded"]:
            raise ValueError("inference and skip outcomes exceed decoded frames")
        if frame_counts["labelled"] > frame_counts["total"]:
            raise ValueError("labelled frames cannot exceed total frames")
        object.__setattr__(self, "frame_counts", frame_counts)
        object.__setattr__(self, "scheduling_summaries", _validate_count_mapping(self.scheduling_summaries, SCHEDULING_FIELDS, "scheduling_summaries"))
        object.__setattr__(self, "timing_summaries", _validate_timing_summaries(self.timing_summaries))
        for name in ("visual_metrics", "resource_metrics"):
            values = getattr(self, name)
            if not isinstance(values, dict):
                raise ValueError(f"{name} must be an object")
            for metric, value in values.items():
                required_text(metric, f"{name} key")
                if isinstance(value, bool) or (value is not None and not isinstance(value, (int, float, dict))):
                    raise ValueError(f"{name}.{metric} has an unsupported value")
            object.__setattr__(self, name, dict(values))
        if not isinstance(self.metric_availability, dict) or any(not isinstance(value, bool) for value in self.metric_availability.values()):
            raise ValueError("metric_availability must map names to booleans")
        all_metrics = {**self.visual_metrics, **self.resource_metrics}
        if set(self.metric_availability) != set(all_metrics):
            raise ValueError("metric_availability must cover every visual and resource metric")
        for name, value in all_metrics.items():
            if self.metric_availability[name] != (value is not None):
                raise ValueError(f"metric availability does not match value for {name!r}")
        unavailable = tuple(required_text(value, "unavailable_metrics") for value in self.unavailable_metrics)
        expected_unavailable = {name for name, available in self.metric_availability.items() if not available}
        if set(unavailable) != expected_unavailable:
            raise ValueError("unavailable_metrics must list every unavailable metric exactly")
        object.__setattr__(self, "unavailable_metrics", unavailable)
        object.__setattr__(self, "failure_codes", tuple(required_text(value, "failure_codes") for value in self.failure_codes))
        artifact_hash = sha256(self.raw_result_artifact_manifest_sha256, "raw_result_artifact_manifest_sha256", optional=True)
        object.__setattr__(self, "raw_result_artifact_manifest_sha256", artifact_hash)
        if self.completion_status == "completed" and artifact_hash is None:
            raise ValueError("completed benchmark result requires an artifact manifest hash")
        measured_counts = [summary["count"] for summary in self.timing_summaries.values() if summary is not None]
        if any(count > frame_counts["total"] for count in measured_counts):
            raise ValueError("timing summary count exceeds total frames")
        scheduling = self.scheduling_summaries
        if scheduling["completed_inference_frames"] > scheduling["requested_inference_frames"]:
            raise ValueError("completed inference frames exceed requested frames")
        if scheduling["deadline_misses"] > scheduling["completed_inference_frames"]:
            raise ValueError("deadline misses exceed completed inference frames")

    def result_record(self):
        return asdict(self)

    @property
    def result_identity_sha256(self):
        return object_sha256(self.result_record())

    def to_record(self):
        return {**self.result_record(), "result_identity_sha256": self.result_identity_sha256}

    def validate_references(self, condition, dataset, model, preprocessing):
        if not isinstance(condition, VisualBenchmarkCondition):
            raise TypeError("condition must be a VisualBenchmarkCondition")
        condition.validate_references(dataset, model, preprocessing)
        checks = {
            "condition_identity_sha256": (self.condition_identity_sha256, condition.condition_identity_sha256),
            "dataset_identity_sha256": (self.dataset_identity_sha256, dataset.dataset_identity_sha256),
            "model_identity_sha256": (self.model_identity_sha256, model.model_identity_sha256),
            "decoder_configuration_id": (self.decoder_configuration_id, dataset.decoder_configuration_id),
            "preprocessing_configuration_id": (self.preprocessing_configuration_id, preprocessing.preprocessing_configuration_id),
        }
        mismatches = [name for name, values in checks.items() if values[0] != values[1]]
        if mismatches:
            raise ValueError("visual benchmark result identity mismatch: " + ", ".join(mismatches))
        return True

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        supplied = values.pop("result_identity_sha256", None)
        values["unavailable_metrics"] = tuple(values["unavailable_metrics"])
        values["failure_codes"] = tuple(values["failure_codes"])
        result = cls(**values)
        if supplied != result.result_identity_sha256:
            raise ValueError("benchmark result identity SHA256 mismatch")
        return result
