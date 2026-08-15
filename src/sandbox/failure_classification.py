"""Stable failure codes for managed sandbox jobs."""

from __future__ import annotations


FAILURE_RULES = (
    ("resource_exhausted", True, ("no space", "disk budget", "out of memory", "oom")),
    ("deadline_exceeded", True, ("timed out", "timeout", "exceeded")),
    ("dependency_unavailable", False, (
        "no module named", "command not found", "dependency", "not installed",
    )),
    ("artifact_invalid", False, (
        "identity mismatch", "manifest", "receipt failed", "artifact invalid",
    )),
    ("runtime_unavailable", True, (
        "px4", "gazebo", "mavsdk", "telemetry", "topic", "connection",
    )),
    ("recovery_unsafe", False, (
        "recovery failed", "could not be safely adopted", "ownership mismatch",
        "operator exited before",
    )),
)


def classify_failure(job):
    if job.state != "failed":
        job.failure_code = None
        job.failure_retryable = None
        return
    if job.stop_requested:
        job.failure_code, job.failure_retryable = "user_cancelled", True
        return
    text = " ".join((job.error or "", *job.diagnostics)).lower()
    for code, retryable, markers in FAILURE_RULES:
        if any(marker in text for marker in markers):
            job.failure_code, job.failure_retryable = code, retryable
            return
    if job.exit_code not in (None, 0):
        job.failure_code, job.failure_retryable = "command_failed", True
    else:
        job.failure_code, job.failure_retryable = "internal_error", False
