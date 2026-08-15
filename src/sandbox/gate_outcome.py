"""Shared outcome vocabulary for sandbox acceptance gates."""

from __future__ import annotations


PASSED = "passed"
PRODUCT_FAILURE = "product_failure"
ENVIRONMENT_UNAVAILABLE = "environment_unavailable"
EXTERNAL_WARNING = "external_warning"
OUTCOMES = frozenset({
    PASSED, PRODUCT_FAILURE, ENVIRONMENT_UNAVAILABLE, EXTERNAL_WARNING,
})


def check_outcome(passed, reason_code=None, detail=None):
    return {
        "passed": bool(passed),
        "outcome": PASSED if passed else PRODUCT_FAILURE,
        "reason_code": reason_code,
        "detail": detail,
    }


def validate_outcome(value):
    outcome = value.get("outcome")
    if outcome not in OUTCOMES:
        raise ValueError("unsupported sandbox gate outcome")
    if value.get("passed") is not (outcome == PASSED):
        raise ValueError("sandbox gate outcome is inconsistent")
