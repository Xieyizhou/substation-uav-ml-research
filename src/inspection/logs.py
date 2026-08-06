"""Bounded log queries with server-side classification."""

from __future__ import annotations

from html import escape

from src.inspection.config import InspectionConfig
from src.inspection.models import LogLine
from src.inspection.readers import bounded_tail


ERROR_TERMS = ("error", "failed", "failure", "traceback", "invalid")
WARNING_TERMS = ("timeout", "timed out", "unstable", "instability", "warning")
SUCCESS_TERMS = ("completed", "complete", "landing confirmed", "validated")


def _level(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ERROR_TERMS):
        return "error"
    if any(term in lowered for term in WARNING_TERMS):
        return "warning"
    if any(term in lowered for term in SUCCESS_TERMS):
        return "success"
    return "info"


def log_tail(config: InspectionConfig, scenario: str, kind: str, limit=200):
    path = config.log(scenario, kind)
    return tuple(
        LogLine(number, escape(text, quote=True), _level(text))
        for number, text in bounded_tail(path, int(limit))
    )
