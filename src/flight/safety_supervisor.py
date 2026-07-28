"""Independent safety policy above learned perception outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyDecision:
    action: str
    speed_scale: float
    reason: str


class SafetySupervisor:
    """Convert perception health and risk into an enforceable flight action."""

    def __init__(self, *, stale_after_s=0.5, minimum_confidence=0.5):
        self.stale_after_s = float(stale_after_s)
        self.minimum_confidence = float(minimum_confidence)

    def evaluate(self, detection, requested_action="log_only"):
        if not detection:
            return SafetyDecision("continue", 1.0, "no active perception source")
        if detection.get("sensor_healthy") is False:
            return SafetyDecision(
                "hover_then_land",
                0.0,
                detection.get("sensor_message") or "sensor is unhealthy",
            )
        age_s = detection.get("sensor_frame_age_s")
        if age_s is not None and age_s > self.stale_after_s:
            return SafetyDecision("hover_then_land", 0.0, "sensor data is stale")
        risk = detection.get("risk_estimate")
        confidence = getattr(risk, "confidence", 1.0) if risk else 1.0
        if confidence < self.minimum_confidence:
            return SafetyDecision("hover", 0.0, "risk confidence is too low")
        level = detection.get("risk_level", "clear")
        if level == "danger":
            if requested_action == "stop_and_land":
                return SafetyDecision("hover_then_land", 0.0, "danger risk")
            return SafetyDecision("replan_or_hover", 0.0, "danger risk")
        if level == "warning":
            return SafetyDecision("slow_down", 0.5, "warning risk")
        return SafetyDecision("continue", 1.0, "risk is acceptable")
