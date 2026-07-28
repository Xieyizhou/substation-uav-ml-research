"""Runtime values shared by the extracted flight execution modules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FlightRuntimeSettings:
    max_horizontal_speed_m_s: float
    max_vertical_speed_m_s: float
    min_risk_speed_m_s: float
    position_gain: float
    reached_horizontal_error_m: float
    reached_vertical_error_m: float
    return_speed_scale: float
    turn_settle_s: float
    waypoint_timeout_mode: object
    connection_timeout_s: float
    position_ready_timeout_s: float
    telemetry_timeout_s: float
    landing_timeout_s: float
    logger_shutdown_timeout_s: float
