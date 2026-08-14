"""Progress-aware waypoint timeout policy for slow simulation runs."""


def parse_waypoint_timeout(value):
    if str(value).lower() == "auto":
        return "auto"
    try:
        timeout_s = float(value)
    except ValueError as error:
        raise ValueError(
            "--waypoint-timeout must be 'auto' or a positive number"
        ) from error
    if timeout_s <= 0:
        raise ValueError("--waypoint-timeout must be 'auto' or a positive number")
    return timeout_s


class WaypointProgressWatchdog:
    """Extend an automatic timeout while position makes bounded progress."""

    def __init__(
        self,
        now_s,
        timeout_s,
        initial_distance_m,
        *,
        enabled,
        progress_step_m=0.25,
        stall_window_s=30.0,
        absolute_factor=4.0,
        absolute_limit_s=300.0,
    ):
        self.enabled = enabled
        self.soft_deadline_s = now_s + timeout_s
        absolute_duration_s = min(timeout_s * absolute_factor, absolute_limit_s)
        self.absolute_deadline_s = now_s + max(timeout_s, absolute_duration_s)
        self.progress_step_m = progress_step_m
        self.stall_window_s = stall_window_s
        self.best_distance_m = initial_distance_m
        self.last_progress_s = now_s

    def observe(self, distance_m, now_s):
        if distance_m is None:
            return
        if self.best_distance_m is None:
            self.best_distance_m = distance_m
            self.last_progress_s = now_s
            return
        if distance_m <= self.best_distance_m - self.progress_step_m:
            self.best_distance_m = distance_m
            self.last_progress_s = now_s

    def should_continue(self, now_s):
        if now_s >= self.absolute_deadline_s:
            return False
        if now_s < self.soft_deadline_s:
            return True
        return self.enabled and now_s - self.last_progress_s <= self.stall_window_s

    def stop_reason(self, now_s):
        if now_s >= self.absolute_deadline_s:
            return "absolute waypoint limit reached"
        if not self.enabled:
            return "configured waypoint timeout reached"
        stalled_s = max(0.0, now_s - self.last_progress_s)
        return f"no 0.25 m progress for {stalled_s:.1f}s"
