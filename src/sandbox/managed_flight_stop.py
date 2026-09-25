"""Share the existing scoped landing protocol with visual managed flights."""

from copy import copy

from src.sandbox.live_replan_stop import cooperative_stop as existing_stop

FLIGHT_ACTIONS = {"live-replan-flight", "visual-replan-flight", "semantic-flight"}


def cooperative_stop(config, job, has_stopped, *, timeout_s=90.):
    if job.action == "semantic-flight":
        import json
        import time
        from src.sandbox.semantic_qualification import run_directory
        out = run_directory(config.project_root, job.scenario_id)
        out.mkdir(parents=True, exist_ok=True)
        marker = out / 'stop-requested.json'
        if not marker.exists():
            with marker.open('x') as stream:
                json.dump(dict(job_id=job.job_id, reason=job.error or 'operator stop', simulation_only=True), stream)
        deadline = time.monotonic() + timeout_s
        while not has_stopped():
            if time.monotonic() >= deadline:
                return False
            time.sleep(.1)
        return True
    if job.action == "visual-replan-flight":
        job = copy(job)
        job.action = "live-replan-flight"
    return existing_stop(config, job, has_stopped, timeout_s=timeout_s)
