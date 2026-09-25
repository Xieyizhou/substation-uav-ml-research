"""Runtime supervision shared by newly started sandbox jobs."""

from __future__ import annotations

import time

from src.sandbox.job_process import stop_job_process
from src.sandbox.storage_policy import output_budget_violation
from src.sandbox.managed_flight_stop import cooperative_stop


def stop_managed_process(process, job, config):
    if not cooperative_stop(config, job, lambda: process.poll() is not None):
        stop_job_process(process)


def monitor_process(process, job, stop_requested, store, config):
    started = time.monotonic()
    next_storage_check = started
    while process.poll() is None:
        if stop_requested.wait(0.2):
            job.state, job.stop_requested = "stopping", True
            store.write(job)
            stop_managed_process(process, job, config)
            return
        now = time.monotonic()
        if now - started > job.timeout_s:
            job.error = f"job exceeded {job.timeout_s:.0f}s timeout"
            stop_managed_process(process, job, config)
            return
        if now >= next_storage_check:
            violation = output_budget_violation(config, job)
            if violation:
                job.error = violation
                stop_managed_process(process, job, config)
                return
            next_storage_check = now + 2.0
