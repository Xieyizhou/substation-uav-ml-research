"""Single-job asynchronous operator for the local research sandbox."""

from __future__ import annotations

import fcntl
import html
import secrets
import threading

from src.inspection.runtime import LocalProcessAdapter, runtime_status
from src.sandbox import job_recovery
from src.sandbox.failure_classification import classify_failure
from src.sandbox.job_models import SandboxJobStore, TERMINAL_STATES, new_preparing_job, utc_now
from src.sandbox.job_process import (
    ownership_is_held, process_alive, start_job_process, stop_job_pid,
    stop_job_process,
)
from src.sandbox.job_runtime import monitor_process, stop_managed_process
from src.sandbox.managed_flight_stop import cooperative_stop, FLIGHT_ACTIONS
from src.sandbox.workflow import (
    materialize_workflow_receipt, materialize_workflow_recipe,
)
from src.sandbox.storage_policy import OutputBudgetExceeded, capture_output_baseline, require_output_budget


class OperatorBusy(RuntimeError):
    pass


def build_command(*args, **kwargs):
    from src.sandbox.job_commands import build_command as implementation

    return implementation(*args, **kwargs)


class SandboxOperator:
    def __init__(self, config, process_adapter=None):
        self.config = config
        self.store = SandboxJobStore(config.sandbox_jobs_root)
        self.process_adapter = process_adapter or LocalProcessAdapter()
        self._guard = threading.RLock()
        self._active = None
        self._thread = None
        self._stop_requested = threading.Event()
        self._lock_handle = None
        self._orphan_pids = []
        self._recover_interrupted_jobs()

    def _recover_interrupted_jobs(self):
        for job in self.store.list(200):
            if job.state in TERMINAL_STATES:
                continue
            try:
                result = job_recovery.read_process_result(self.store, job.job_id)
            except (OSError, ValueError, TypeError) as error:
                job.state, job.ended_at = "failed", utc_now()
                job.error = f"invalid managed process result: {error}"
                classify_failure(job)
                self.store.write(job)
                continue
            if result is not None:
                job.recovered = True
                job.diagnostics.append("managed result recovered after operator restart")
                job_recovery.apply_process_result(job, result)
                self._finish_recovered(job)
                continue
            alive = process_alive(job.pid)
            owned = alive and ownership_is_held(
                self.store.ownership_path(job.job_id), job.ownership_token
            )
            if self._active is None and owned:
                self._adopt(job)
                continue
            job.state, job.ended_at = "failed", utc_now()
            job.error = "operator exited before recording a terminal job state"
            if alive:
                self._orphan_pids.append(job.pid)
                job.diagnostics.append(
                    "live process identity could not be safely adopted"
                )
            classify_failure(job)
            self.store.write(job)

    def _adopt(self, job):
        self._acquire_lock()
        job.state = "running"
        job.recovered = True
        job.diagnostics.append("managed process adopted after operator restart")
        self.store.write(job)
        self._active = job
        self._thread = threading.Thread(
            target=self._monitor_recovered,
            args=(job,), name=f"sandbox-recovered-{job.action}", daemon=True,
        )
        self._thread.start()

    def _monitor_recovered(self, job):
        def ownership_released():
            return not ownership_is_held(
                self.store.ownership_path(job.job_id), job.ownership_token
            )

        def stop_recovered():
            if not cooperative_stop(self.config, job, ownership_released):
                stop_job_pid(job.pid, stopped=ownership_released)

        try:
            next_storage_check = 0.0
            while not ownership_released():
                if self._stop_requested.wait(0.2):
                    job.state, job.stop_requested = "stopping", True
                    self.store.write(job)
                    if process_alive(job.pid):
                        stop_recovered()
                    break
                violation = job_recovery.recovered_budget_violation(
                    self.config, job, next_storage_check
                )
                next_storage_check = violation[1]
                if violation[0]:
                    job.error = violation[0]
                    if process_alive(job.pid):
                        stop_recovered()
                    break
                if job_recovery.elapsed_seconds(job) > job.timeout_s:
                    job.error = f"job exceeded {job.timeout_s:.0f}s timeout"
                    if process_alive(job.pid):
                        stop_recovered()
                    break
            result = job_recovery.read_process_result(self.store, job.job_id)
            if result is not None:
                job_recovery.apply_process_result(job, result)
            else:
                job.state = "failed"
                job.error = job.error or "managed process ended without an exit record"
                job.ended_at = utc_now()
        except Exception as error:
            job.state, job.ended_at = "failed", utc_now()
            job.error = f"recovery failed: {type(error).__name__}: {error}"
        self._finish_recovered(job)

    def _finish_recovered(self, job):
        job_recovery.finalize_recovered(self.config.project_root, self.store, job)
        with self._guard:
            if self._active is job:
                self._active = None
                self._release_lock()

    def _acquire_lock(self):
        path = self.config.sandbox_operator_root / "active.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise OperatorBusy("another sandbox operator job is active") from error
        self._lock_handle = handle

    def _release_lock(self):
        if self._lock_handle is None:
            return
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        self._lock_handle.close()
        self._lock_handle = None

    def _runtime_conflicts(self):
        return [item.name for item in runtime_status(self.process_adapter) if item.alive]

    def start(self, action, scenario_id=None, parameters=None):
        with self._guard:
            if self._active is not None:
                raise OperatorBusy(f"sandbox job {self._active.job_id} is active")
            self._orphan_pids = [
                pid for pid in self._orphan_pids if process_alive(pid)
            ]
            if self._orphan_pids:
                raise OperatorBusy(
                    "an interrupted sandbox job is still alive; inspect PID "
                    + ", ".join(str(pid) for pid in self._orphan_pids)
                )
            command = build_command(self.config, action, scenario_id, parameters)
            try:
                budget = require_output_budget(self.config, command.action)
                budget["output_baseline_bytes"] = capture_output_baseline(
                    self.config, command.budget_paths)
            except OutputBudgetExceeded as error:
                raise OperatorBusy(str(error)) from error
            conflicts = self._runtime_conflicts() if command.requires_runtime_idle else []
            if conflicts:
                raise OperatorBusy(
                    "runtime processes already exist: " + ", ".join(conflicts)
                )
            self._acquire_lock()
            job = new_preparing_job(
                action, command, budget, secrets.token_hex(16),
            )
            try:
                self.store.write(job)
                recipe = materialize_workflow_recipe(
                    self.config.project_root, self.store, job, command
                )
            except Exception as error:
                job.state = "failed"
                job.ended_at = utc_now()
                job.error = f"workflow recipe failed: {error}"
                classify_failure(job)
                self.store.write(job)
                self._release_lock()
                raise
            self._active = job
            self._stop_requested.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(job, command.argv, recipe),
                name=f"sandbox-{action}",
                daemon=True,
            )
            self._thread.start()
            return job.to_public_record()

    def _run(self, job, argv, recipe):
        process = handle = None
        try:
            process, handle = start_job_process(
                argv, self.store.log_path(job.job_id), self.config.project_root,
                self.store.process_result_path(job.job_id),
                self.store.ownership_path(job.job_id), job.ownership_token,
            )
            with self._guard:
                job.state = "running"
                job.started_at = utc_now()
                job.pid = process.pid
                self.store.write(job)
            monitor_process(
                process, job, self._stop_requested, self.store, self.config
            )
            result = job_recovery.read_process_result(self.store, job.job_id)
            if result is not None:
                job_recovery.apply_process_result(job, result)
            else:
                job.exit_code = process.poll()
                job.state = "failed"
                job.error = job.error or "managed process ended without an exit record"
        except Exception as error:
            job.state = "failed"
            job.error = f"{type(error).__name__}: {error}"
            if process is not None and process.poll() is None:
                stop_managed_process(process, job, self.config)
        finally:
            if handle is not None and not handle.closed:
                handle.close()
            job.ended_at = utc_now()
            job_recovery.append_diagnostics(self.store, job)
            classify_failure(job)
            with self._guard:
                self.store.write(job)
                try:
                    materialize_workflow_receipt(
                        self.config.project_root, self.store, job, recipe
                    )
                except Exception as error:
                    job.state = "failed"
                    job.error = f"workflow receipt failed: {error}"
                    classify_failure(job)
                    self.store.write(job)
                self._active = None
                self._release_lock()

    def stop(self, job_id):
        with self._guard:
            if self._active is None or self._active.job_id != job_id:
                raise ValueError("sandbox job is not active")
            self._active.state = "stopping"
            self._active.stop_requested = True
            self.store.write(self._active)
            self._stop_requested.set()
            return self._active.to_public_record()

    def status(self):
        with self._guard:
            active = None if self._active is None else self._active.to_public_record()
        return {
            "state": "idle" if active is None else active["state"],
            "active_job": active,
            "history": [job.to_public_record() for job in self.store.list(20)],
        }

    def log(self, job_id, limit=200):
        job = self.store.read(job_id)
        if job.sensitive:
            raise ValueError("logs for blind collection jobs are not exposed")
        lines = self.store.log_path(job_id).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        bounded = lines[-max(1, min(int(limit), 1000)) :]
        return [html.escape(line) for line in bounded]

    def shutdown(self):
        with self._guard:
            active = self._active
        if active is not None:
            self.stop(active.job_id)
        if self._thread is not None:
            # Allow the bounded landing request plus forced-cleanup fallback
            # to finish before this server's daemon monitor is torn down.
            self._thread.join(timeout=130.0 if active is not None and active.action in FLIGHT_ACTIONS else 40.0)
