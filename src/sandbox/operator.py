"""Single-job asynchronous operator for the local research sandbox."""

from __future__ import annotations

import fcntl
import html
from pathlib import Path
import threading
import time

from src.inspection.runtime import LocalProcessAdapter, runtime_status
from src.sandbox.job_commands import build_command
from src.sandbox.job_models import (
    SandboxJob,
    SandboxJobStore,
    TERMINAL_STATES,
    new_job_id,
    utc_now,
)
from src.sandbox.job_process import process_alive, start_job_process, stop_job_process
from src.sandbox.workflow import (
    materialize_workflow_receipt,
    materialize_workflow_recipe,
)


class OperatorBusy(RuntimeError):
    pass


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
            job.state = "failed"
            job.ended_at = utc_now()
            job.error = "operator exited before recording a terminal job state"
            if process_alive(job.pid):
                self._orphan_pids.append(job.pid)
                job.diagnostics.append(
                    "recorded process is still alive; inspect runtime before starting"
                )
            self.store.write(job)

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
            conflicts = self._runtime_conflicts() if command.requires_runtime_idle else []
            if conflicts:
                raise OperatorBusy(
                    "runtime processes already exist: " + ", ".join(conflicts)
                )
            self._acquire_lock()
            job = SandboxJob(
                job_id=new_job_id(action),
                action=command.action,
                state="preparing",
                created_at=utc_now(),
                timeout_s=command.timeout_s,
                sensitive=command.sensitive,
                scenario_id=command.scenario_id,
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
            return job.to_record()

    def _run(self, job, argv, recipe):
        process = handle = None
        started = time.monotonic()
        try:
            process, handle = start_job_process(
                argv, self.store.log_path(job.job_id), self.config.project_root
            )
            with self._guard:
                job.state = "running"
                job.started_at = utc_now()
                job.pid = process.pid
                self.store.write(job)
            while process.poll() is None:
                if self._stop_requested.wait(0.2):
                    job.state = "stopping"
                    job.stop_requested = True
                    self.store.write(job)
                    stop_job_process(process)
                    break
                if time.monotonic() - started > job.timeout_s:
                    job.error = f"job exceeded {job.timeout_s:.0f}s timeout"
                    stop_job_process(process)
                    break
            job.exit_code = process.poll()
            if job.exit_code == 0 and not job.stop_requested and job.error is None:
                job.state = "complete"
            else:
                job.state = "failed"
                if job.error is None:
                    job.error = (
                        "job stopped by request"
                        if job.stop_requested
                        else f"job exited with code {job.exit_code}"
                    )
        except Exception as error:
            job.state = "failed"
            job.error = f"{type(error).__name__}: {error}"
            if process is not None and process.poll() is None:
                stop_job_process(process)
        finally:
            if handle is not None and not handle.closed:
                handle.close()
            job.ended_at = utc_now()
            self._append_diagnostics(job)
            with self._guard:
                self.store.write(job)
                try:
                    materialize_workflow_receipt(
                        self.config.project_root, self.store, job, recipe
                    )
                except Exception as error:
                    job.state = "failed"
                    job.error = f"workflow receipt failed: {error}"
                    self.store.write(job)
                self._active = None
                self._release_lock()

    def _append_diagnostics(self, job):
        if job.state != "failed" or job.sensitive:
            return
        try:
            lines = self.store.log_path(job.job_id).read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return
        noteworthy = [
            line.strip()
            for line in lines
            if any(word in line.lower() for word in ("error", "failed", "timeout"))
        ]
        job.diagnostics.extend(noteworthy[-5:])

    def stop(self, job_id):
        with self._guard:
            if self._active is None or self._active.job_id != job_id:
                raise ValueError("sandbox job is not active")
            self._active.state = "stopping"
            self._active.stop_requested = True
            self.store.write(self._active)
            self._stop_requested.set()
            return self._active.to_record()

    def status(self):
        with self._guard:
            active = None if self._active is None else self._active.to_record()
        return {
            "state": "idle" if active is None else active["state"],
            "active_job": active,
            "history": [job.to_record() for job in self.store.list(20)],
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
            self._thread.join(timeout=40.0)
