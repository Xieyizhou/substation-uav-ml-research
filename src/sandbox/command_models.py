"""Typed command specification used by the sandbox operator."""

from __future__ import annotations

from dataclasses import dataclass

from src.sandbox.workflow import WorkflowArtifact


@dataclass(frozen=True)
class SandboxCommand:
    action: str
    argv: tuple[str, ...]
    timeout_s: float
    scenario_id: str | None = None
    sensitive: bool = False
    workflow: str = "managed_operation"
    artifacts: tuple[WorkflowArtifact, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    budget_paths: tuple[str, ...] = ()
    requires_runtime_idle: bool = True
