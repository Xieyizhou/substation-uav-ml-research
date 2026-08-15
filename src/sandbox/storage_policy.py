"""Disk budgets and read-only storage summaries for sandbox outputs."""

from __future__ import annotations

from pathlib import Path
import shutil


MIB = 1024 ** 2
GIB = 1024 ** 3
DEFAULT_BUDGET_BYTES = GIB
ACTION_BUDGET_BYTES = {
    "doctor": 16 * MIB,
    "package-inspect-v2": 64 * MIB,
    "sandbox-demo": 64 * MIB,
    "sandbox-acceptance": GIB,
    "flight-smoke": GIB,
    "lidar-challenge-gate": 2 * GIB,
    "lidar-replay-gate": 4 * GIB,
    "lidar-closed-loop-next": 4 * GIB,
    "experiment-run": 8 * GIB,
    "training-view-v2": 8 * GIB,
    "training-smoke-v2": 16 * GIB,
    "collection-single": 8 * GIB,
    "collection-gate": 40 * GIB,
}


class OutputBudgetExceeded(RuntimeError):
    pass


def directory_size(path):
    total = 0
    for candidate in Path(path).rglob("*") if Path(path).exists() else ():
        try:
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def output_budget(config, action):
    usage = shutil.disk_usage(config.project_root)
    budget = ACTION_BUDGET_BYTES.get(action, DEFAULT_BUDGET_BYTES)
    reserve = max(0, int(config.minimum_free_gib * GIB))
    required = budget + reserve
    return {
        "action": action,
        "budget_bytes": budget,
        "reserve_bytes": reserve,
        "free_bytes": usage.free,
        "required_free_bytes": required,
        "passed": usage.free >= required,
    }


def require_output_budget(config, action):
    result = output_budget(config, action)
    if not result["passed"]:
        free = result["free_bytes"] / GIB
        required = result["required_free_bytes"] / GIB
        raise OutputBudgetExceeded(
            f"disk budget requires {required:.1f} GiB free; {free:.1f} GiB available"
        )
    return result


def output_budget_violation(config, job):
    usage = shutil.disk_usage(config.project_root)
    if usage.free < job.disk_reserve_bytes:
        return "disk budget exceeded: configured free-space reserve was reached"
    if job.disk_free_bytes_at_start is None:
        return None
    consumed = max(0, job.disk_free_bytes_at_start - usage.free)
    if consumed > job.output_budget_bytes:
        return "disk budget exceeded: managed job consumed its output allowance"
    return None


def storage_summary(config):
    root = config.project_root / "outputs/sandbox"
    usage = shutil.disk_usage(config.project_root)
    groups = []
    for path in sorted(root.iterdir()) if root.is_dir() else ():
        if path.is_dir() and not path.is_symlink():
            groups.append({"name": path.name, "bytes": directory_size(path)})
    return {
        "profile": config.profile,
        "sandbox_output_root": root.relative_to(config.project_root).as_posix(),
        "sandbox_output_bytes": sum(item["bytes"] for item in groups),
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "minimum_free_bytes": int(config.minimum_free_gib * GIB),
        "action_budgets_bytes": dict(sorted(ACTION_BUDGET_BYTES.items())),
        "groups": groups,
        "formal_retention_locked": config.profile == "formal",
    }
