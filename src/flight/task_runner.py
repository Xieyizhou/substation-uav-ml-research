"""Run compact flight-task presets through the existing reliable flight engine."""

from src.flight.task_presets import task_by_id
from src.flight.task_runtime import managed_task_runtime
from src.maps.map_catalog import current_map
from src.maps.target_catalog import current_target


def flight_main(args):
    """Lazy-load the larger flight engine only when a task actually runs."""
    from src.flight.fly_astar_path import main

    return main(args)


def task_arguments(task_id, extra_args=None):
    """Combine a task preset with optional advanced CLI overrides."""
    preset = task_by_id(task_id)
    return [*preset["args"], *(extra_args or [])]


def print_task_summary(task_id):
    """Print the selected task, map, and destination before execution."""
    preset = task_by_id(task_id)
    map_entry = current_map()
    target = current_target(map_entry)
    print(f"Task: {task_id} — {preset['display_name']}")
    print(f"Map: {map_entry['id']} — {map_entry['display_name']}")
    print(
        f"Destination: {target['id']} — {target['display_name']} "
        f"at cell {tuple(target['cell'])}"
    )


def run_task(task_id, extra_args=None):
    """Execute one preset in-process so exit codes and managed PIDs stay correct."""
    args = task_arguments(task_id, extra_args)
    print_task_summary(task_id)
    with managed_task_runtime(args):
        result = flight_main(args)
    return 0 if result is None else int(result)
