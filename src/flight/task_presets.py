"""Small, reusable flight-task presets for the selected map and destination."""


COMMON_FLIGHT_ARGS = [
    "--compact-output",
    "--altitude",
    "1.5",
    "--max-speed",
    "0.8",
    "--return-speed-scale",
    "0.7",
    "--waypoint-acceptance",
    "0.3",
]


TASKS = {
    "preview_route": {
        "display_name": "Preview Route",
        "description": "Plan the selected route without connecting to PX4.",
        "args": ["--dry-run", "--compact-output"],
    },
    "fly_to_point": {
        "display_name": "Fly to Point",
        "description": "Fly one way from the start to the selected destination.",
        "args": COMMON_FLIGHT_ARGS,
    },
    "fly_round_trip": {
        "display_name": "Fly Round Trip",
        "description": "Fly to the selected destination, return, and land.",
        "args": [*COMMON_FLIGHT_ARGS, "--return-home"],
    },
    "fly_with_perception": {
        "display_name": "Perception Flight",
        "description": "Fly a round trip and slow down near detected obstacles.",
        "args": [
            *COMMON_FLIGHT_ARGS,
            "--return-home",
            "--enable-perception",
            "--risk-action",
            "slow_down",
            "--waypoint-timeout",
            "auto",
            "--min-risk-speed",
            "0.3",
        ],
    },
    "replan_log_only": {
        "display_name": "Replan Analysis",
        "description": "Evaluate local replans without replacing the active route.",
        "args": [
            *COMMON_FLIGHT_ARGS,
            "--return-home",
            "--enable-perception",
            "--risk-action",
            "log_only",
            "--enable-local-replan",
            "--replan-mode",
            "log_only",
            "--replan-risk-level",
            "danger",
        ],
    },
    "fly_with_replan": {
        "display_name": "Active Replan Flight",
        "description": "Fly slowly and replace the route after a successful replan.",
        "args": [
            "--altitude",
            "1.5",
            "--max-speed",
            "0.5",
            "--return-speed-scale",
            "0.6",
            "--waypoint-acceptance",
            "0.4",
            "--return-home",
            "--enable-perception",
            "--risk-action",
            "log_only",
            "--enable-local-replan",
            "--replan-mode",
            "active",
            "--replan-risk-level",
            "danger",
            "--replan-cooldown",
            "5.0",
            "--max-replans",
            "3",
        ],
    },
}


def task_by_id(task_id):
    """Return a task preset or raise a concise error."""
    try:
        return TASKS[task_id]
    except KeyError as error:
        available = ", ".join(TASKS)
        raise ValueError(f"Unknown flight task {task_id!r}. Available: {available}") from error
