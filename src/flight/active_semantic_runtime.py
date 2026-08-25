"""Flight adapter for active-semantic waypoint replacement.

Dynamic obstacle replanning is evaluated before this adapter. Decisions are
consumed once and only at an armed scheduling boundary.
"""

import asyncio

from mavsdk.offboard import VelocityNedYaw

from src.flight.flight_state import publish_mission_event
from src.flight.active_inspection_trial import truncate_trial_route
from src.flight.semantic_decision_queue import pending_queue_depth

RUNTIME_MODE = "active_semantic_inspection"


def replacement_waypoints(decision, *, safety_replan_active):
    if safety_replan_active or decision.get("kind") not in {"equipment_observation", "exploration"}:
        return []
    return [
        {
            "name": f"SIRWP{index:02d}",
            "east_m": float(item["east_m"]),
            "north_m": float(item["north_m"]),
            "down_m": -abs(float(item["altitude_m"])),
            "yaw_deg": float(item.get("yaw_deg", 0)),
            **(
                {"confirmation_anchor": True}
                if item.get("confirmation_anchor") else {}
            ),
        }
        for index, item in enumerate(decision.get("replacement_waypoints", []), 1)
    ]


async def active_semantic_replacement(
    drone, phase_state, replan_config, replan_state, *, safety_replan_active,
):
    if replan_config.get("semantic_runtime_mode") != RUNTIME_MODE:
        return None
    if safety_replan_active or not replan_state.get("semantic_replacement_armed", True):
        return None
    decisions = replan_config.get("semantic_decisions", [])
    index = int(replan_state.get("semantic_decision_index", 0))
    if index >= len(decisions):
        return None
    decision = decisions[index]
    publish_mission_event(
        phase_state,
        "semantic_replacement_requested",
        decision_id=decision.get("decision_id"),
        route_identity=decision.get("route_identity"),
        kind=decision.get("kind"),
        track_id=decision.get("track_id") or decision.get("candidate_id"),
        queue_depth=pending_queue_depth(replan_config, replan_state),
    )
    print(
        "Semantic replacement requested: "
        f"{str(decision.get('decision_id', 'unknown'))[:12]} "
        f"kind={decision.get('kind')}"
    )
    replacement = replacement_waypoints(
        decision, safety_replan_active=safety_replan_active,
    )
    trial = replan_config.get("semantic_trial")
    trial_truncated = False
    if trial is not None:
        count = int(replan_state.get("semantic_trial_replacements", 0))
        if count >= trial["max_semantic_replacements"]:
            replan_state["semantic_trial_stop_requested"] = True
            replan_state["semantic_mission_complete"] = True
            replan_state["terminal_reason"] = "replacement_limit"
            publish_mission_event(
                phase_state,
                "semantic_replacement_rejected",
                decision_id=decision.get("decision_id"),
                route_identity=decision.get("route_identity"),
                reason="replacement_limit",
                queue_depth=pending_queue_depth(replan_config, replan_state),
            )
            return None
        replacement, trial_truncated, trial_distance = truncate_trial_route(
            replacement,
            trial["max_route_distance_m"],
            trial["minimum_altitude_m"],
        )
        if decision.get("kind") == "exploration" and replacement:
            scan = trial.get("exploration_yaw_scan_deg", [])
            if scan:
                confirmation = bool(
                    decision.get("features", {}).get("confirmation_sweep")
                )
                expanded = []
                for waypoint in replacement:
                    expanded.append(waypoint)
                    if confirmation and not waypoint.get("confirmation_anchor"):
                        continue
                    if not confirmation and waypoint is not replacement[-1]:
                        continue
                    expanded.extend({
                        **waypoint,
                        "yaw_deg": float(yaw),
                        "dwell_s": float(trial["exploration_yaw_dwell_s"]),
                    } for yaw in scan)
                replacement = expanded
                for offset, waypoint in enumerate(replacement, 1):
                    waypoint["name"] = f"SIRWP{offset:02d}"
    if not replacement:
        replan_state["semantic_decision_index"] = index + 1
        publish_mission_event(
            phase_state,
            "semantic_replacement_rejected",
            decision_id=decision.get("decision_id"),
            route_identity=decision.get("route_identity"),
            reason="empty_or_unreachable_route",
            queue_depth=pending_queue_depth(replan_config, replan_state),
        )
        print(
            "Semantic replacement rejected: "
            f"{str(decision.get('decision_id', 'unknown'))[:12]} "
            "reason=empty_or_unreachable_route"
        )
        return None
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    replan_state["semantic_decision_index"] = index + 1
    replan_state["semantic_replacement_armed"] = False
    replan_state["semantic_active_decision"] = decision
    replan_state["semantic_active_waypoint_count"] = len(replacement)
    replan_state["semantic_active_route_truncated"] = trial_truncated
    if trial is not None:
        replan_state["semantic_trial_replacements"] = count + 1
    publish_mission_event(
        phase_state, "semantic_route_replaced",
        decision_id=decision.get("decision_id"),
        candidate_id=decision.get("candidate_id"),
        route_identity=decision.get("route_identity"),
        waypoint_count=len(replacement),
    )
    publish_mission_event(
        phase_state,
        "semantic_replacement_accepted",
        decision_id=decision.get("decision_id"),
        route_identity=decision.get("route_identity"),
        kind=decision.get("kind"),
        track_id=decision.get("track_id") or decision.get("candidate_id"),
        queue_depth=pending_queue_depth(replan_config, replan_state),
    )
    print(
        "Semantic replacement accepted: "
        f"{str(decision.get('decision_id', 'unknown'))[:12]} "
        f"waypoints={len(replacement)}"
    )
    if trial_truncated:
        publish_mission_event(
            phase_state,
            "trial_route_truncated",
            decision_id=decision.get("decision_id"),
            executed_distance_m=round(trial_distance, 6),
            waypoint_count=len(replacement),
        )
    return replacement


async def complete_semantic_waypoint(
    drone, phase_state, replan_config, replan_state, waypoint, now_s,
):
    """Publish final semantic-route feedback and wait briefly for its replacement."""
    decision = replan_state.get("semantic_active_decision")
    expected = int(replan_state.get("semantic_active_waypoint_count", 0))
    name = str(waypoint.get("name", ""))
    is_semantic = name.startswith("SIRWP") and expected > 0
    is_final = is_semantic and name == f"SIRWP{expected:02d}"
    if not is_final:
        if not is_semantic:
            replan_state["semantic_replacement_armed"] = True
        return None
    route_truncated = bool(replan_state.get("semantic_active_route_truncated"))
    feedback = {
        "event": "target_completed"
        if decision and decision.get("kind") == "equipment_observation"
        and not route_truncated else "waypoint_reached",
        "candidate_id": decision.get("candidate_id") if decision else None,
        "decision_id": decision.get("decision_id") if decision else None,
        "route_identity": decision.get("route_identity") if decision else None,
        "timestamp_s": now_s,
    }
    replan_config.setdefault("semantic_feedback", []).append(feedback)
    publish_mission_event(
        phase_state,
        feedback["event"],
        candidate_id=feedback["candidate_id"],
        decision_id=feedback["decision_id"],
        route_identity=feedback["route_identity"],
        route_truncated=route_truncated,
    )
    replan_state["semantic_replacement_armed"] = True
    trial = replan_config.get("semantic_trial")
    if (
        trial is not None
        and int(replan_state.get("semantic_trial_replacements", 0))
        >= trial["max_semantic_replacements"]
    ):
        replan_state["semantic_trial_stop_requested"] = True
        replan_state["semantic_mission_complete"] = True
        publish_mission_event(
            phase_state,
            "trial_completed",
            reason="replacement_limit",
            replacement_count=trial["max_semantic_replacements"],
        )
        return None
    deadline = asyncio.get_running_loop().time() + float(
        replan_config.get("semantic_feedback_wait_s", 2.0)
    )
    while asyncio.get_running_loop().time() < deadline:
        if replan_state.get("semantic_mission_complete"):
            return None
        decisions = replan_config.get("semantic_decisions", [])
        if int(replan_state.get("semantic_decision_index", 0)) < len(decisions):
            return await active_semantic_replacement(
                drone, phase_state, replan_config, replan_state,
                safety_replan_active=False,
            )
        await asyncio.sleep(.05)
    return None
