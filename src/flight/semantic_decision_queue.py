"""Deterministic queue envelopes for live semantic route decisions."""

from __future__ import annotations

import hashlib
import json


def _identity(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decision_envelope(decision, *, created_at, trigger=None):
    route = decision.get("replacement_waypoints", [])
    route_identity = _identity(route)
    stable = {
        "trigger": trigger or decision.get("trigger", "planner_decision"),
        "kind": decision.get("kind", "exploration"),
        "candidate_id": decision.get("candidate_id"),
        "track_id": decision.get("track_id") or decision.get("candidate_id"),
        "route_identity": route_identity,
        "replacement_waypoints": route,
    }
    return {
        **decision,
        **stable,
        "decision_id": decision.get("decision_id") or _identity(stable),
        "created_at": float(created_at),
    }


def enqueue_decision(replan_config, replan_state, decision, *, created_at, trigger=None):
    envelope = decision_envelope(decision, created_at=created_at, trigger=trigger)
    seen = replan_state.setdefault("semantic_decision_ids", set())
    if envelope["decision_id"] in seen:
        return None
    if replan_state.get("semantic_replacement_armed") is False:
        return None
    if pending_queue_depth(replan_config, replan_state) >= 1:
        return None
    seen.add(envelope["decision_id"])
    queue = replan_config.setdefault("semantic_decisions", [])
    queue.append(envelope)
    replan_state["semantic_queue_depth"] = len(queue) - int(
        replan_state.get("semantic_decision_index", 0)
    )
    return envelope


def pending_queue_depth(replan_config, replan_state) -> int:
    return max(
        0,
        len(replan_config.get("semantic_decisions", []))
        - int(replan_state.get("semantic_decision_index", 0)),
    )
