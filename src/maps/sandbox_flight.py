"""Adapters from sandbox routes to the stable flight runtime contracts."""

from __future__ import annotations

from src.maps.sandbox_contracts import SandboxMap, SandboxMission
from src.maps.sandbox_routes import SandboxRoute
from src.vision.collection.route import ObservationWaypoint, VisualRoute


def visual_route_for_sandbox(
    map_value: SandboxMap,
    mission: SandboxMission,
    route: SandboxRoute,
):
    if mission.mission_type != "equipment_inspection":
        raise ValueError("only equipment inspection uses a visual observation route")
    target = next(
        item for item in map_value.objects
        if item.object_id == mission.target_object_id
    )
    waypoints = tuple(
        ObservationWaypoint(
            item.waypoint_id,
            item.mission_phase,
            item.east_m,
            item.north_m,
            item.altitude_m,
            item.yaw_deg,
            item.hold_s,
            "labelled_target",
            item.transit_cells,
        )
        for item in route.waypoints
    )
    return VisualRoute(
        route.route_id,
        map_value.map_id,
        target.asset_id,
        target.object_id,
        route.start_cell,
        waypoints,
        tuple(route.return_grid_path[1:]),
    )
