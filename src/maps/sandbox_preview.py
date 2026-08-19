"""Machine-readable and SVG previews for sandbox map revisions."""

from __future__ import annotations

from io import BytesIO

from src.maps.sandbox_contracts import SandboxMap
from src.maps.sandbox_geometry import oriented_corners


def preview_record(map_value: SandboxMap, routes):
    return {
        "sandbox_map_preview_schema_version": 1,
        "map_identity_sha256": map_value.map_identity_sha256,
        "width_m": map_value.width_m,
        "height_m": map_value.height_m,
        "start": [map_value.start_east_m, map_value.start_north_m],
        "objects": [
            {
                "object_id": item.object_id,
                "asset_id": item.asset_id,
                "label_role": item.label_role,
                "corners": [list(point) for point in oriented_corners(item)],
            }
            for item in map_value.objects
        ],
        "routes": [route.to_record() for route in routes],
    }


def preview_svg(map_value: SandboxMap, routes, *, width=960, height=720):
    scale = min((width - 80) / map_value.width_m, (height - 80) / map_value.height_m)
    origin_x, origin_y = 40.0, 40.0

    def point(east, north):
        return origin_x + east * scale, height - origin_y - north * scale

    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<rect x="{origin_x}" y="{height-origin_y-map_value.height_m*scale:.3f}" width="{map_value.width_m*scale:.3f}" height="{map_value.height_m*scale:.3f}" fill="#f7f8f6" stroke="#9aa39f"/>',
    ]
    for grid_x in range(int(map_value.width_m) + 1):
        x, _ = point(grid_x, 0)
        rows.append(f'<line x1="{x:.3f}" y1="{origin_y}" x2="{x:.3f}" y2="{height-origin_y}" stroke="#e8ece9"/>')
    for grid_y in range(int(map_value.height_m) + 1):
        _, y = point(0, grid_y)
        rows.append(f'<line x1="{origin_x}" y1="{y:.3f}" x2="{origin_x+map_value.width_m*scale:.3f}" y2="{y:.3f}" stroke="#e8ece9"/>')
    colors = {"target": "#dbe9e4", "background": "#d8dcda"}
    for item in map_value.objects:
        polygon = " ".join(f"{x:.3f},{y:.3f}" for x, y in (point(*value) for value in oriented_corners(item)))
        rows.append(f'<polygon points="{polygon}" fill="{colors[item.label_role]}" stroke="#20342e" stroke-width="1.5"/>')
        x, y = point(item.east_m, item.north_m)
        rows.append(f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#17211e">{item.asset_id}</text>')
    route_colors = ("#2878b8", "#178064", "#9b6a21")
    for route_index, route in enumerate(routes):
        points = [point(cell[0] + 0.5, cell[1] + 0.5) for cell in route.grid_path]
        path = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
        rows.append(f'<polyline points="{path}" fill="none" stroke="{route_colors[route_index % len(route_colors)]}" stroke-width="3" stroke-dasharray="7 5"/>')
        for index, waypoint in enumerate(route.waypoints, start=1):
            x, y = point(waypoint.east_m, waypoint.north_m)
            rows.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="8" fill="#2878b8"/><text x="{x:.3f}" y="{y+4:.3f}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="white">{index}</text>')
    start_x, start_y = point(map_value.start_east_m, map_value.start_north_m)
    rows.append(f'<circle cx="{start_x:.3f}" cy="{start_y:.3f}" r="10" fill="#087253"/><text x="{start_x:.3f}" y="{start_y+4:.3f}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="white">S</text>')
    rows.append("</svg>")
    return "\n".join(rows) + "\n"


def preview_png(map_value: SandboxMap, routes):
    from src.plotting_runtime import configure_matplotlib

    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    figure, axis = plt.subplots(figsize=(8, 6), dpi=120)
    axis.set_facecolor("#f7f8f6")
    axis.set_xlim(0, map_value.width_m)
    axis.set_ylim(0, map_value.height_m)
    axis.set_aspect("equal")
    axis.grid(True, color="#e8ece9", linewidth=0.7)
    axis.set_xlabel("east (m)")
    axis.set_ylabel("north (m)")
    for item in map_value.objects:
        axis.add_patch(Polygon(
            oriented_corners(item), closed=True,
            facecolor="#dbe9e4" if item.label_role == "target" else "#d8dcda",
            edgecolor="#20342e", linewidth=1.2,
        ))
        axis.text(item.east_m, item.north_m, item.asset_id, ha="center", va="center", fontsize=7)
    colors = ("#2878b8", "#178064", "#9b6a21")
    for index, route in enumerate(routes):
        axis.plot(
            [cell[0] + 0.5 for cell in route.grid_path],
            [cell[1] + 0.5 for cell in route.grid_path],
            linestyle="--", linewidth=1.8, color=colors[index % len(colors)],
        )
    axis.scatter([map_value.start_east_m], [map_value.start_north_m], s=70, color="#087253", zorder=5)
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", metadata={"Software": "UAV Research Sandbox"})
    plt.close(figure)
    return output.getvalue()
