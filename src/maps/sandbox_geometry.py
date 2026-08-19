"""Geometry and conservative grid rasterization for sandbox maps."""

from __future__ import annotations

import math

from src.maps.sandbox_contracts import SandboxMapObject

Point = tuple[float, float]
Cell = tuple[int, int]


def oriented_corners(item: SandboxMapObject) -> tuple[Point, ...]:
    half_w, half_d = item.width_m / 2.0, item.depth_m / 2.0
    angle = math.radians(item.yaw_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners = []
    for local_x, local_y in ((-half_w, -half_d), (half_w, -half_d), (half_w, half_d), (-half_w, half_d)):
        corners.append((
            item.east_m + local_x * cos_a - local_y * sin_a,
            item.north_m + local_x * sin_a + local_y * cos_a,
        ))
    return tuple(corners)


def _axes(points: tuple[Point, ...]):
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        edge = (next_point[0] - point[0], next_point[1] - point[1])
        length = math.hypot(*edge)
        if length:
            yield (-edge[1] / length, edge[0] / length)


def _projection(points: tuple[Point, ...], axis: Point):
    values = [point[0] * axis[0] + point[1] * axis[1] for point in points]
    return min(values), max(values)


def polygons_overlap(first: tuple[Point, ...], second: tuple[Point, ...], *, tolerance=1e-9):
    for axis in (*tuple(_axes(first)), *tuple(_axes(second))):
        first_min, first_max = _projection(first, axis)
        second_min, second_max = _projection(second, axis)
        if first_max <= second_min + tolerance or second_max <= first_min + tolerance:
            return False
    return True


def objects_overlap(first: SandboxMapObject, second: SandboxMapObject):
    return polygons_overlap(oriented_corners(first), oriented_corners(second))


def object_inside_map(item: SandboxMapObject, width_m: float, height_m: float):
    return all(
        0.0 <= east <= width_m and 0.0 <= north <= height_m
        for east, north in oriented_corners(item)
    )


def object_cells(item: SandboxMapObject, width: int, height: int) -> set[Cell]:
    polygon = oriented_corners(item)
    min_x = max(0, math.floor(min(point[0] for point in polygon)))
    max_x = min(width - 1, math.ceil(max(point[0] for point in polygon)) - 1)
    min_y = max(0, math.floor(min(point[1] for point in polygon)))
    max_y = min(height - 1, math.ceil(max(point[1] for point in polygon)) - 1)
    cells = set()
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            square = ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1))
            if polygons_overlap(polygon, square) or any(
                x <= east <= x + 1 and y <= north <= y + 1 for east, north in polygon
            ):
                cells.add((x, y))
    return cells


def point_cell(east_m: float, north_m: float) -> Cell:
    return math.floor(east_m), math.floor(north_m)
