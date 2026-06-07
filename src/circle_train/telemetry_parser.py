from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


ERANGEL_MAP_NAMES = {"Erangel_Main", "Baltic_Main"}


@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    shard_id: str
    map_name: str
    game_mode: str
    created_at: str
    telemetry_url: str


@dataclass(frozen=True)
class PlaneRouteRecord:
    match_id: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    angle: float
    route_group: str
    confidence: float


@dataclass(frozen=True)
class CircleRecord:
    match_id: str
    phase: int
    center_x: float
    center_y: float
    radius: float
    timestamp: str
    elapsed_time: float | None


def parse_match(match_payload: dict[str, Any], shard_id: str) -> MatchRecord | None:
    data = match_payload.get("data", {})
    attributes = data.get("attributes", {})
    match_id = data.get("id", "")
    telemetry_url = ""

    for included in match_payload.get("included", []):
        if included.get("type") == "asset":
            telemetry_url = included.get("attributes", {}).get("URL", "")
            break

    if not match_id or not telemetry_url:
        return None

    return MatchRecord(
        match_id=match_id,
        shard_id=shard_id,
        map_name=attributes.get("mapName", ""),
        game_mode=attributes.get("gameMode", ""),
        created_at=attributes.get("createdAt", ""),
        telemetry_url=telemetry_url,
    )


def parse_circles(match_id: str, telemetry: list[dict[str, Any]]) -> list[CircleRecord]:
    circles: dict[int, CircleRecord] = {}

    for event in telemetry:
        if event.get("_T") != "LogGameStatePeriodic":
            continue

        phase = _phase_from_is_game(event.get("common", {}).get("isGame"))
        if phase is None or phase in circles:
            continue

        game_state = event.get("gameState", {})
        position = game_state.get("poisonGasWarningPosition") or game_state.get("safetyZonePosition")
        radius = game_state.get("poisonGasWarningRadius") or game_state.get("safetyZoneRadius")

        if not position or radius is None or radius <= 0:
            continue

        circles[phase] = CircleRecord(
            match_id=match_id,
            phase=phase,
            center_x=float(position.get("x", 0.0)),
            center_y=float(position.get("y", 0.0)),
            radius=float(radius),
            timestamp=event.get("_D", ""),
            elapsed_time=_as_optional_float(game_state.get("elapsedTime")),
        )

    return [circles[phase] for phase in sorted(circles) if 1 <= phase <= 9]


def parse_plane_route(match_id: str, telemetry: list[dict[str, Any]]) -> PlaneRouteRecord | None:
    points_by_time: dict[int, list[tuple[float, float]]] = defaultdict(list)

    for event in telemetry:
        if event.get("_T") != "LogPlayerPosition":
            continue

        if not _is_close(_as_optional_float(event.get("common", {}).get("isGame")), 0.1):
            continue

        elapsed_time = event.get("elapsedTime")
        character = event.get("character", {})
        location = character.get("location", {})
        x = _as_optional_float(location.get("x"))
        y = _as_optional_float(location.get("y"))

        if elapsed_time is None or x is None or y is None:
            continue

        points_by_time[int(float(elapsed_time))].append((x, y))

    if len(points_by_time) < 2:
        return None

    timeline = []
    for elapsed_time, points in sorted(points_by_time.items()):
        if len(points) < 10:
            continue
        avg_x = sum(point[0] for point in points) / len(points)
        avg_y = sum(point[1] for point in points) / len(points)
        timeline.append((elapsed_time, avg_x, avg_y, len(points)))

    if len(timeline) < 2:
        return None

    start = timeline[0]
    end = timeline[-1]
    dx = end[1] - start[1]
    dy = end[2] - start[2]
    distance = math.hypot(dx, dy)

    if distance <= 0:
        return None

    angle = math.degrees(math.atan2(dy, dx))
    route_group = classify_route(start[1], start[2], end[1], end[2], angle)
    confidence = min(1.0, len(timeline) / 20.0)

    return PlaneRouteRecord(
        match_id=match_id,
        start_x=start[1],
        start_y=start[2],
        end_x=end[1],
        end_y=end[2],
        angle=angle,
        route_group=route_group,
        confidence=confidence,
    )


def classify_route(start_x: float, start_y: float, end_x: float, end_y: float, angle: float) -> str:
    normalized_angle = (angle + 360.0) % 360.0
    mid_x = (start_x + end_x) / 2.0
    mid_y = (start_y + end_y) / 2.0

    if 337.5 <= normalized_angle or normalized_angle < 22.5:
        base = "west_to_east"
    elif 157.5 <= normalized_angle < 202.5:
        base = "east_to_west"
    elif 67.5 <= normalized_angle < 112.5:
        base = "north_to_south"
    elif 247.5 <= normalized_angle < 292.5:
        base = "south_to_north"
    elif 22.5 <= normalized_angle < 67.5:
        base = "northwest_to_southeast"
    elif 112.5 <= normalized_angle < 157.5:
        base = "northeast_to_southwest"
    elif 202.5 <= normalized_angle < 247.5:
        base = "southeast_to_northwest"
    else:
        base = "southwest_to_northeast"

    if mid_y > 560000:
        return f"{base}_military"
    if mid_x < 220000:
        return f"{base}_georgopol"
    if mid_x > 590000 and mid_y < 320000:
        return f"{base}_yasnaya"
    if 330000 <= mid_x <= 500000 and 300000 <= mid_y <= 500000:
        return f"{base}_pochinki"
    if mid_y < 210000:
        return f"{base}_severny"

    return base


def _phase_from_is_game(value: Any) -> int | None:
    number = _as_optional_float(value)
    if number is None:
        return None
    if not number.is_integer():
        return None
    phase = int(number)
    if 1 <= phase <= 9:
        return phase
    return None


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_close(value: float | None, target: float, tolerance: float = 0.01) -> bool:
    if value is None:
        return False
    return abs(value - target) <= tolerance
