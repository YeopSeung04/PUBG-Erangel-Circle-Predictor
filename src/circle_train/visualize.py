from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from circle_train.config import DATA_DIR, PROCESSED_DIR
from circle_train.train import build_features, predict_circle


DEFAULT_REGISTERED_MAP_PATH = DATA_DIR / "assets" / "erangel_registered_to_reference.png"
DEFAULT_OFFICIAL_MAP_PATH = DATA_DIR / "assets" / "erangel_official_map.webp"
DEFAULT_MODEL_PATH = DATA_DIR / "models" / "circle_transition_model.pkl"
DEFAULT_OUTPUT_DIR = DATA_DIR / "visualizations"
DEFAULT_VISUAL_WORLD_SIZE = 800000.0


@dataclass(frozen=True)
class MapTransform:
    world_size: float
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a PUBG Erangel plane route and circle prediction on a map.")
    parser.add_argument("--input", default=str(PROCESSED_DIR / "match_sequences.csv"))
    parser.add_argument("--map-image", default=str(default_map_path()))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output", default=None)
    parser.add_argument("--match-id", default=None)
    parser.add_argument("--match-index", type=int, default=0)
    parser.add_argument("--phase", type=int, default=1, choices=range(1, 9))
    parser.add_argument(
        "--world-size",
        type=float,
        default=DEFAULT_VISUAL_WORLD_SIZE,
        help="Visual map coordinate size in centimeters. Erangel's visible grid is 8km, so default is 800000.",
    )
    parser.add_argument("--map-left", type=float, default=0.0)
    parser.add_argument("--map-top", type=float, default=0.0)
    parser.add_argument("--map-right", type=float, default=None)
    parser.add_argument("--map-bottom", type=float, default=None)
    parser.add_argument("--draw-grid", action="store_true", default=True, help="Draw PUBG-style grid labels over the clean map.")
    parser.add_argument("--hide-grid", action="store_false", dest="draw_grid", help="Do not draw grid labels.")
    parser.add_argument("--show-actual-next", action="store_true", help="Draw the actual next circle for debugging.")
    parser.add_argument("--show-prediction", action="store_true", help="Draw the predicted next circle for debugging.")
    parser.add_argument("--show-legend", action="store_true", help="Draw a debug legend box.")
    args = parser.parse_args()

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise SystemExit("Pillow is required for visualization. Run: py -3 -m pip install Pillow") from error

    input_path = Path(args.input)
    map_path = Path(args.map_image)
    model_path = Path(args.model)
    row = select_match_row(input_path, args.match_id, args.match_index)

    image = Image.open(map_path).convert("RGBA")
    transform = MapTransform(
        world_size=args.world_size,
        left=args.map_left,
        top=args.map_top,
        right=args.map_right if args.map_right is not None else float(image.size[0]),
        bottom=args.map_bottom if args.map_bottom is not None else float(image.size[1]),
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    if args.draw_grid:
        draw_grid(draw, transform, font)

    plane_start = [float(row["plane_start_x"]), float(row["plane_start_y"])]
    plane_end = [float(row["plane_end_x"]), float(row["plane_end_y"])]
    map_route = clip_infinite_line_to_world_square(plane_start, plane_end, transform.world_size)
    if map_route is None:
        raise SystemExit("Plane route line does not intersect the visual map square.")

    plane_entry, plane_exit = order_route_by_raw_direction(map_route, plane_start, plane_end)
    draw_plane_route(draw, plane_entry, plane_exit, transform)

    current_circle = read_circle(row, args.phase)
    next_circle = read_circle(row, args.phase + 1)
    if current_circle is None or next_circle is None:
        raise SystemExit(f"Match does not contain phase {args.phase} -> {args.phase + 1} circle data.")

    draw_circle(draw, current_circle, transform, (245, 245, 245, 235), width=3)

    prediction = None
    if model_path.exists() and (args.show_prediction or args.show_legend):
        prediction = predict_next_circle(model_path, row, args.phase, current_circle)

    if args.show_actual_next:
        draw_circle(draw, next_circle, transform, (0, 210, 255, 230), width=3)

    if args.show_prediction and prediction is not None:
        draw_circle(draw, prediction, transform, (255, 50, 80, 235), width=3)
        draw_error_line(draw, prediction, next_circle, transform)

    if args.show_legend:
        draw_legend(draw, row, args.phase, prediction, next_circle, transform, font)

    rendered = Image.alpha_composite(image, overlay).convert("RGB")

    output_path = resolve_output_path(args.output, row["match_id"], args.phase)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "match_id": row["match_id"],
                "route_group": row.get("route_group"),
                "phase": args.phase,
                "visual_world_size": transform.world_size,
                "raw_plane_start": plane_start,
                "raw_plane_end": plane_end,
                "plane_entry": plane_entry,
                "plane_exit": plane_exit,
                "current_circle": current_circle,
                "actual_next_circle": next_circle,
                "predicted_next_circle": prediction,
                "center_error_m": center_error_m(prediction, next_circle) if prediction is not None else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def default_map_path() -> Path:
    if DEFAULT_REGISTERED_MAP_PATH.exists():
        return DEFAULT_REGISTERED_MAP_PATH

    return DEFAULT_OFFICIAL_MAP_PATH


def select_match_row(input_path: Path, match_id: str | None, match_index: int) -> dict[str, str]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise SystemExit(f"No rows found: {input_path}")

    if match_id:
        for row in rows:
            if row["match_id"] == match_id:
                return row

        raise SystemExit(f"Match ID not found: {match_id}")

    if match_index < 0 or match_index >= len(rows):
        raise SystemExit(f"match-index out of range: {match_index}. Available range: 0..{len(rows) - 1}")

    return rows[match_index]


def predict_next_circle(
    model_path: Path,
    row: dict[str, str],
    phase: int,
    current_circle: list[float],
) -> list[float]:
    with model_path.open("rb") as file:
        model: dict[str, Any] = pickle.load(file)

    route_groups = model["route_groups"]
    target_mode = model.get("target_mode", "absolute")
    features = build_features(row, phase, current_circle, route_groups)
    return predict_circle(model["phase_models"], phase, features, target_mode, current_circle)


def read_circle(row: dict[str, str], phase: int) -> list[float] | None:
    keys = [f"p{phase}_center_x", f"p{phase}_center_y", f"p{phase}_radius"]
    if any(row.get(key) in (None, "") for key in keys):
        return None

    return [float(row[key]) for key in keys]


def world_to_pixel(x: float, y: float, transform: MapTransform) -> tuple[float, float]:
    return (
        transform.left + x / transform.world_size * transform.width,
        transform.bottom - y / transform.world_size * transform.height,
    )


def clip_infinite_line_to_world_square(
    start: list[float],
    end: list[float],
    world_size: float,
) -> tuple[list[float], list[float]] | None:
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0.0 and dy == 0.0:
        return None

    intersections: list[tuple[float, list[float]]] = []
    add_line_intersection(intersections, x0, y0, dx, dy, x=0.0, world_size=world_size)
    add_line_intersection(intersections, x0, y0, dx, dy, x=world_size, world_size=world_size)
    add_line_intersection(intersections, x0, y0, dx, dy, y=0.0, world_size=world_size)
    add_line_intersection(intersections, x0, y0, dx, dy, y=world_size, world_size=world_size)

    unique: list[tuple[float, list[float]]] = []
    for t, point in sorted(intersections, key=lambda value: value[0]):
        if not any(distance_squared(point, existing_point) < 1e-4 for _, existing_point in unique):
            unique.append((t, point))

    if len(unique) < 2:
        return None

    return (unique[0][1], unique[-1][1])


def add_line_intersection(
    intersections: list[tuple[float, list[float]]],
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    world_size: float,
    x: float | None = None,
    y: float | None = None,
) -> None:
    if x is not None:
        if dx == 0.0:
            return
        t = (x - x0) / dx
        candidate_y = y0 + t * dy
        if -1e-6 <= candidate_y <= world_size + 1e-6:
            intersections.append((t, [x, min(max(candidate_y, 0.0), world_size)]))
        return

    if y is not None:
        if dy == 0.0:
            return
        t = (y - y0) / dy
        candidate_x = x0 + t * dx
        if -1e-6 <= candidate_x <= world_size + 1e-6:
            intersections.append((t, [min(max(candidate_x, 0.0), world_size), y]))


def order_route_by_raw_direction(
    route: tuple[list[float], list[float]],
    raw_start: list[float],
    raw_end: list[float],
) -> tuple[list[float], list[float]]:
    first, second = route
    dx = raw_end[0] - raw_start[0]
    dy = raw_end[1] - raw_start[1]
    first_projection = (first[0] - raw_start[0]) * dx + (first[1] - raw_start[1]) * dy
    second_projection = (second[0] - raw_start[0]) * dx + (second[1] - raw_start[1]) * dy
    if first_projection <= second_projection:
        return first, second

    return second, first


def draw_plane_route(draw: Any, entry: list[float], exit_point: list[float], transform: MapTransform) -> None:
    entry_px = world_to_pixel(entry[0], entry[1], transform)
    exit_px = world_to_pixel(exit_point[0], exit_point[1], transform)
    route_color = (235, 235, 235, 235)
    draw.line([entry_px, exit_px], fill=route_color, width=3)
    draw_entry_triangle(draw, entry_px, exit_px, route_color)
    draw.ellipse(
        (exit_px[0] - 6, exit_px[1] - 6, exit_px[0] + 6, exit_px[1] + 6),
        fill=(210, 210, 210, 235),
        outline=(255, 255, 255, 245),
        width=1,
    )


def draw_grid(draw: Any, transform: MapTransform, font: Any) -> None:
    column_labels = list("ABCDEFGH")
    row_labels = list("IJKLMNOP")
    grid_color = (0, 0, 0, 95)
    label_color = (245, 245, 245, 245)
    cell_width = transform.width / 8.0
    cell_height = transform.height / 8.0

    for index in range(9):
        x = transform.left + cell_width * index
        y = transform.top + cell_height * index
        draw.line((x, transform.top, x, transform.bottom), fill=grid_color, width=1)
        draw.line((transform.left, y, transform.right, y), fill=grid_color, width=1)

    for index, label in enumerate(column_labels):
        x = transform.left + cell_width * index + 14.0
        draw.text((x, transform.top + 7.0), label, fill=label_color, font=font, stroke_width=1, stroke_fill=(0, 0, 0, 210))

    for index, label in enumerate(row_labels):
        y = transform.top + cell_height * index + 18.0
        draw.text((transform.left + 5.0, y), label, fill=label_color, font=font, stroke_width=1, stroke_fill=(0, 0, 0, 210))


def draw_entry_triangle(
    draw: Any,
    entry_px: tuple[float, float],
    exit_px: tuple[float, float],
    color: tuple[int, int, int, int],
) -> None:
    dx = exit_px[0] - entry_px[0]
    dy = exit_px[1] - entry_px[1]
    length = math.hypot(dx, dy)
    if length == 0.0:
        return

    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    tip = (entry_px[0] + ux * 14.0, entry_px[1] + uy * 14.0)
    back = (entry_px[0] - ux * 8.0, entry_px[1] - uy * 8.0)
    left = (back[0] + px * 9.0, back[1] + py * 9.0)
    right = (back[0] - px * 9.0, back[1] - py * 9.0)
    draw.polygon([tip, left, right], fill=color)


def draw_circle(
    draw: Any,
    circle: list[float],
    transform: MapTransform,
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    x, y, radius = circle
    center_x, center_y = world_to_pixel(x, y, transform)
    pixel_radius_x = radius / transform.world_size * transform.width
    pixel_radius_y = radius / transform.world_size * transform.height
    draw.ellipse(
        (
            center_x - pixel_radius_x,
            center_y - pixel_radius_y,
            center_x + pixel_radius_x,
            center_y + pixel_radius_y,
        ),
        outline=color,
        width=width,
    )


def draw_error_line(draw: Any, predicted: list[float], actual: list[float], transform: MapTransform) -> None:
    predicted_px = world_to_pixel(predicted[0], predicted[1], transform)
    actual_px = world_to_pixel(actual[0], actual[1], transform)
    draw.line([predicted_px, actual_px], fill=(255, 255, 255, 220), width=2)


def draw_legend(
    draw: Any,
    row: dict[str, str],
    phase: int,
    prediction: list[float] | None,
    actual_next_circle: list[float],
    transform: MapTransform,
    font: Any,
) -> None:
    lines = [
        f"Match: {row['match_id']}",
        f"Route: {row.get('route_group', 'unknown')}",
        f"Phase: P{phase} -> P{phase + 1}",
        f"Visual world: {transform.world_size / 100000:.2f} km square",
    ]
    if prediction is not None:
        lines.append(f"Prediction center error: {center_error_m(prediction, actual_next_circle):.2f} m")

    text = "\n".join(lines)
    padding = 12
    line_height = 14
    box_width = 440
    box_height = padding * 2 + line_height * len(lines)
    draw.rounded_rectangle((12, 12, 12 + box_width, 12 + box_height), radius=8, fill=(0, 0, 0, 180))
    draw.multiline_text((24, 24), text, fill=(255, 255, 255, 255), font=font, spacing=3)


def center_error_m(predicted: list[float], actual: list[float]) -> float:
    dx = predicted[0] - actual[0]
    dy = predicted[1] - actual[1]
    return ((dx * dx + dy * dy) ** 0.5) / 100.0


def distance_squared(first: list[float], second: list[float]) -> float:
    dx = first[0] - second[0]
    dy = first[1] - second[1]
    return dx * dx + dy * dy


def resolve_output_path(output: str | None, match_id: str, phase: int) -> Path:
    if output:
        return Path(output)

    safe_match_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in match_id)
    return DEFAULT_OUTPUT_DIR / f"erangel_replay_style_{safe_match_id}_p{phase}.png"


if __name__ == "__main__":
    raise SystemExit(main())

