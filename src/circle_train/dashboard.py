from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from circle_train.config import DATA_DIR, PROCESSED_DIR
from circle_train.train import MAP_SIZE, clamp, parse_float
from circle_train.visualize import (
    DEFAULT_VISUAL_WORLD_SIZE,
    MapTransform,
    center_error_m,
    clip_infinite_line_to_world_square,
    default_map_path,
    draw_circle,
    draw_error_line,
    draw_grid,
    draw_plane_route,
    order_route_by_raw_direction,
    predict_next_circle,
    read_circle,
)


DEFAULT_METRICS_PATH = DATA_DIR / "models" / "circle_transition_metrics.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "dashboard" / "training_dashboard.html"
DEFAULT_MODEL_PATH = DATA_DIR / "models" / "circle_transition_model.pkl"
DEFAULT_SEQUENCES_PATH = PROCESSED_DIR / "match_sequences.csv"
PHASE1_PREDICTION_MAP_FILENAME = "phase1_prediction_map_latest.png"
TRANSITION_PREDICTION_MAP_FILENAME = "prediction_map_latest.png"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a self-contained model training dashboard.")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH), help="Path to training metrics JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Dashboard HTML output path.")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = read_json(metrics_path)
    summary = build_data_summary()
    phase1_prediction_visual = build_phase1_prediction_visualization(output_path.parent / PHASE1_PREDICTION_MAP_FILENAME)
    transition_prediction_visual = build_transition_prediction_visualization(
        output_path.parent / TRANSITION_PREDICTION_MAP_FILENAME
    )
    html_content = build_dashboard_html(
        metrics,
        summary,
        metrics_path,
        phase1_prediction_visual,
        transition_prediction_visual,
    )
    output_path.write_text(html_content, encoding="utf-8", newline="\n")

    print(f"Saved dashboard: {output_path}")
    return 0


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_data_summary() -> dict:
    matches_path = PROCESSED_DIR / "matches.csv"
    sequences_path = PROCESSED_DIR / "match_sequences.csv"
    circles_path = PROCESSED_DIR / "circles.csv"
    plane_routes_path = PROCESSED_DIR / "plane_routes.csv"
    phase_vectors_path = PROCESSED_DIR / "phase_vectors.csv"
    route_summary_path = PROCESSED_DIR / "route_phase1_summary.csv"

    match_rows = read_csv_rows(matches_path)
    sequence_rows = read_csv_rows(sequences_path)
    circle_rows = read_csv_rows(circles_path)
    plane_route_rows = read_csv_rows(plane_routes_path)
    phase_vector_rows = read_csv_rows(phase_vectors_path)
    route_summary_rows = read_csv_rows(route_summary_path)

    date_counts = Counter()
    shard_counts = Counter()
    mode_counts = Counter()
    for row in match_rows:
        created_at = row.get("created_at", "")
        date = created_at[:10] if len(created_at) >= 10 else "unknown"
        date_counts[date] += 1
        shard_counts[row.get("shard_id", "unknown")] += 1
        mode_counts[row.get("game_mode", "unknown")] += 1

    route_counts = Counter(row.get("route_group", "unknown") for row in plane_route_rows)
    phase_counts = Counter(row.get("phase", "unknown") for row in circle_rows)
    transition_phase_counts = Counter(row.get("phase", "unknown") for row in phase_vector_rows)

    top_route_phase1 = []
    for row in route_summary_rows:
        top_route_phase1.append(
            {
                "route_group": row.get("route_group", "unknown"),
                "match_count": int_float(row.get("match_count")),
                "avg_p1_center_x": int_float(row.get("avg_p1_center_x")),
                "avg_p1_center_y": int_float(row.get("avg_p1_center_y")),
                "avg_p1_radius": int_float(row.get("avg_p1_radius")),
            }
        )
    top_route_phase1.sort(key=lambda item: item["match_count"], reverse=True)

    return {
        "matches": len(match_rows),
        "match_sequences": len(sequence_rows),
        "circles": len(circle_rows),
        "plane_routes": len(plane_route_rows),
        "phase_vectors": len(phase_vector_rows),
        "date_counts": sorted(date_counts.items()),
        "shard_counts": shard_counts.most_common(),
        "mode_counts": mode_counts.most_common(),
        "route_counts": route_counts.most_common(),
        "phase_counts": sorted_counter_by_int_key(phase_counts),
        "transition_phase_counts": sorted_counter_by_int_key(transition_phase_counts),
        "top_route_phase1": top_route_phase1[:10],
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def int_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def sorted_counter_by_int_key(counter: Counter) -> list[tuple[str, int]]:
    def key_value(item: tuple[str, int]) -> tuple[int, str]:
        key, _ = item
        try:
            return int(key), key
        except ValueError:
            return 9999, key

    return sorted(counter.items(), key=key_value)


def build_phase1_prediction_visualization(output_path: Path) -> dict[str, Any] | None:
    if not DEFAULT_SEQUENCES_PATH.exists():
        return None

    row = select_latest_valid_phase1_row(DEFAULT_SEQUENCES_PATH)
    if row is None:
        return None

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    prediction = predict_phase1_circle(DEFAULT_SEQUENCES_PATH, row)
    if prediction is None:
        return None

    actual_phase1_circle = read_circle(row, 1)
    if actual_phase1_circle is None:
        return None

    map_path = default_map_path()
    image = Image.open(map_path).convert("RGBA")
    transform = MapTransform(
        world_size=DEFAULT_VISUAL_WORLD_SIZE,
        left=0.0,
        top=0.0,
        right=float(image.size[0]),
        bottom=float(image.size[1]),
    )

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    draw_grid(draw, transform, font)

    plane_start = [float(row["plane_start_x"]), float(row["plane_start_y"])]
    plane_end = [float(row["plane_end_x"]), float(row["plane_end_y"])]
    map_route = clip_infinite_line_to_world_square(plane_start, plane_end, transform.world_size)
    if map_route is not None:
        plane_entry, plane_exit = order_route_by_raw_direction(map_route, plane_start, plane_end)
        draw_plane_route(draw, plane_entry, plane_exit, transform)
    else:
        plane_entry = None
        plane_exit = None

    error_m = center_error_m(prediction, actual_phase1_circle)
    draw_circle(draw, actual_phase1_circle, transform, (0, 210, 255, 235), width=3)
    draw_circle(draw, prediction, transform, (255, 50, 80, 240), width=4)
    draw_error_line(draw, prediction, actual_phase1_circle, transform)
    draw_phase1_prediction_legend(
        draw,
        row=row,
        error_m=error_m,
        predicted_circle=prediction,
        actual_circle=actual_phase1_circle,
        font=font,
    )

    rendered = Image.alpha_composite(image, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)

    return {
        "image_src": output_path.name,
        "image_path": str(output_path),
        "match_id": row["match_id"],
        "route_group": row.get("route_group", "unknown"),
        "phase": 1,
        "center_error_m": error_m,
        "actual_circle": actual_phase1_circle,
        "predicted_circle": prediction,
        "plane_entry": plane_entry,
        "plane_exit": plane_exit,
        "model_type": "phase1_random_forest_route_only",
    }


def predict_phase1_circle(input_path: Path, target_row: dict[str, str]) -> list[float] | None:
    source_rows = read_csv_rows(input_path)
    valid_rows = [row for row in source_rows if is_valid_phase1_row(row)]
    if len(valid_rows) < 2:
        return None

    route_groups = sorted({(row.get("route_group") or "unknown").strip() for row in valid_rows})
    train_rows = [row for row in valid_rows if row.get("match_id") != target_row.get("match_id")]
    if not train_rows:
        train_rows = valid_rows

    try:
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return predict_phase1_circle_by_route_mean(train_rows, target_row)

    features = np.asarray([build_phase1_features(row, route_groups) for row in train_rows], dtype=np.float32)
    targets = np.asarray([build_phase1_target(row) for row in train_rows], dtype=np.float32)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
    )
    model.fit(features, targets)
    prediction = model.predict([build_phase1_features(target_row, route_groups)])[0]
    if hasattr(prediction, "tolist"):
        prediction = prediction.tolist()

    return [
        clamp(float(prediction[0]) * MAP_SIZE, 0.0, MAP_SIZE),
        clamp(float(prediction[1]) * MAP_SIZE, 0.0, MAP_SIZE),
        clamp(float(prediction[2]) * MAP_SIZE, 0.0, MAP_SIZE),
    ]


def predict_phase1_circle_by_route_mean(train_rows: list[dict[str, str]], target_row: dict[str, str]) -> list[float] | None:
    target_route_group = (target_row.get("route_group") or "unknown").strip()
    route_rows = [row for row in train_rows if (row.get("route_group") or "unknown").strip() == target_route_group]
    candidate_rows = route_rows if route_rows else train_rows
    if not candidate_rows:
        return None

    sums = [0.0, 0.0, 0.0]
    for row in candidate_rows:
        circle = read_circle(row, 1)
        if circle is None:
            continue
        sums[0] += circle[0]
        sums[1] += circle[1]
        sums[2] += circle[2]

    count = len(candidate_rows)
    return [value / count for value in sums]


def select_latest_valid_phase1_row(input_path: Path) -> dict[str, str] | None:
    rows = read_csv_rows(input_path)
    for row in reversed(rows):
        if is_valid_phase1_row(row):
            return row

    return None


def is_valid_phase1_row(row: dict[str, str]) -> bool:
    required_keys = [
        "plane_start_x",
        "plane_start_y",
        "plane_end_x",
        "plane_end_y",
        "p1_center_x",
        "p1_center_y",
        "p1_radius",
    ]
    return all(row.get(key) not in (None, "") for key in required_keys)


def build_phase1_features(row: dict[str, str], route_groups: list[str]) -> list[float]:
    route_group = (row.get("route_group") or "unknown").strip()
    plane_start_x = parse_float(row.get("plane_start_x")) or 0.0
    plane_start_y = parse_float(row.get("plane_start_y")) or 0.0
    plane_end_x = parse_float(row.get("plane_end_x")) or 0.0
    plane_end_y = parse_float(row.get("plane_end_y")) or 0.0
    plane_angle = math.radians(parse_float(row.get("plane_angle")) or 0.0)
    plane_dx = plane_end_x - plane_start_x
    plane_dy = plane_end_y - plane_start_y
    plane_mid_x = (plane_start_x + plane_end_x) / 2.0
    plane_mid_y = (plane_start_y + plane_end_y) / 2.0
    plane_length = math.hypot(plane_dx, plane_dy)

    features = [
        1.0,
        plane_start_x / MAP_SIZE,
        plane_start_y / MAP_SIZE,
        plane_end_x / MAP_SIZE,
        plane_end_y / MAP_SIZE,
        plane_dx / MAP_SIZE,
        plane_dy / MAP_SIZE,
        plane_mid_x / MAP_SIZE,
        plane_mid_y / MAP_SIZE,
        plane_length / MAP_SIZE,
        math.sin(plane_angle),
        math.cos(plane_angle),
    ]
    features.extend(1.0 if route_group == group else 0.0 for group in route_groups)
    return features


def build_phase1_target(row: dict[str, str]) -> list[float]:
    circle = read_circle(row, 1)
    if circle is None:
        return [0.0, 0.0, 0.0]

    return [circle[0] / MAP_SIZE, circle[1] / MAP_SIZE, circle[2] / MAP_SIZE]


def draw_phase1_prediction_legend(
    draw: Any,
    row: dict[str, str],
    error_m: float,
    predicted_circle: list[float],
    actual_circle: list[float],
    font: Any,
) -> None:
    lines = [
        "Phase 1 prediction map",
        f"Match: {row['match_id']}",
        f"Route: {row.get('route_group', 'unknown')}",
        "Input: plane route only",
        "Cyan: actual P1 circle",
        "Red: predicted P1 circle",
        f"Center error: {error_m:.2f} m",
        f"Pred radius: {predicted_circle[2] / 100.0:.2f} m",
        f"Actual radius: {actual_circle[2] / 100.0:.2f} m",
    ]
    padding = 12
    line_height = 14
    box_width = 430
    box_height = padding * 2 + line_height * len(lines) + 6
    draw.rounded_rectangle((12, 12, 12 + box_width, 12 + box_height), radius=10, fill=(0, 0, 0, 188))
    draw.multiline_text(
        (24, 24),
        "\n".join(lines),
        fill=(255, 255, 255, 255),
        font=font,
        spacing=3,
    )


def build_transition_prediction_visualization(output_path: Path, phase: int = 1) -> dict[str, Any] | None:
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_SEQUENCES_PATH.exists():
        return None

    row = select_latest_valid_prediction_row(DEFAULT_SEQUENCES_PATH, phase)
    if row is None:
        return None

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    map_path = default_map_path()
    image = Image.open(map_path).convert("RGBA")
    transform = MapTransform(
        world_size=DEFAULT_VISUAL_WORLD_SIZE,
        left=0.0,
        top=0.0,
        right=float(image.size[0]),
        bottom=float(image.size[1]),
    )

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    draw_grid(draw, transform, font)

    plane_start = [float(row["plane_start_x"]), float(row["plane_start_y"])]
    plane_end = [float(row["plane_end_x"]), float(row["plane_end_y"])]
    map_route = clip_infinite_line_to_world_square(plane_start, plane_end, transform.world_size)
    if map_route is not None:
        plane_entry, plane_exit = order_route_by_raw_direction(map_route, plane_start, plane_end)
        draw_plane_route(draw, plane_entry, plane_exit, transform)
    else:
        plane_entry = None
        plane_exit = None

    current_circle = read_circle(row, phase)
    actual_next_circle = read_circle(row, phase + 1)
    if current_circle is None or actual_next_circle is None:
        return None

    predicted_next_circle = predict_next_circle(DEFAULT_MODEL_PATH, row, phase, current_circle)
    error_m = center_error_m(predicted_next_circle, actual_next_circle)

    draw_circle(draw, current_circle, transform, (245, 245, 245, 235), width=3)
    draw_circle(draw, actual_next_circle, transform, (0, 210, 255, 235), width=3)
    draw_circle(draw, predicted_next_circle, transform, (255, 50, 80, 240), width=4)
    draw_error_line(draw, predicted_next_circle, actual_next_circle, transform)
    draw_prediction_legend(
        draw,
        row=row,
        phase=phase,
        error_m=error_m,
        predicted_next_circle=predicted_next_circle,
        actual_next_circle=actual_next_circle,
        font=font,
    )

    rendered = Image.alpha_composite(image, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)

    return {
        "image_src": output_path.name,
        "image_path": str(output_path),
        "match_id": row["match_id"],
        "route_group": row.get("route_group", "unknown"),
        "phase": phase,
        "next_phase": phase + 1,
        "center_error_m": error_m,
        "current_circle": current_circle,
        "actual_next_circle": actual_next_circle,
        "predicted_next_circle": predicted_next_circle,
        "plane_entry": plane_entry,
        "plane_exit": plane_exit,
    }


def select_latest_valid_prediction_row(input_path: Path, phase: int) -> dict[str, str] | None:
    rows = read_csv_rows(input_path)
    required_keys = [
        "plane_start_x",
        "plane_start_y",
        "plane_end_x",
        "plane_end_y",
        f"p{phase}_center_x",
        f"p{phase}_center_y",
        f"p{phase}_radius",
        f"p{phase + 1}_center_x",
        f"p{phase + 1}_center_y",
        f"p{phase + 1}_radius",
    ]

    for row in reversed(rows):
        if all(row.get(key) not in (None, "") for key in required_keys):
            return row

    return None


def draw_prediction_legend(
    draw: Any,
    row: dict[str, str],
    phase: int,
    error_m: float,
    predicted_next_circle: list[float],
    actual_next_circle: list[float],
    font: Any,
) -> None:
    lines = [
        "Prediction map",
        f"Match: {row['match_id']}",
        f"Route: {row.get('route_group', 'unknown')}",
        f"Phase: P{phase} -> P{phase + 1}",
        "White: current circle",
        "Cyan: actual next circle",
        "Red: predicted next circle",
        f"Center error: {error_m:.2f} m",
        f"Pred radius: {predicted_next_circle[2] / 100.0:.2f} m",
        f"Actual radius: {actual_next_circle[2] / 100.0:.2f} m",
    ]
    padding = 12
    line_height = 14
    box_width = 430
    box_height = padding * 2 + line_height * len(lines) + 6
    draw.rounded_rectangle((12, 12, 12 + box_width, 12 + box_height), radius=10, fill=(0, 0, 0, 188))
    draw.multiline_text(
        (24, 24),
        "\n".join(lines),
        fill=(255, 255, 255, 255),
        font=font,
        spacing=3,
    )


def build_dashboard_html(
    metrics: dict,
    summary: dict,
    metrics_path: Path,
    phase1_prediction_visual: dict[str, Any] | None,
    transition_prediction_visual: dict[str, Any] | None,
) -> str:
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    model_type = metrics.get("model_type", "unknown")
    algorithm = metrics.get("algorithm", "unknown")
    target_mode = metrics.get("target_mode", "unknown")

    current_baseline = metrics["baselines"]["current_center"]
    phase_mean_baseline = metrics["baselines"]["phase_mean_delta"]
    phase_route_baseline = metrics["baselines"]["phase_route_mean_delta"]

    center_improvement = current_baseline["center_mae_m"] - metrics["test_center_mae_m"]
    radius_improvement = current_baseline["radius_mae_m"] - metrics["test_radius_mae_m"]
    phase_mean_radius_improvement = phase_mean_baseline["radius_mae_m"] - metrics["test_radius_mae_m"]

    per_phase = metrics.get("test_by_phase", {})
    center_by_phase = [(f"P{phase}", values["center_mae_m"]) for phase, values in per_phase.items()]
    radius_by_phase = [(f"P{phase}", values["radius_mae_m"]) for phase, values in per_phase.items()]
    transition_phase_counts = [(f"P{phase}", count) for phase, count in summary["transition_phase_counts"]]

    route_items = [(route, count) for route, count in summary["route_counts"][:10]]
    collection_items = [(date, count) for date, count in summary["date_counts"][-14:]]
    interpretation_note = build_interpretation_note(
        algorithm=algorithm,
        center_mae=metrics["test_center_mae_m"],
        radius_mae=metrics["test_radius_mae_m"],
    )
    phase1_prediction_map_section = build_phase1_prediction_map_section(phase1_prediction_visual)
    transition_prediction_map_section = build_transition_prediction_map_section(transition_prediction_visual)

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PUBG Erangel Circle Training Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1120;
      --panel: #111827;
      --panel-2: #162033;
      --line: #25324a;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --green: #34d399;
      --blue: #60a5fa;
      --orange: #fb923c;
      --red: #f87171;
      --purple: #a78bfa;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #1f3a5f 0, var(--bg) 34rem);
      color: var(--text);
      font-family: "Segoe UI", "Noto Sans KR", Arial, sans-serif;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-start;
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      letter-spacing: -0.04em;
    }}
    h2 {{
      margin: 0 0 16px;
      font-size: 20px;
    }}
    p {{ color: var(--muted); line-height: 1.6; }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(17, 24, 39, 0.72);
      color: var(--muted);
      font-size: 13px;
      margin: 4px 4px 0 0;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      gap: 16px;
    }}
    .cards {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin-bottom: 16px;
    }}
    .two {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .three {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .panel {{
      background: linear-gradient(180deg, rgba(22, 32, 51, 0.92), rgba(17, 24, 39, 0.94));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.22);
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 30px;
      font-weight: 800;
      letter-spacing: -0.04em;
    }}
    .metric-sub {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .good {{ color: var(--green); }}
    .bad {{ color: var(--red); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: right;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .note {{
      border-left: 4px solid var(--orange);
      padding: 12px 14px;
      background: rgba(251, 146, 60, 0.08);
      border-radius: 12px;
      color: #fed7aa;
      line-height: 1.55;
    }}
    code {{
      color: #bfdbfe;
      background: rgba(96, 165, 250, 0.1);
      padding: 2px 5px;
      border-radius: 6px;
    }}
    .map-wrap {{
      overflow: hidden;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #020617;
    }}
    .map-image {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .legend-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 10px 0 16px;
    }}
    .legend-dot {{
      display: inline-block;
      width: 11px;
      height: 11px;
      border-radius: 999px;
      margin-right: 6px;
      vertical-align: -1px;
    }}
    @media (max-width: 900px) {{
      main {{ padding: 18px; }}
      header {{ flex-direction: column; }}
      .cards, .two, .three {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <section>
      <h1>PUBG Erangel Circle Training Dashboard</h1>
      <p>현재까지 수집한 에란겔 경기 데이터로 학습한 자기장 transition 모델 결과입니다.</p>
      <span class="pill">generated: {escape(generated_at)}</span>
      <span class="pill">metrics: {escape(str(metrics_path))}</span>
      <span class="pill">model: {escape(model_type)}</span>
      <span class="pill">algorithm: {escape(algorithm)}</span>
      <span class="pill">target: {escape(target_mode)}</span>
    </section>
  </header>

  <section class="grid cards">
    {metric_card("Matches", f"{summary['matches']:,}", "수집된 경기 수")}
    {metric_card("Transition Rows", f"{summary['phase_vectors']:,}", "학습 가능한 phase 이동 row")}
    {metric_card("Test Center MAE", f"{metrics['test_center_mae_m']:.2f} m", metric_delta_text(center_improvement, "current-center baseline 대비"))}
    {metric_card("Test Radius MAE", f"{metrics['test_radius_mae_m']:.2f} m", metric_delta_text(radius_improvement, "current-radius baseline 대비"))}
  </section>

  <section class="grid three">
    {metric_card("Train Rows", f"{metrics['train_rows']:,}", f"{metrics['train_matches']:,} matches")}
    {metric_card("Test Rows", f"{metrics['test_rows']:,}", f"{metrics['test_matches']:,} matches")}
    {metric_card("Route Groups", f"{metrics.get('route_group_count', 0):,}", f"features: {metrics.get('feature_count', 0):,}")}
  </section>

  <section class="grid two" style="margin-top: 16px;">
    <article class="panel">
      <h2>Center Error by Phase</h2>
      {bar_chart(center_by_phase, "m", "#60a5fa", value_decimals=1)}
    </article>
    <article class="panel">
      <h2>Radius Error by Phase</h2>
      {bar_chart(radius_by_phase, "m", "#34d399", value_decimals=2)}
    </article>
  </section>

  <section class="grid two" style="margin-top: 16px;">
    <article class="panel">
      <h2>Baseline Comparison</h2>
      <table>
        <thead>
          <tr>
            <th>Model / Baseline</th>
            <th>Center MAE</th>
            <th>Radius MAE</th>
          </tr>
        </thead>
        <tbody>
          {baseline_row("Current model", metrics["test_center_mae_m"], metrics["test_radius_mae_m"])}
          {baseline_row("Current center/radius", current_baseline["center_mae_m"], current_baseline["radius_mae_m"])}
          {baseline_row("Phase mean delta", phase_mean_baseline["center_mae_m"], phase_mean_baseline["radius_mae_m"])}
          {baseline_row("Phase + route mean delta", phase_route_baseline["center_mae_m"], phase_route_baseline["radius_mae_m"])}
        </tbody>
      </table>
      <p>
        반경 예측은 단순 현재 반경 유지 대비 {radius_improvement:.2f}m 개선,
        phase 평균 반경 기준 대비 {phase_mean_radius_improvement:.2f}m 개선입니다.
        중심점 예측은 아직 baseline과 거의 같은 수준이라 feature/비행기 동선 품질 개선이 필요합니다.
      </p>
    </article>
    <article class="panel">
      <h2>Transition Rows by Phase</h2>
      {bar_chart(transition_phase_counts, "rows", "#a78bfa", value_decimals=0)}
    </article>
  </section>

  <section class="grid two" style="margin-top: 16px;">
    <article class="panel">
      <h2>Top Plane Route Groups</h2>
      {bar_chart(route_items, "matches", "#fb923c", value_decimals=0)}
    </article>
    <article class="panel">
      <h2>Collection by Match Date</h2>
      {bar_chart(collection_items, "matches", "#60a5fa", value_decimals=0)}
    </article>
  </section>

  <section class="panel" style="margin-top: 16px;">
    <h2>Top Route Group P1 Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Route group</th>
          <th>Matches</th>
          <th>Avg P1 X</th>
          <th>Avg P1 Y</th>
          <th>Avg P1 Radius</th>
        </tr>
      </thead>
      <tbody>
        {route_summary_rows(summary["top_route_phase1"])}
      </tbody>
    </table>
  </section>

  <section class="panel" style="margin-top: 16px;">
    <h2>해석</h2>
    <div class="note">
      {interpretation_note}
    </div>
  </section>

  {phase1_prediction_map_section}

  {transition_prediction_map_section}
</main>
</body>
</html>
"""


def build_phase1_prediction_map_section(prediction_visual: dict[str, Any] | None) -> str:
    if prediction_visual is None:
        return """
        <section class="panel" style="margin-top: 16px;">
          <h2>Phase 1 Prediction Map</h2>
          <p>1페이즈 예측 지도 이미지를 생성하지 못했습니다. match_sequences.csv를 확인하세요.</p>
        </section>
        """

    predicted = prediction_visual["predicted_circle"]
    actual = prediction_visual["actual_circle"]
    radius_error_m = abs(predicted[2] - actual[2]) / 100.0

    return f"""
    <section class="panel" style="margin-top: 16px;">
      <h2>Phase 1 Prediction Map - 비행기 경로만 보고 1페이즈 예측</h2>
      <p>
        비행기 경로/route group만 입력으로 사용해서 P1 첫 자기장 위치를 예측하고,
        실제 P1 서클과 지도 위에서 비교했습니다.
      </p>
      <div class="legend-row">
        <span class="pill"><span class="legend-dot" style="background:#00d2ff;"></span>실제 P1 서클</span>
        <span class="pill"><span class="legend-dot" style="background:#ff3250;"></span>예측 P1 서클</span>
        <span class="pill">중심 오차: {prediction_visual["center_error_m"]:.2f} m</span>
        <span class="pill">반경 오차: {radius_error_m:.2f} m</span>
        <span class="pill">model: {escape(prediction_visual["model_type"])}</span>
        <span class="pill">route: {escape(prediction_visual["route_group"])}</span>
      </div>
      <div class="map-wrap">
        <img class="map-image" src="{escape(prediction_visual["image_src"])}" alt="Erangel phase 1 prediction result">
      </div>
      <p>
        Match ID: <code>{escape(prediction_visual["match_id"])}</code><br>
        빨간 원이 예측한 1페이즈, 하늘색 원이 실제 1페이즈입니다.
      </p>
    </section>
    """


def build_transition_prediction_map_section(prediction_visual: dict[str, Any] | None) -> str:
    if prediction_visual is None:
        return """
        <section class="panel" style="margin-top: 16px;">
          <h2>Transition Prediction Map</h2>
          <p>예측 지도 이미지를 생성하지 못했습니다. 모델 파일과 match_sequences.csv를 확인하세요.</p>
        </section>
        """

    predicted = prediction_visual["predicted_next_circle"]
    actual = prediction_visual["actual_next_circle"]
    radius_error_m = abs(predicted[2] - actual[2]) / 100.0

    return f"""
    <section class="panel" style="margin-top: 16px;">
      <h2>Transition Prediction Map - P{prediction_visual["phase"]}에서 P{prediction_visual["next_phase"]} 예측</h2>
      <p>
        최신 유효 match 하나를 골라 P{prediction_visual["phase"]} 입력 기준으로
        P{prediction_visual["next_phase"]} 자기장을 예측하고, 에란겔 지도 위에 표시했습니다.
      </p>
      <div class="legend-row">
        <span class="pill"><span class="legend-dot" style="background:#f8fafc;"></span>현재 서클</span>
        <span class="pill"><span class="legend-dot" style="background:#00d2ff;"></span>실제 다음 서클</span>
        <span class="pill"><span class="legend-dot" style="background:#ff3250;"></span>예측 다음 서클</span>
        <span class="pill">중심 오차: {prediction_visual["center_error_m"]:.2f} m</span>
        <span class="pill">반경 오차: {radius_error_m:.2f} m</span>
        <span class="pill">route: {escape(prediction_visual["route_group"])}</span>
      </div>
      <div class="map-wrap">
        <img class="map-image" src="{escape(prediction_visual["image_src"])}" alt="Erangel map prediction result">
      </div>
      <p>
        Match ID: <code>{escape(prediction_visual["match_id"])}</code><br>
        예측 원은 빨간색, 실제 다음 원은 하늘색입니다. 두 중심을 잇는 흰 선이 중심 오차입니다.
      </p>
    </section>
    """


def build_interpretation_note(algorithm: str, center_mae: float, radius_mae: float) -> str:
    algorithm_note = (
        "현재 결과는 <code>random-forest</code> 모델 기준입니다. "
        "Windows 로컬 학습에서는 <code>--jobs -1</code>보다 <code>--jobs 1</code>이 안정적으로 완료되었습니다."
        if algorithm == "random-forest"
        else f"현재 결과는 <code>{escape(algorithm)}</code> 모델 기준입니다."
    )
    return (
        f"{algorithm_note} "
        f"중심 좌표 MAE는 약 {center_mae:.2f}m, 반경 MAE는 약 {radius_mae:.2f}m입니다. "
        "반경은 단순 baseline보다 개선됐지만 중심 좌표 MAE는 baseline과 차이가 작습니다. "
        "다음 개선 포인트는 비행기 동선 추정 로직 정제, route feature 보강, P1 기준 위치/섬 여부/육지 마스크 같은 공간 feature 추가입니다."
    )


def metric_card(label: str, value: str, subtext: str) -> str:
    return f"""
    <article class="panel">
      <div class="metric-label">{escape(label)}</div>
      <div class="metric-value">{escape(value)}</div>
      <div class="metric-sub">{subtext}</div>
    </article>
    """


def metric_delta_text(delta: float, suffix: str) -> str:
    css_class = "good" if delta >= 0 else "bad"
    sign = "+" if delta >= 0 else ""
    return f'<span class="{css_class}">{sign}{delta:.2f} m</span> {escape(suffix)}'


def baseline_row(label: str, center_mae: float, radius_mae: float) -> str:
    return f"""
    <tr>
      <td>{escape(label)}</td>
      <td>{center_mae:.2f} m</td>
      <td>{radius_mae:.2f} m</td>
    </tr>
    """


def route_summary_rows(rows: Iterable[dict]) -> str:
    rendered_rows = []
    for row in rows:
        rendered_rows.append(
            f"""
            <tr>
              <td>{escape(row["route_group"])}</td>
              <td>{row["match_count"]:,.0f}</td>
              <td>{row["avg_p1_center_x"]:,.0f}</td>
              <td>{row["avg_p1_center_y"]:,.0f}</td>
              <td>{row["avg_p1_radius"]:,.0f}</td>
            </tr>
            """
        )
    return "\n".join(rendered_rows)


def bar_chart(items: list[tuple[str, float | int]], unit: str, color: str, value_decimals: int) -> str:
    if not items:
        return "<p>No data</p>"

    width = 860
    row_height = 34
    left = 210
    right = 110
    top = 18
    bar_height = 18
    height = top * 2 + len(items) * row_height
    max_value = max(float(value) for _, value in items) or 1.0
    bar_width = width - left - right

    rows = []
    for index, (label, raw_value) in enumerate(items):
        value = float(raw_value)
        y = top + index * row_height
        length = 0 if max_value == 0 else max(1.0, (value / max_value) * bar_width)
        value_text = f"{value:,.{value_decimals}f} {unit}"
        rows.append(
            f"""
            <text x="0" y="{y + 15}" fill="#cbd5e1" font-size="13">{escape(label)}</text>
            <rect x="{left}" y="{y}" width="{bar_width}" height="{bar_height}" rx="8" fill="rgba(148,163,184,0.13)" />
            <rect x="{left}" y="{y}" width="{length:.2f}" height="{bar_height}" rx="8" fill="{color}" />
            <text x="{left + bar_width + 12}" y="{y + 14}" fill="#e5e7eb" font-size="12">{escape(value_text)}</text>
            """
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="bar chart">
      {''.join(rows)}
    </svg>
    """


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
