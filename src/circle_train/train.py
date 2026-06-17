from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from circle_train.config import DATA_DIR, PROCESSED_DIR, ensure_data_dirs


MAP_SIZE = 816000.0
DEFAULT_MODEL_PATH = DATA_DIR / "models" / "circle_transition_model.pkl"
DEFAULT_METRICS_PATH = DATA_DIR / "models" / "circle_transition_metrics.json"
TARGET_NAMES_BY_MODE = {
    "absolute": ["next_center_x", "next_center_y", "next_radius"],
    "displacement": ["next_delta_x", "next_delta_y", "next_radius"],
}


@dataclass(frozen=True)
class TransitionRow:
    match_id: str
    route_group: str
    phase: int
    features: list[float]
    target: list[float]
    current_circle: list[float]
    next_circle: list[float]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a PUBG Erangel circle transition model.")
    parser.add_argument(
        "--input",
        default=str(PROCESSED_DIR / "match_sequences.csv"),
        help="Path to match_sequences.csv.",
    )
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL_PATH), help="Model pickle output path.")
    parser.add_argument("--metrics-output", default=str(DEFAULT_METRICS_PATH), help="Metrics JSON output path.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Holdout ratio by match ID.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic train/test split seed.")
    parser.add_argument(
        "--algorithm",
        choices=["random-forest", "ridge"],
        default="random-forest",
        help="Model algorithm to train.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["displacement", "absolute"],
        default="displacement",
        help="Use next-center displacement or absolute next-center coordinates as the target.",
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Train one global transition model instead of one model per phase.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=0.01, help="Ridge regularization strength.")
    parser.add_argument("--rf-estimators", type=int, default=100, help="RandomForest tree count.")
    parser.add_argument("--rf-max-depth", type=int, default=12, help="RandomForest max tree depth.")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=5, help="RandomForest minimum leaf size.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs for scikit-learn models.")
    args = parser.parse_args()

    ensure_data_dirs()
    input_path = Path(args.input)
    model_output_path = Path(args.model_output)
    metrics_output_path = Path(args.metrics_output)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_transition_rows(input_path, target_mode=args.target_mode)
    if not rows:
        raise SystemExit(f"No trainable transition rows found: {input_path}")

    route_groups = sorted({row.route_group for row in rows})
    rows = read_transition_rows(input_path, route_groups, target_mode=args.target_mode)
    train_rows, test_rows = split_by_match(rows, args.test_ratio, args.seed)

    feature_names = build_feature_names(route_groups)
    trainer = build_trainer(args, len(feature_names))
    if args.single_model:
        global_model = trainer(train_rows)
        phase_models = {phase: global_model for phase in range(1, 9)}
        phase_scoped = False
    else:
        phase_models = train_phase_models(train_rows, trainer)
        phase_scoped = True

    model_type = build_model_type(args.algorithm, args.target_mode, phase_scoped)
    metrics = evaluate_model(phase_models, train_rows, test_rows, args.target_mode)
    metrics.update(
        {
            "input": str(input_path),
            "model_output": str(model_output_path),
            "metrics_output": str(metrics_output_path),
            "model_type": model_type,
            "algorithm": args.algorithm,
            "target_mode": args.target_mode,
            "phase_scoped": phase_scoped,
            "feature_count": len(feature_names),
            "route_group_count": len(route_groups),
            "ridge_alpha": args.ridge_alpha,
            "rf_estimators": args.rf_estimators,
            "rf_max_depth": args.rf_max_depth,
            "rf_min_samples_leaf": args.rf_min_samples_leaf,
            "seed": args.seed,
            "test_ratio": args.test_ratio,
        }
    )

    model = {
        "model_type": model_type,
        "algorithm": args.algorithm,
        "target_mode": args.target_mode,
        "map_size": MAP_SIZE,
        "target_names": TARGET_NAMES_BY_MODE[args.target_mode],
        "feature_names": feature_names,
        "route_groups": route_groups,
        "phase_models": phase_models,
        "metrics": metrics,
    }

    with model_output_path.open("wb") as file:
        pickle.dump(model, file)

    with metrics_output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
        file.write("\n")

    current_baseline = metrics["baselines"]["current_center"]
    print(f"Model type: {model_type}")
    print(f"Training rows: {metrics['train_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Baseline current-center MAE: {current_baseline['center_mae_m']:.2f} m")
    print(f"Train center MAE: {metrics['train_center_mae_m']:.2f} m")
    print(f"Test center MAE: {metrics['test_center_mae_m']:.2f} m")
    print(f"Test radius MAE: {metrics['test_radius_mae_m']:.2f} m")
    print(f"Saved model: {model_output_path}")
    print(f"Saved metrics: {metrics_output_path}")
    return 0


def build_model_type(algorithm: str, target_mode: str, phase_scoped: bool) -> str:
    scope = "phase" if phase_scoped else "global"
    normalized_algorithm = algorithm.replace("-", "_")
    return f"{scope}_{normalized_algorithm}_{target_mode}"


def read_transition_rows(
    input_path: Path,
    route_groups: list[str] | None = None,
    target_mode: str = "displacement",
) -> list[TransitionRow]:
    with input_path.open("r", newline="", encoding="utf-8-sig") as file:
        source_rows = list(csv.DictReader(file))

    if route_groups is None:
        route_groups = sorted({(row.get("route_group") or "unknown").strip() for row in source_rows})

    transitions: list[TransitionRow] = []
    for source_row in source_rows:
        for phase in range(1, 9):
            current_circle = read_circle(source_row, phase)
            next_circle = read_circle(source_row, phase + 1)
            if current_circle is None or next_circle is None:
                continue

            features = build_features(source_row, phase, current_circle, route_groups)
            target = build_target(current_circle, next_circle, target_mode)
            transitions.append(
                TransitionRow(
                    match_id=source_row["match_id"],
                    route_group=(source_row.get("route_group") or "unknown").strip(),
                    phase=phase,
                    features=features,
                    target=target,
                    current_circle=current_circle,
                    next_circle=next_circle,
                )
            )

    return transitions


def build_target(current_circle: list[float], next_circle: list[float], target_mode: str) -> list[float]:
    if target_mode == "absolute":
        return [next_circle[0] / MAP_SIZE, next_circle[1] / MAP_SIZE, next_circle[2] / MAP_SIZE]

    if target_mode == "displacement":
        return [
            (next_circle[0] - current_circle[0]) / MAP_SIZE,
            (next_circle[1] - current_circle[1]) / MAP_SIZE,
            next_circle[2] / MAP_SIZE,
        ]

    raise ValueError(f"Unsupported target mode: {target_mode}")


def read_circle(row: dict[str, str], phase: int) -> list[float] | None:
    values = [
        parse_float(row.get(f"p{phase}_center_x")),
        parse_float(row.get(f"p{phase}_center_y")),
        parse_float(row.get(f"p{phase}_radius")),
    ]
    if any(value is None for value in values):
        return None

    return [float(value) for value in values]


def build_features(
    row: dict[str, str],
    phase: int,
    current_circle: list[float],
    route_groups: list[str],
) -> list[float]:
    route_group = (row.get("route_group") or "unknown").strip()
    plane_start_x = parse_float(row.get("plane_start_x")) or 0.0
    plane_start_y = parse_float(row.get("plane_start_y")) or 0.0
    plane_end_x = parse_float(row.get("plane_end_x")) or 0.0
    plane_end_y = parse_float(row.get("plane_end_y")) or 0.0
    plane_angle = math.radians(parse_float(row.get("plane_angle")) or 0.0)
    plane_mid_x = (plane_start_x + plane_end_x) / 2.0
    plane_mid_y = (plane_start_y + plane_end_y) / 2.0
    current_x, current_y, current_radius = current_circle

    previous_dx = 0.0
    previous_dy = 0.0
    previous_shrink = 0.0
    previous_circle = read_circle(row, phase - 1) if phase > 1 else None
    if previous_circle is not None:
        previous_dx = (current_x - previous_circle[0]) / MAP_SIZE
        previous_dy = (current_y - previous_circle[1]) / MAP_SIZE
        previous_shrink = current_radius / previous_circle[2] if previous_circle[2] else 0.0

    features = [
        1.0,
        phase / 8.0,
        plane_start_x / MAP_SIZE,
        plane_start_y / MAP_SIZE,
        plane_end_x / MAP_SIZE,
        plane_end_y / MAP_SIZE,
        (plane_end_x - plane_start_x) / MAP_SIZE,
        (plane_end_y - plane_start_y) / MAP_SIZE,
        math.sin(plane_angle),
        math.cos(plane_angle),
        current_x / MAP_SIZE,
        current_y / MAP_SIZE,
        current_radius / MAP_SIZE,
        (current_x - plane_mid_x) / MAP_SIZE,
        (current_y - plane_mid_y) / MAP_SIZE,
        previous_dx,
        previous_dy,
        previous_shrink,
    ]
    features.extend(1.0 if route_group == group else 0.0 for group in route_groups)
    return features


def build_feature_names(route_groups: list[str]) -> list[str]:
    names = [
        "bias",
        "phase",
        "plane_start_x",
        "plane_start_y",
        "plane_end_x",
        "plane_end_y",
        "plane_dx",
        "plane_dy",
        "plane_angle_sin",
        "plane_angle_cos",
        "current_center_x",
        "current_center_y",
        "current_radius",
        "current_from_route_mid_x",
        "current_from_route_mid_y",
        "previous_dx",
        "previous_dy",
        "previous_shrink",
    ]
    names.extend(f"route_group={group}" for group in route_groups)
    return names


def split_by_match(
    rows: list[TransitionRow],
    test_ratio: float,
    seed: int,
) -> tuple[list[TransitionRow], list[TransitionRow]]:
    match_ids = sorted({row.match_id for row in rows})
    rng = random.Random(seed)
    rng.shuffle(match_ids)
    test_count = max(1, int(len(match_ids) * test_ratio))
    test_match_ids = set(match_ids[:test_count])
    train_rows = [row for row in rows if row.match_id not in test_match_ids]
    test_rows = [row for row in rows if row.match_id in test_match_ids]
    return train_rows, test_rows


def build_trainer(args: argparse.Namespace, feature_count: int) -> Callable[[list[TransitionRow]], Any]:
    if args.algorithm == "ridge":
        return lambda rows: train_ridge_model(rows, feature_count, args.ridge_alpha)

    if args.algorithm == "random-forest":
        return lambda rows: train_random_forest_model(
            rows=rows,
            estimator_count=args.rf_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            seed=args.seed,
            jobs=args.jobs,
        )

    raise ValueError(f"Unsupported algorithm: {args.algorithm}")


def train_phase_models(
    rows: list[TransitionRow],
    trainer: Callable[[list[TransitionRow]], Any],
) -> dict[int, Any]:
    phase_models: dict[int, Any] = {}
    for phase in range(1, 9):
        phase_rows = [row for row in rows if row.phase == phase]
        if not phase_rows:
            continue

        phase_models[phase] = trainer(phase_rows)

    return phase_models


def train_random_forest_model(
    rows: list[TransitionRow],
    estimator_count: int,
    max_depth: int | None,
    min_samples_leaf: int,
    seed: int,
    jobs: int,
) -> Any:
    try:
        from joblib import parallel_backend
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
    except ImportError as error:
        raise SystemExit("scikit-learn is required for --algorithm random-forest. Run: py -3 -m pip install scikit-learn") from error

    model = RandomForestRegressor(
        n_estimators=estimator_count,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=jobs,
    )
    features = np.asarray([row.features for row in rows], dtype=np.float32)
    targets = np.asarray([row.target for row in rows], dtype=np.float32)
    with parallel_backend("threading"):
        model.fit(features, targets)
    return model


def train_ridge_model(
    rows: list[TransitionRow],
    feature_count: int,
    ridge_alpha: float,
) -> list[list[float]]:
    weights_by_target: list[list[float]] = []
    for target_index in range(len(TARGET_NAMES_BY_MODE["absolute"])):
        normal_matrix = [[0.0 for _ in range(feature_count)] for _ in range(feature_count)]
        normal_vector = [0.0 for _ in range(feature_count)]

        for row in rows:
            x = row.features
            y = row.target[target_index]
            for i in range(feature_count):
                normal_vector[i] += x[i] * y
                for j in range(feature_count):
                    normal_matrix[i][j] += x[i] * x[j]

        for i in range(feature_count):
            normal_matrix[i][i] += ridge_alpha

        weights_by_target.append(solve_linear_system(normal_matrix, normal_vector))

    return weights_by_target


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(size)]

    for pivot_index in range(size):
        best_row = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
        if abs(augmented[best_row][pivot_index]) < 1e-12:
            continue

        if best_row != pivot_index:
            augmented[pivot_index], augmented[best_row] = augmented[best_row], augmented[pivot_index]

        pivot = augmented[pivot_index][pivot_index]
        for column_index in range(pivot_index, size + 1):
            augmented[pivot_index][column_index] /= pivot

        for row_index in range(size):
            if row_index == pivot_index:
                continue

            factor = augmented[row_index][pivot_index]
            if factor == 0.0:
                continue

            for column_index in range(pivot_index, size + 1):
                augmented[row_index][column_index] -= factor * augmented[pivot_index][column_index]

    return [augmented[i][size] for i in range(size)]


def evaluate_model(
    phase_models: dict[int, Any],
    train_rows: list[TransitionRow],
    test_rows: list[TransitionRow],
    target_mode: str,
) -> dict[str, Any]:
    train_metrics = evaluate_rows(phase_models, train_rows, target_mode)
    test_metrics = evaluate_rows(phase_models, test_rows, target_mode)
    baseline_metrics = evaluate_baselines(train_rows, test_rows)
    per_phase_metrics = {}
    for phase in range(1, 9):
        phase_test_rows = [row for row in test_rows if row.phase == phase]
        if phase_test_rows:
            per_phase_metrics[str(phase)] = evaluate_rows(phase_models, phase_test_rows, target_mode)

    return {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_matches": len({row.match_id for row in train_rows}),
        "test_matches": len({row.match_id for row in test_rows}),
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "baselines": baseline_metrics,
        "test_by_phase": per_phase_metrics,
    }


def evaluate_rows(
    phase_models: dict[int, Any],
    rows: list[TransitionRow],
    target_mode: str,
) -> dict[str, float]:
    return evaluate_prediction_function(
        rows,
        lambda row: predict_circle(phase_models, row.phase, row.features, target_mode, row.current_circle),
    )


def evaluate_baselines(
    train_rows: list[TransitionRow],
    test_rows: list[TransitionRow]) -> dict[str, dict[str, float]]:
    phase_means = build_phase_mean_predictions(train_rows)
    phase_route_means = build_phase_route_mean_predictions(train_rows)

    return {
        "current_center": evaluate_prediction_function(
            test_rows,
            lambda row: [row.current_circle[0], row.current_circle[1], row.current_circle[2]],
        ),
        "phase_mean_delta": evaluate_prediction_function(
            test_rows,
            lambda row: predict_from_mean(row, phase_means[row.phase]),
        ),
        "phase_route_mean_delta": evaluate_prediction_function(
            test_rows,
            lambda row: predict_from_mean(
                row,
                phase_route_means.get((row.phase, row.route_group), phase_means[row.phase]),
            ),
        ),
    }


def build_phase_mean_predictions(rows: list[TransitionRow]) -> dict[int, tuple[float, float, float]]:
    sums: dict[int, list[float]] = {}
    for row in rows:
        phase_sum = sums.setdefault(row.phase, [0.0, 0.0, 0.0, 0.0])
        phase_sum[0] += row.next_circle[0] - row.current_circle[0]
        phase_sum[1] += row.next_circle[1] - row.current_circle[1]
        phase_sum[2] += row.next_circle[2]
        phase_sum[3] += 1.0

    return {phase: (values[0] / values[3], values[1] / values[3], values[2] / values[3]) for phase, values in sums.items()}


def build_phase_route_mean_predictions(rows: list[TransitionRow]) -> dict[tuple[int, str], tuple[float, float, float]]:
    sums: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        key = (row.phase, row.route_group)
        route_sum = sums.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        route_sum[0] += row.next_circle[0] - row.current_circle[0]
        route_sum[1] += row.next_circle[1] - row.current_circle[1]
        route_sum[2] += row.next_circle[2]
        route_sum[3] += 1.0

    return {key: (values[0] / values[3], values[1] / values[3], values[2] / values[3]) for key, values in sums.items()}


def predict_from_mean(row: TransitionRow, mean_prediction: tuple[float, float, float]) -> list[float]:
    return [
        clamp(row.current_circle[0] + mean_prediction[0], 0.0, MAP_SIZE),
        clamp(row.current_circle[1] + mean_prediction[1], 0.0, MAP_SIZE),
        clamp(mean_prediction[2], 0.0, MAP_SIZE),
    ]


def evaluate_prediction_function(
    rows: list[TransitionRow],
    predict_fn: Callable[[TransitionRow], list[float]],
) -> dict[str, float]:
    if not rows:
        return {
            "center_mae_m": 0.0,
            "center_rmse_m": 0.0,
            "radius_mae_m": 0.0,
            "radius_rmse_m": 0.0,
        }

    center_errors: list[float] = []
    radius_errors: list[float] = []
    for row in rows:
        predicted_x, predicted_y, predicted_radius = predict_fn(row)
        actual_x, actual_y, actual_radius = row.next_circle

        center_errors.append(math.hypot(predicted_x - actual_x, predicted_y - actual_y) / 100.0)
        radius_errors.append(abs(predicted_radius - actual_radius) / 100.0)

    return {
        "center_mae_m": mean(center_errors),
        "center_rmse_m": root_mean_square(center_errors),
        "radius_mae_m": mean(radius_errors),
        "radius_rmse_m": root_mean_square(radius_errors),
    }


def predict_circle(
    phase_models: dict[int, Any],
    phase: int,
    features: list[float],
    target_mode: str,
    current_circle: list[float],
) -> list[float]:
    target_prediction = predict_target(phase_models, phase, features)
    return target_to_circle(target_prediction, target_mode, current_circle)


def predict_target(
    phase_models: dict[int, Any],
    phase: int,
    features: list[float],
) -> list[float]:
    phase_model = phase_models.get(phase)
    if phase_model is None:
        raise ValueError(f"No trained model for phase {phase}.")

    if isinstance(phase_model, list):
        return [sum(weight * feature for weight, feature in zip(weights, features, strict=True)) for weights in phase_model]

    prediction = phase_model.predict([features])[0]
    if hasattr(prediction, "tolist"):
        return [float(value) for value in prediction.tolist()]

    return [float(value) for value in prediction]


def target_to_circle(target_prediction: list[float], target_mode: str, current_circle: list[float]) -> list[float]:
    if target_mode == "absolute":
        predicted_x = target_prediction[0] * MAP_SIZE
        predicted_y = target_prediction[1] * MAP_SIZE
    elif target_mode == "displacement":
        predicted_x = current_circle[0] + target_prediction[0] * MAP_SIZE
        predicted_y = current_circle[1] + target_prediction[1] * MAP_SIZE
    else:
        raise ValueError(f"Unsupported target mode: {target_mode}")

    predicted_radius = target_prediction[2] * MAP_SIZE
    return [
        clamp(predicted_x, 0.0, MAP_SIZE),
        clamp(predicted_y, 0.0, MAP_SIZE),
        clamp(predicted_radius, 0.0, MAP_SIZE),
    ]


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None

    return float(value)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def root_mean_square(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
