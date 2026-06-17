from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from circle_train.config import DATA_DIR
from circle_train.train import build_features, predict_circle


DEFAULT_MODEL_PATH = DATA_DIR / "models" / "circle_transition_model.pkl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict the next PUBG Erangel circle from a trained model.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to a trained model pickle.")
    parser.add_argument("--phase", type=int, required=True, choices=range(1, 9), help="Current phase, 1 through 8.")
    parser.add_argument("--route-group", default="unknown", help="Plane route group label.")
    parser.add_argument("--plane-start-x", type=float, required=True)
    parser.add_argument("--plane-start-y", type=float, required=True)
    parser.add_argument("--plane-end-x", type=float, required=True)
    parser.add_argument("--plane-end-y", type=float, required=True)
    parser.add_argument("--plane-angle", type=float, default=0.0)
    parser.add_argument("--current-center-x", type=float, required=True)
    parser.add_argument("--current-center-y", type=float, required=True)
    parser.add_argument("--current-radius", type=float, required=True)
    parser.add_argument("--previous-center-x", type=float, default=None)
    parser.add_argument("--previous-center-y", type=float, default=None)
    parser.add_argument("--previous-radius", type=float, default=None)
    args = parser.parse_args()

    model_path = Path(args.model)
    with model_path.open("rb") as file:
        model = pickle.load(file)

    phase_models = model["phase_models"]
    route_groups = model["route_groups"]
    target_mode = model.get("target_mode", "absolute")
    source_row = {
        "route_group": args.route_group,
        "plane_start_x": str(args.plane_start_x),
        "plane_start_y": str(args.plane_start_y),
        "plane_end_x": str(args.plane_end_x),
        "plane_end_y": str(args.plane_end_y),
        "plane_angle": str(args.plane_angle),
    }

    if args.phase > 1 and has_previous_circle(args):
        previous_phase = args.phase - 1
        source_row[f"p{previous_phase}_center_x"] = str(args.previous_center_x)
        source_row[f"p{previous_phase}_center_y"] = str(args.previous_center_y)
        source_row[f"p{previous_phase}_radius"] = str(args.previous_radius)

    current_circle = [args.current_center_x, args.current_center_y, args.current_radius]
    features = build_features(source_row, args.phase, current_circle, route_groups)
    predicted_x, predicted_y, predicted_radius = predict_circle(
        phase_models,
        args.phase,
        features,
        target_mode,
        current_circle,
    )

    result = {
        "model": str(model_path),
        "model_type": model.get("model_type"),
        "algorithm": model.get("algorithm"),
        "target_mode": target_mode,
        "phase": args.phase,
        "next_phase": args.phase + 1,
        "prediction": {
            "center_x": predicted_x,
            "center_y": predicted_y,
            "radius": predicted_radius,
            "center_x_m": predicted_x / 100.0,
            "center_y_m": predicted_y / 100.0,
            "radius_m": predicted_radius / 100.0,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def has_previous_circle(args: argparse.Namespace) -> bool:
    return (
        args.previous_center_x is not None
        and args.previous_center_y is not None
        and args.previous_radius is not None
    )


if __name__ == "__main__":
    raise SystemExit(main())
