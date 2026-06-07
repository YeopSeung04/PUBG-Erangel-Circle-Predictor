from __future__ import annotations

import argparse
import csv
import math
import sqlite3

from circle_train.config import PROCESSED_DIR, ensure_data_dirs, load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze collected PUBG circle sequence data.")
    parser.add_argument("--database", default=None, help="SQLite database path.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("vectors", help="Export phase movement vectors and shrink ratios.")
    subparsers.add_parser("route-summary", help="Export phase 1 summary by route group.")

    args = parser.parse_args()
    ensure_data_dirs()
    settings = load_settings(args.database)

    if args.command == "vectors":
        export_vectors(settings.database_path)
        print("Exported data/processed/phase_vectors.csv.")
        return 0

    if args.command == "route-summary":
        export_route_summary(settings.database_path)
        print("Exported data/processed/route_phase1_summary.csv.")
        return 0

    return 1


def export_vectors(database_path) -> None:
    output_path = PROCESSED_DIR / "phase_vectors.csv"
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                c1.match_id,
                p.route_group,
                c1.phase,
                c1.center_x,
                c1.center_y,
                c1.radius,
                c2.center_x,
                c2.center_y,
                c2.radius
            FROM circles c1
            JOIN circles c2
                ON c2.match_id = c1.match_id
                AND c2.phase = c1.phase + 1
            LEFT JOIN plane_routes p
                ON p.match_id = c1.match_id
            ORDER BY c1.match_id, c1.phase
            """
        ).fetchall()

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "match_id",
                "route_group",
                "phase",
                "dx",
                "dy",
                "distance",
                "angle",
                "shrink_ratio",
            ]
        )
        for row in rows:
            dx = row[6] - row[3]
            dy = row[7] - row[4]
            distance = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))
            shrink_ratio = row[8] / row[5] if row[5] else None
            writer.writerow([row[0], row[1], row[2], dx, dy, distance, angle, shrink_ratio])


def export_route_summary(database_path) -> None:
    output_path = PROCESSED_DIR / "route_phase1_summary.csv"
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                COALESCE(p.route_group, 'unknown') AS route_group,
                COUNT(*) AS match_count,
                AVG(c.center_x) AS avg_p1_center_x,
                AVG(c.center_y) AS avg_p1_center_y,
                AVG(c.radius) AS avg_p1_radius
            FROM circles c
            LEFT JOIN plane_routes p
                ON p.match_id = c.match_id
            WHERE c.phase = 1
            GROUP BY COALESCE(p.route_group, 'unknown')
            ORDER BY match_count DESC
            """
        ).fetchall()

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["route_group", "match_count", "avg_p1_center_x", "avg_p1_center_y", "avg_p1_radius"])
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
