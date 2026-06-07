from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from circle_train.config import PROCESSED_DIR
from circle_train.telemetry_parser import CircleRecord, MatchRecord, PlaneRouteRecord


class CircleTrainStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    shard_id TEXT NOT NULL,
                    map_name TEXT NOT NULL,
                    game_mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    telemetry_url TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plane_routes (
                    match_id TEXT PRIMARY KEY,
                    start_x REAL NOT NULL,
                    start_y REAL NOT NULL,
                    end_x REAL NOT NULL,
                    end_y REAL NOT NULL,
                    angle REAL NOT NULL,
                    route_group TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    FOREIGN KEY (match_id) REFERENCES matches(match_id)
                );

                CREATE TABLE IF NOT EXISTS circles (
                    match_id TEXT NOT NULL,
                    phase INTEGER NOT NULL,
                    center_x REAL NOT NULL,
                    center_y REAL NOT NULL,
                    radius REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    elapsed_time REAL,
                    PRIMARY KEY (match_id, phase),
                    FOREIGN KEY (match_id) REFERENCES matches(match_id)
                );
                """
            )

    def has_match(self, match_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM matches WHERE match_id = ?",
                (match_id,),
            ).fetchone()
            return row is not None

    def save_match_sequence(
        self,
        match: MatchRecord,
        plane_route: PlaneRouteRecord | None,
        circles: list[CircleRecord],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO matches
                (match_id, shard_id, map_name, game_mode, created_at, telemetry_url)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    match.match_id,
                    match.shard_id,
                    match.map_name,
                    match.game_mode,
                    match.created_at,
                    match.telemetry_url,
                ),
            )

            if plane_route:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO plane_routes
                    (match_id, start_x, start_y, end_x, end_y, angle, route_group, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plane_route.match_id,
                        plane_route.start_x,
                        plane_route.start_y,
                        plane_route.end_x,
                        plane_route.end_y,
                        plane_route.angle,
                        plane_route.route_group,
                        plane_route.confidence,
                    ),
                )

            connection.executemany(
                """
                INSERT OR REPLACE INTO circles
                (match_id, phase, center_x, center_y, radius, timestamp, elapsed_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        circle.match_id,
                        circle.phase,
                        circle.center_x,
                        circle.center_y,
                        circle.radius,
                        circle.timestamp,
                        circle.elapsed_time,
                    )
                    for circle in circles
                ],
            )

    def export(self) -> None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self._export_table("matches", PROCESSED_DIR / "matches.csv")
        self._export_table("plane_routes", PROCESSED_DIR / "plane_routes.csv")
        self._export_table("circles", PROCESSED_DIR / "circles.csv")
        self._export_wide_dataset(PROCESSED_DIR / "match_sequences.csv")

    def _export_table(self, table_name: str, path: Path) -> None:
        with self._connect() as connection:
            rows = connection.execute(f"SELECT * FROM {table_name}").fetchall()
            field_names = [description[0] for description in connection.execute(f"SELECT * FROM {table_name}").description]

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(field_names)
            writer.writerows(rows)

    def _export_wide_dataset(self, path: Path) -> None:
        fields = [
            "match_id",
            "route_group",
            "plane_start_x",
            "plane_start_y",
            "plane_end_x",
            "plane_end_y",
            "plane_angle",
        ]
        for phase in range(1, 10):
            fields.extend([f"p{phase}_center_x", f"p{phase}_center_y", f"p{phase}_radius"])

        with self._connect() as connection:
            matches = connection.execute(
                """
                SELECT m.match_id, p.route_group, p.start_x, p.start_y, p.end_x, p.end_y, p.angle
                FROM matches m
                LEFT JOIN plane_routes p ON p.match_id = m.match_id
                ORDER BY m.created_at
                """
            ).fetchall()

            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(fields)

                for match in matches:
                    circles = {
                        row[0]: row[1:]
                        for row in connection.execute(
                            """
                            SELECT phase, center_x, center_y, radius
                            FROM circles
                            WHERE match_id = ?
                            ORDER BY phase
                            """,
                            (match[0],),
                        ).fetchall()
                    }
                    row = list(match)
                    for phase in range(1, 10):
                        row.extend(circles.get(phase, ("", "", "")))
                    writer.writerow(row)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path)
