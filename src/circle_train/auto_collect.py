from __future__ import annotations

import argparse
import ctypes
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from circle_train.analysis import export_route_summary, export_vectors
from circle_train.collector import collect_one_match, extract_sample_match_ids, format_http_error
from circle_train.config import ensure_data_dirs, load_settings
from circle_train.pubg_api import PubgApiClient
from circle_train.storage import CircleTrainStore


DEFAULT_SHARDS = "steam,kakao"


@dataclass
class AutoCollectResult:
    scanned: int = 0
    saved: int = 0
    already_saved: int = 0
    sample_windows: int = 0
    errors: int = 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run automated PUBG Erangel circle data collection.")
    parser.add_argument("--database", default=None, help="SQLite database path.")
    parser.add_argument("--target-new", type=int, default=1000, help="Stop after this many newly saved matches.")
    parser.add_argument("--days", type=int, default=14, help="Recent UTC sample days to scan.")
    parser.add_argument("--shards", default=DEFAULT_SHARDS, help="Comma-separated shard list.")
    parser.add_argument("--min-circles", type=int, default=2, help="Minimum circle phases required.")
    parser.add_argument("--strict-full-sequence", action="store_true", help="Require all 9 circle phases.")
    parser.add_argument("--include-non-erangel", action="store_true")
    parser.add_argument("--allow-missing-plane-route", action="store_true")
    parser.add_argument("--repeat", action="store_true", help="Run forever at the given interval.")
    parser.add_argument("--interval-hours", type=float, default=24.0, help="Repeat interval in hours.")
    parser.add_argument("--no-notify", action="store_true", help="Disable Windows completion notification.")
    parser.add_argument("--fail-on-errors", action="store_true", help="Return non-zero when sample windows fail.")
    args = parser.parse_args()

    if args.strict_full_sequence:
        args.min_circles = 9

    while True:
        started_at = datetime.now()
        exit_code, result = run_once(args)
        elapsed = datetime.now() - started_at

        message = (
            "PUBG Erangel circle collection finished.\n\n"
            f"Saved new matches: {result.saved}\n"
            f"Scanned matches: {result.scanned}\n"
            f"Already saved: {result.already_saved}\n"
            f"Sample windows: {result.sample_windows}\n"
            f"Errors: {result.errors}\n"
            f"Elapsed: {str(elapsed).split('.')[0]}"
        )

        print(message)
        if not args.no_notify:
            notify("PUBG Erangel Circle Predictor", message)

        if not args.repeat:
            return exit_code

        sleep_seconds = max(60.0, args.interval_hours * 60.0 * 60.0)
        next_run = datetime.now() + timedelta(seconds=sleep_seconds)
        print(f"Next run: {next_run:%Y-%m-%d %H:%M:%S}")
        time.sleep(sleep_seconds)


def run_once(args: argparse.Namespace) -> tuple[int, AutoCollectResult]:
    ensure_data_dirs()
    settings = load_settings(args.database)
    store = CircleTrainStore(settings.database_path)
    store.initialize()

    shards = [shard.strip() for shard in args.shards.split(",") if shard.strip()]
    sample_starts = get_recent_sample_starts(args.days)
    result = AutoCollectResult()

    for shard in shards:
        client = PubgApiClient(
            api_key=settings.api_key,
            shard=shard,
            requests_per_minute=settings.requests_per_minute,
        )

        for created_at_start in sample_starts:
            if result.saved >= args.target_new:
                break

            result.sample_windows += 1
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] scan shard={shard} createdAt-start={created_at_start}")
            try:
                samples = client.get_samples(created_at_start)
            except requests.HTTPError as error:
                result.errors += 1
                print(f"samples request failed: {format_http_error(error)}", file=sys.stderr)
                continue

            for match_id in extract_sample_match_ids(samples):
                if result.saved >= args.target_new:
                    break
                if store.has_match(match_id):
                    result.already_saved += 1
                    continue

                saved = collect_one_match(
                    client=client,
                    store=store,
                    shard=shard,
                    match_id=match_id,
                    include_non_erangel=args.include_non_erangel,
                    min_circles=args.min_circles,
                    require_plane_route=not args.allow_missing_plane_route,
                    quiet_skip=True,
                )
                result.scanned += 1
                if saved:
                    result.saved += 1
                    print(f"saved {result.saved}/{args.target_new}: {match_id}")

    store.export()
    export_vectors(settings.database_path)
    export_route_summary(settings.database_path)
    if args.fail_on_errors and result.errors > 0:
        return 1, result
    return 0, result


def get_recent_sample_starts(days: int) -> list[str]:
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return [
        (today - timedelta(days=offset)).isoformat().replace("+00:00", "Z")
        for offset in range(1, max(1, days) + 1)
    ]


def notify(title: str, message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(None, message, title, 0x00001000)
    except Exception:
        print(f"{title}: {message}")


if __name__ == "__main__":
    raise SystemExit(main())
