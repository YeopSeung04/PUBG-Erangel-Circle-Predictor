from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import requests

from circle_train.config import ensure_data_dirs, load_settings
from circle_train.pubg_api import PubgApiClient
from circle_train.storage import CircleTrainStore
from circle_train.telemetry_parser import ERANGEL_MAP_NAMES, parse_circles, parse_match, parse_plane_route


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect PUBG Erangel circle sequence data.")
    parser.add_argument("--database", default=None, help="SQLite database path.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    samples_parser = subparsers.add_parser("collect-samples", help="Collect matches from PUBG samples.")
    samples_parser.add_argument("--limit", type=int, default=10, help="Maximum saved Erangel matches.")
    samples_parser.add_argument("--created-at-start", default=None, help="UTC ISO time for samples filter.")
    samples_parser.add_argument("--include-non-erangel", action="store_true")
    samples_parser.add_argument("--min-circles", type=int, default=9)
    samples_parser.add_argument("--allow-missing-plane-route", action="store_true")
    samples_parser.add_argument("--quiet-skip", action="store_true")

    history_parser = subparsers.add_parser("collect-history", help="Collect sample windows across recent days and shards.")
    history_parser.add_argument("--limit", type=int, default=1000, help="Maximum saved match sequences.")
    history_parser.add_argument("--days", type=int, default=14, help="Number of recent UTC days to scan.")
    history_parser.add_argument("--shards", default="steam,kakao", help="Comma-separated shard list.")
    history_parser.add_argument("--include-non-erangel", action="store_true")
    history_parser.add_argument("--min-circles", type=int, default=9)
    history_parser.add_argument("--allow-missing-plane-route", action="store_true")
    history_parser.add_argument("--quiet-skip", action="store_true")

    match_parser = subparsers.add_parser("collect-match", help="Collect one match by ID.")
    match_parser.add_argument("--match-id", required=True)
    match_parser.add_argument("--include-non-erangel", action="store_true")
    match_parser.add_argument("--min-circles", type=int, default=9)
    match_parser.add_argument("--allow-missing-plane-route", action="store_true")

    subparsers.add_parser("export", help="Export SQLite tables to CSV.")

    args = parser.parse_args()

    ensure_data_dirs()
    settings = load_settings(args.database)
    store = CircleTrainStore(settings.database_path)
    store.initialize()

    if args.command == "export":
        store.export()
        print("Exported CSV files to data/processed.")
        return 0

    client = PubgApiClient(
        api_key=settings.api_key,
        shard=settings.shard,
        requests_per_minute=settings.requests_per_minute,
    )

    if args.command == "collect-samples":
        try:
            samples = client.get_samples(args.created_at_start)
        except requests.HTTPError as error:
            print(f"samples request failed: {format_http_error(error)}", file=sys.stderr)
            return 1

        match_ids = extract_sample_match_ids(samples)
        saved_count = 0

        for match_id in match_ids:
            if saved_count >= args.limit:
                break
            if store.has_match(match_id):
                continue

            result = collect_one_match(
                client=client,
                store=store,
                shard=settings.shard,
                match_id=match_id,
                include_non_erangel=args.include_non_erangel,
                min_circles=args.min_circles,
                require_plane_route=not args.allow_missing_plane_route,
                quiet_skip=args.quiet_skip,
            )
            if result:
                saved_count += 1
                print(f"saved {saved_count}/{args.limit}: {match_id}")

        print(f"Done. Saved {saved_count} match sequences.")
        return 0

    if args.command == "collect-history":
        total_saved = 0
        shards = [shard.strip() for shard in args.shards.split(",") if shard.strip()]
        sample_starts = get_recent_sample_starts(args.days)

        for shard in shards:
            shard_client = PubgApiClient(
                api_key=settings.api_key,
                shard=shard,
                requests_per_minute=settings.requests_per_minute,
            )
            for created_at_start in sample_starts:
                if total_saved >= args.limit:
                    break

                print(f"scan shard={shard} createdAt-start={created_at_start}")
                try:
                    samples = shard_client.get_samples(created_at_start)
                except requests.HTTPError as error:
                    print(f"samples request failed: {format_http_error(error)}", file=sys.stderr)
                    continue

                match_ids = extract_sample_match_ids(samples)
                stats = CollectionStats()
                for match_id in match_ids:
                    if total_saved >= args.limit:
                        break
                    if store.has_match(match_id):
                        stats.already_saved += 1
                        continue

                    result = collect_one_match(
                        client=shard_client,
                        store=store,
                        shard=shard,
                        match_id=match_id,
                        include_non_erangel=args.include_non_erangel,
                        min_circles=args.min_circles,
                        require_plane_route=not args.allow_missing_plane_route,
                        quiet_skip=args.quiet_skip,
                    )
                    stats.scanned += 1
                    if result:
                        total_saved += 1
                        stats.saved += 1
                        print(f"saved total={total_saved}/{args.limit}: {match_id}")

                print(
                    "window done: "
                    f"scanned={stats.scanned} saved={stats.saved} already_saved={stats.already_saved}"
                )

        print(f"Done. Saved {total_saved} new match sequences.")
        return 0

    if args.command == "collect-match":
        saved = collect_one_match(
            client=client,
            store=store,
            shard=settings.shard,
            match_id=args.match_id,
            include_non_erangel=args.include_non_erangel,
            min_circles=args.min_circles,
            require_plane_route=not args.allow_missing_plane_route,
            quiet_skip=False,
        )
        print("Saved match sequence." if saved else "Match was skipped.")
        return 0

    return 1


def collect_one_match(
    client: PubgApiClient,
    store: CircleTrainStore,
    shard: str,
    match_id: str,
    include_non_erangel: bool,
    min_circles: int,
    require_plane_route: bool,
    quiet_skip: bool,
) -> bool:
    try:
        match_payload = client.get_match(match_id)
        match = parse_match(match_payload, shard)
        if not match:
            print_skip(quiet_skip, f"skip {match_id}: missing match or telemetry URL")
            return False

        if not include_non_erangel and match.map_name not in ERANGEL_MAP_NAMES:
            print_skip(quiet_skip, f"skip {match_id}: map={match.map_name}")
            return False

        telemetry = client.get_telemetry(match_id, match.telemetry_url)
        circles = parse_circles(match_id, telemetry)
        if len(circles) < min_circles:
            print_skip(quiet_skip, f"skip {match_id}: circles={len(circles)}")
            return False

        plane_route = parse_plane_route(match_id, telemetry)
        if plane_route is None:
            if require_plane_route:
                print_skip(quiet_skip, f"skip {match_id}: plane route could not be inferred")
                return False
            print(f"warn {match_id}: plane route could not be inferred")

        store.save_match_sequence(match, plane_route, circles)
        return True
    except Exception as error:
        print(f"error {match_id}: {error}", file=sys.stderr)
        return False


@dataclass
class CollectionStats:
    scanned: int = 0
    saved: int = 0
    already_saved: int = 0


def get_recent_sample_starts(days: int) -> list[str]:
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return [
        (today - timedelta(days=offset)).isoformat().replace("+00:00", "Z")
        for offset in range(1, max(1, days) + 1)
    ]


def print_skip(quiet_skip: bool, message: str) -> None:
    if not quiet_skip:
        print(message)


def extract_sample_match_ids(samples: dict) -> list[str]:
    data = samples.get("data", {})
    if isinstance(data, list):
        return [item["id"] for item in data if item.get("type") == "match"]

    matches = data.get("relationships", {}).get("matches", {}).get("data", [])
    return [item["id"] for item in matches if item.get("type") == "match"]


def format_http_error(error: requests.HTTPError) -> str:
    response = error.response
    if response is None:
        return str(error)

    detail = response.text.strip()
    if len(detail) > 300:
        detail = f"{detail[:300]}..."
    return f"{response.status_code} {response.reason} - {detail}"


if __name__ == "__main__":
    raise SystemExit(main())
