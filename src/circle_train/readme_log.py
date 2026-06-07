from __future__ import annotations

import argparse
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from circle_train.config import DEFAULT_DATABASE_PATH, PROJECT_ROOT


LOG_START = "<!-- COLLECTION_LOG_START -->"
LOG_END = "<!-- COLLECTION_LOG_END -->"


def main() -> int:
    parser = argparse.ArgumentParser(description="Update README collection log.")
    parser.add_argument("--readme", default=str(PROJECT_ROOT / "README.md"))
    parser.add_argument("--database", default=str(DEFAULT_DATABASE_PATH))
    parser.add_argument("--date", default=current_kst_date())
    parser.add_argument("--added", type=int, required=True)
    args = parser.parse_args()

    readme_path = Path(args.readme)
    database_path = Path(args.database)
    total_matches = count_matches(database_path)

    update_readme_log(readme_path, args.date, args.added, total_matches)
    return 0


def count_matches(database_path: Path) -> int:
    if not database_path.exists():
        return 0

    with sqlite3.connect(database_path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0])


def update_readme_log(readme_path: Path, date: str, added: int, total_matches: int) -> None:
    content = readme_path.read_text(encoding="utf-8")
    existing_entries = parse_existing_entries(content)
    entries = [(entry_date, entry_count) for entry_date, entry_count in existing_entries if entry_date != date]
    entries.insert(0, (date, added))

    log_block = build_log_block(entries, total_matches)
    if LOG_START in content and LOG_END in content:
        before = content.split(LOG_START, 1)[0].rstrip()
        after = content.split(LOG_END, 1)[1].lstrip()
        updated = f"{before}\n\n{log_block}\n\n{after}".rstrip() + "\n"
    else:
        updated = content.rstrip() + "\n\n" + log_block + "\n"

    readme_path.write_text(updated, encoding="utf-8", newline="\n")


def parse_existing_entries(content: str) -> list[tuple[str, int]]:
    if LOG_START not in content or LOG_END not in content:
        return []

    block = content.split(LOG_START, 1)[1].split(LOG_END, 1)[0]
    entries: list[tuple[str, int]] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue

        if ":" not in stripped:
            continue

        date_part, count_part = stripped[2:].split(":", 1)
        digits = "".join(character for character in count_part if character.isdigit())
        if digits:
            entries.append((date_part.strip(), int(digits)))

    return entries


def build_log_block(entries: list[tuple[str, int]], total_matches: int) -> str:
    lines = [
        LOG_START,
        "## 수집 로그",
        "",
        f"총 수집 데이터: {total_matches}개",
        "",
    ]

    for date, count in entries[:90]:
        lines.append(f"- {date}: {count}개 데이터")

    lines.append(LOG_END)
    return "\n".join(lines)


def current_kst_date() -> str:
    kst = UTC.utcoffset(None) or timedelta()
    now = datetime.now(UTC) + kst + timedelta(hours=9)
    return f"{now.year}/{now.month:02d}{now.day:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
