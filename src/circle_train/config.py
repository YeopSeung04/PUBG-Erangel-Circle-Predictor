from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_DATABASE_PATH = DATA_DIR / "circle_train.sqlite"


@dataclass(frozen=True)
class Settings:
    api_key: str
    shard: str
    requests_per_minute: int
    database_path: Path


def load_settings(database_path: str | None = None) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = "".join(os.getenv("PUBG_API_KEY", "").split())
    shard = os.getenv("PUBG_SHARD", "steam").strip()
    requests_per_minute = int(os.getenv("PUBG_REQUESTS_PER_MINUTE", "10"))

    db_path = Path(database_path) if database_path else DEFAULT_DATABASE_PATH

    return Settings(
        api_key=api_key,
        shard=shard,
        requests_per_minute=requests_per_minute,
        database_path=db_path,
    )


def ensure_data_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
