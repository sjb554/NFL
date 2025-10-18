"""Configuration helpers for environment-driven settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

import yaml


@dataclass(frozen=True)
class OddsAPIConfig:
    """Settings for The Odds API access."""

    api_key: str | None
    base_url: str = "https://api.the-odds-api.com/v4"
    sport_key: str = "americanfootball_nfl"
    regions: str = "us"
    markets: str = "spreads"


@dataclass(frozen=True)
class NFLFastRConfig:
    """Settings for nflfastR data access via the public GitHub mirror."""

    base_url: str = (
        "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data"
    )
    team_stats_path: str = "team_stats/team_stats.csv.gz"


@dataclass(frozen=True)
class ProjectSettings:
    """Project-level modeling placeholders loaded from settings.yaml."""

    markets: list[str]
    min_ev: float
    kelly_fraction: float
    max_stake_pct: float
    lookback_weeks: int
    train_seasons: list[int]


@dataclass(frozen=True)
class AppConfig:
    odds_api: OddsAPIConfig
    nflfastr: NFLFastRConfig
    settings: ProjectSettings


_DEFAULT_SETTINGS: dict[str, Any] = {
    "markets": ["ml", "spread", "total"],
    "min_ev": 0.02,
    "kelly_fraction": 0.25,
    "max_stake_pct": 0.02,
    "lookback_weeks": 6,
    "train_seasons": [2022, 2023, 2024],
}


def load_settings(path: str | os.PathLike[str] = "settings.yaml") -> ProjectSettings:
    """Load project settings from the given YAML file."""

    settings_path = Path(path)
    data: dict[str, Any] = {}
    if settings_path.exists():
        raw = settings_path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(raw) or {}
        if not isinstance(loaded, dict):  # pragma: no cover - safety
            raise ValueError("settings.yaml must contain a mapping")
        data = loaded

    merged = {**_DEFAULT_SETTINGS, **data}

    markets = [str(m).strip() for m in merged.get("markets", []) if str(m).strip()]
    train_seasons = [int(season) for season in merged.get("train_seasons", [])]

    return ProjectSettings(
        markets=markets or _DEFAULT_SETTINGS["markets"],
        min_ev=float(merged["min_ev"]),
        kelly_fraction=float(merged["kelly_fraction"]),
        max_stake_pct=float(merged["max_stake_pct"]),
        lookback_weeks=int(merged["lookback_weeks"]),
        train_seasons=train_seasons or _DEFAULT_SETTINGS["train_seasons"],
    )


def load_config(settings_path: str | os.PathLike[str] = "settings.yaml") -> AppConfig:
    """Load configuration from environment variables and settings.yaml."""

    odds_api = OddsAPIConfig(
        api_key=os.getenv("ODDS_API_KEY"),
    )
    nflfastr = NFLFastRConfig(
        base_url=os.getenv(
            "NFL_FASTR_BASE_URL",
            "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data",
        ),
        team_stats_path=os.getenv(
            "NFL_FASTR_TEAM_STATS_PATH", "team_stats/team_stats.csv.gz"
        ),
    )
    settings = load_settings(settings_path)
    return AppConfig(odds_api=odds_api, nflfastr=nflfastr, settings=settings)


__all__ = [
    "AppConfig",
    "NFLFastRConfig",
    "OddsAPIConfig",
    "ProjectSettings",
    "load_config",
    "load_settings",
]
