"""Configuration helpers for environment-driven settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Mapping

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
class VegasSettings:
    """Guardrails for the one-click Vegas workflow."""

    max_total_stake_pct: float
    min_books_per_selection: int
    stale_minutes_ok: int
    default_bankroll: float


@dataclass(frozen=True)
class AppConfig:
    odds_api: OddsAPIConfig
    nflfastr: NFLFastRConfig
    settings: ProjectSettings
    vegas: VegasSettings


_DEFAULT_PROJECT: dict[str, Any] = {
    "markets": ["ml", "spread", "total"],
    "min_ev": 0.02,
    "kelly_fraction": 0.25,
    "max_stake_pct": 0.02,
    "lookback_weeks": 8,
    "train_seasons": [2022, 2023, 2024],
}

_DEFAULT_VEGAS: dict[str, Any] = {
    "max_total_stake_pct": 0.05,
    "min_books_per_selection": 3,
    "stale_minutes_ok": 60,
    "default_bankroll": 5000.0,
}


def load_config(settings_path: str | os.PathLike[str] = "settings.yaml") -> AppConfig:
    """Load configuration from environment variables and settings.yaml."""

    raw = _load_settings_file(settings_path)

    odds_api = OddsAPIConfig(api_key=os.getenv("ODDS_API_KEY"))
    nflfastr = NFLFastRConfig(
        base_url=os.getenv(
            "NFL_FASTR_BASE_URL",
            "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data",
        ),
        team_stats_path=os.getenv(
            "NFL_FASTR_TEAM_STATS_PATH",
            "team_stats/team_stats.csv.gz",
        ),
    )

    project = _build_project_settings(raw)
    vegas = _build_vegas_settings(raw)

    return AppConfig(odds_api=odds_api, nflfastr=nflfastr, settings=project, vegas=vegas)


def _load_settings_file(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    settings_path = Path(path)
    if not settings_path.exists():
        return {}
    raw = settings_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw) or {}
    if not isinstance(loaded, dict):  # pragma: no cover - safety
        raise ValueError("settings.yaml must contain a mapping")
    return loaded


def _build_project_settings(data: Mapping[str, Any]) -> ProjectSettings:
    merged = {**_DEFAULT_PROJECT, **{k: v for k, v in data.items() if k in _DEFAULT_PROJECT}}
    markets = [str(m).strip() for m in merged.get("markets", []) if str(m).strip()]
    train_seasons = [int(season) for season in merged.get("train_seasons", [])]
    return ProjectSettings(
        markets=markets or _DEFAULT_PROJECT["markets"],
        min_ev=float(merged["min_ev"]),
        kelly_fraction=float(merged["kelly_fraction"]),
        max_stake_pct=float(merged["max_stake_pct"]),
        lookback_weeks=int(merged["lookback_weeks"]),
        train_seasons=train_seasons or _DEFAULT_PROJECT["train_seasons"],
    )


def _build_vegas_settings(data: Mapping[str, Any]) -> VegasSettings:
    merged = {**_DEFAULT_VEGAS, **{k: v for k, v in data.items() if k in _DEFAULT_VEGAS}}
    return VegasSettings(
        max_total_stake_pct=float(merged["max_total_stake_pct"]),
        min_books_per_selection=int(merged["min_books_per_selection"]),
        stale_minutes_ok=int(merged["stale_minutes_ok"]),
        default_bankroll=float(merged["default_bankroll"]),
    )


__all__ = [
    "AppConfig",
    "NFLFastRConfig",
    "OddsAPIConfig",
    "ProjectSettings",
    "VegasSettings",
    "load_config",
]
