"""Helpers for lightweight access to nflfastR public data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
import csv
import gzip
import logging
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import requests

from .config import NFLFastRConfig

logger = logging.getLogger(__name__)

_TEAM_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_team/stats_team_week_{season}.csv.gz"
)
_TEAM_CACHE_DIR = Path("data_cache") / "nflfastr"
_TEAM_CACHE_MAX_AGE = timedelta(hours=12)


@dataclass(slots=True, frozen=True)
class TeamStat:
    team: str
    season: int
    games: int
    points_for: float
    points_against: float

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "TeamStat":
        return cls(
            team=row.get("team", ""),
            season=_safe_int(row.get("season")),
            games=_safe_int(row.get("games")),
            points_for=_safe_float(row.get("points")),
            points_against=_safe_float(row.get("points_against")),
        )


@dataclass(slots=True, frozen=True)
class TeamFeatures:
    """Lightweight view of recent team performance."""

    team: str
    offensive_epa_per_play: float
    defensive_epa_per_play: float
    success_rate_off: float
    success_rate_def: float
    turnover_rate: float
    pace: float


class NFLFastRClient:
    """Client for the nflfastR public CSV exports."""

    def __init__(
        self,
        config: NFLFastRConfig,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._config = config
        self._timeout = timeout
        self._session = session or requests.Session()

    def fetch_latest_team_stats(self, limit: int | None = 10) -> list[TeamStat]:
        """Return a handful of most recent team-level summary rows."""

        url = f"{self._config.base_url}/{self._config.team_stats_path}"
        logger.debug("Fetching team stats from %s", url)
        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()

        rows = _iterate_csv_rows(response.content)
        stats: list[TeamStat] = []
        for row in rows:
            stats.append(TeamStat.from_row(row))
            if limit is not None and len(stats) >= limit:
                break

        return stats

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "NFLFastRClient":  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc_info) -> None:  # pragma: no cover - convenience
        self.close()


def get_team_features(
    weeks: int = 6,
    season: int = 2025,
    *,
    cache_dir: Path | None = None,
) -> dict[str, TeamFeatures]:
    """Return rolling team features for the requested season."""

    cache_dir = cache_dir or _TEAM_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    content = _load_team_week_csv(season, cache_dir)
    if content is None:
        logger.warning("Unable to load nflfastR weekly team stats.")
        return {}

    df = pd.read_csv(BytesIO(content), compression="gzip")
    if df.empty:
        return {}

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    if season not in df["season"].dropna().unique():
        fallback = int(df["season"].dropna().max())
        logger.info("Season %s not available in weekly stats; falling back to %s.", season, fallback)
        season = fallback

    df = df[(df["season"] == season) & (df["season_type"] == "REG")].copy()
    if df.empty:
        return {}

    numeric_cols = [
        "attempts",
        "carries",
        "sacks_suffered",
        "passing_epa",
        "rushing_epa",
        "passing_first_downs",
        "rushing_first_downs",
        "passing_interceptions",
        "rushing_fumbles_lost",
        "sack_fumbles_lost",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    df["plays"] = df["attempts"] + df["carries"] + df["sacks_suffered"]
    df["off_epa"] = df["passing_epa"] + df["rushing_epa"]
    df["first_downs"] = df["passing_first_downs"] + df["rushing_first_downs"]
    df["turnovers"] = (
        df["passing_interceptions"] + df["rushing_fumbles_lost"] + df["sack_fumbles_lost"]
    )

    opponent_cols = [
        "season",
        "week",
        "team",
        "passing_epa",
        "rushing_epa",
        "attempts",
        "carries",
        "sacks_suffered",
        "passing_first_downs",
        "rushing_first_downs",
        "passing_interceptions",
        "rushing_fumbles_lost",
        "sack_fumbles_lost",
    ]
    opp_df = df[opponent_cols].rename(
        columns={
            "team": "opp_team",
            "passing_epa": "opp_passing_epa",
            "rushing_epa": "opp_rushing_epa",
            "attempts": "opp_attempts",
            "carries": "opp_carries",
            "sacks_suffered": "opp_sacks_suffered",
            "passing_first_downs": "opp_passing_first_downs",
            "rushing_first_downs": "opp_rushing_first_downs",
            "passing_interceptions": "opp_passing_interceptions",
            "rushing_fumbles_lost": "opp_rushing_fumbles_lost",
            "sack_fumbles_lost": "opp_sack_fumbles_lost",
        }
    )

    df = df.merge(
        opp_df,
        left_on=["season", "week", "opponent_team"],
        right_on=["season", "week", "opp_team"],
        how="left",
    )

    df.fillna(0, inplace=True)

    df["opp_plays"] = (
        df["opp_attempts"] + df["opp_carries"] + df["opp_sacks_suffered"]
    ).replace({0: 1})

    df["offensive_epa_per_play"] = (df["off_epa"] / df["plays"].replace({0: 1})).clip(-1.5, 1.5)
    df["defensive_epa_per_play"] = (
        -(df["opp_passing_epa"] + df["opp_rushing_epa"]) / df["opp_plays"]
    ).clip(-1.5, 1.5)
    df["success_rate_off"] = (df["first_downs"] / df["plays"].replace({0: 1})).clip(0, 1)
    df["success_rate_def"] = (
        1 - (
            (
                df["opp_passing_first_downs"] + df["opp_rushing_first_downs"]
            )
            / df["opp_plays"]
        )
    ).clip(0, 1)
    df["turnover_rate"] = (df["turnovers"] / df["plays"].replace({0: 1})).clip(0, 1)
    df["pace"] = df["plays"].clip(lower=0)

    feature_cols = [
        "offensive_epa_per_play",
        "defensive_epa_per_play",
        "success_rate_off",
        "success_rate_def",
        "turnover_rate",
        "pace",
    ]

    df.sort_values(["team", "week"], inplace=True)
    rolling = (
        df.groupby("team")[feature_cols]
        .apply(lambda group: group.tail(max(weeks, 1)).mean())
        .fillna(0.0)
    )

    league_average = rolling.mean().to_dict()

    features: dict[str, TeamFeatures] = {}
    for team, row in rolling.iterrows():
        features[team] = TeamFeatures(
            team=team,
            offensive_epa_per_play=float(row["offensive_epa_per_play"]),
            defensive_epa_per_play=float(row["defensive_epa_per_play"]),
            success_rate_off=float(row["success_rate_off"]),
            success_rate_def=float(row["success_rate_def"]),
            turnover_rate=float(row["turnover_rate"]),
            pace=float(row["pace"]),
        )

    features["LEAGUE_AVG"] = TeamFeatures(
        team="LEAGUE_AVG",
        offensive_epa_per_play=float(league_average["offensive_epa_per_play"]),
        defensive_epa_per_play=float(league_average["defensive_epa_per_play"]),
        success_rate_off=float(league_average["success_rate_off"]),
        success_rate_def=float(league_average["success_rate_def"]),
        turnover_rate=float(league_average["turnover_rate"]),
        pace=float(league_average["pace"]),
    )

    return features


def _load_team_week_csv(season: int, cache_dir: Path) -> bytes | None:
    cache_path = cache_dir / f"stats_team_week_{season}.csv.gz"
    if cache_path.exists():
        modified = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        if datetime.now(timezone.utc) - modified <= _TEAM_CACHE_MAX_AGE:
            return cache_path.read_bytes()

    url = _TEAM_WEEK_URL.format(season=season)
    logger.debug("Downloading weekly team stats from %s", url)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to download weekly team stats: %s", exc)
        if cache_path.exists():
            logger.info("Using stale cached weekly stats for season %s.", season)
            return cache_path.read_bytes()
        return None

    cache_path.write_bytes(response.content)
    return response.content


def _iterate_csv_rows(blob: bytes) -> Iterable[Mapping[str, str]]:
    with gzip.open(BytesIO(blob), "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        yield from reader


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def _safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


__all__ = ["NFLFastRClient", "TeamStat", "TeamFeatures", "get_team_features"]
