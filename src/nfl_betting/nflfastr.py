"""Lightweight nflfastR feature helpers with cached-neutral fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import gzip
import math
import pandas as pd

from .teams import normalize_team

CACHE_DIR = Path("data_cache") / "nflfastR"


@dataclass(frozen=True)
class TeamFeatures:
    team: str
    offensive_epa_per_play: float
    defensive_epa_per_play: float
    success_rate_off: float
    success_rate_def: float
    turnover_rate: float
    pace: float


def get_team_features(
    teams: Sequence[str],
    weeks: int = 6,
    date: str | None = None,
    cache_dir: str | Path = CACHE_DIR,
    *,
    season: int | None = None,
    return_metadata: bool = False,
    as_df: bool = False,
) -> pd.DataFrame | dict[str, TeamFeatures] | tuple[pd.DataFrame | dict[str, TeamFeatures], dict[str, object]]:
    """Return recent team features, falling back to neutral values if needed."""

    teams = [team.strip() for team in teams if str(team).strip()]
    if not teams:
        data = _neutral_features([])
        if return_metadata:
            meta = {"cache_path": None, "weeks": weeks, "season": season, "teams": []}
            return (data.set_index("team") if as_df else {}, meta)
        return data.set_index("team") if as_df else {}

    team_codes: list[str] = []
    for team in teams:
        try:
            code = normalize_team(team)
        except ValueError:
            code = team.upper()
        team_codes.append(code)
    team_codes = list(dict.fromkeys(team_codes))

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    stats_path: Path | None = None
    if season is not None:
        candidate = cache_path / f"stats_team_week_{int(season)}.csv.gz"
        if candidate.exists():
            stats_path = candidate
    if stats_path is None:
        candidates = sorted(cache_path.glob("stats_team_week_*.csv.gz"), reverse=True)
        if candidates:
            stats_path = candidates[0]
            if season is None:
                try:
                    season = int(stats_path.stem.split("_")[-1])
                except (ValueError, IndexError):
                    season = None

    df: pd.DataFrame | None = None
    if stats_path and stats_path.exists():
        import gzip

        try:
            with gzip.open(stats_path, "rt") as handle:
                df = pd.read_csv(handle)
        except Exception:
            df = None

    feature_rows: list[dict[str, float | str]] = []
    if df is not None and not df.empty:
        work_df = df.copy()
        if season is not None:
            work_df = work_df[work_df["season"] == int(season)]
        if "season_type" in work_df.columns:
            work_df = work_df[work_df["season_type"].str.upper() == "REG"]
        work_df = work_df[work_df["team"].isin(team_codes)]
        if not work_df.empty:
            work_df = work_df.sort_values(["team", "week"], ascending=[True, False])
            work_df["plays"] = (work_df["attempts"].clip(lower=0) + work_df["carries"].clip(lower=0)).replace(0, 1)
            feature_rows = []
            for code in team_codes:
                team_slice = work_df[work_df["team"] == code].head(max(weeks, 1))
                if team_slice.empty:
                    continue
                plays = float(team_slice["plays"].sum()) or 1.0
                games_played = max(len(team_slice), 1)
                off_epa = (team_slice["passing_epa"].sum() + team_slice["rushing_epa"].sum()) / plays
                def_factor = (
                    team_slice.get("def_sacks", 0).sum()
                    + team_slice.get("def_interceptions", 0).sum()
                    + team_slice.get("def_tackles_for_loss", 0).sum()
                )
                def_epa = def_factor / plays
                success_off = (
                    team_slice["passing_first_downs"].sum() + team_slice["rushing_first_downs"].sum()
                ) / plays
                success_off = float(max(0.0, min(success_off, 1.0)))
                success_def = 0.5 + (def_factor / max(plays * 5.0, 1.0))
                success_def = float(max(0.0, min(success_def, 1.0)))
                turnovers = (
                    team_slice["passing_interceptions"].sum() + team_slice["rushing_fumbles_lost"].sum()
                )
                turnover_rate = float(max(0.0, turnovers / plays))
                pace = float(plays / games_played)
                feature_rows.append(
                    {
                        "team": code,
                        "offensive_epa_per_play": off_epa,
                        "defensive_epa_per_play": def_epa,
                        "success_rate_off": success_off,
                        "success_rate_def": success_def,
                        "turnover_rate": turnover_rate,
                        "pace": pace,
                    }
                )

    feature_df = pd.DataFrame(feature_rows)
    if feature_df.empty:
        feature_df = _neutral_features(team_codes)
    else:
        missing = [code for code in team_codes if code not in feature_df["team"].values]
        if missing:
            feature_df = pd.concat([feature_df, _neutral_features(missing)], ignore_index=True)

    league_row = feature_df.drop(columns=["team"]).mean().to_dict()
    league_row["team"] = "LEAGUE_AVG"
    feature_df = pd.concat([feature_df, pd.DataFrame([league_row])], ignore_index=True)

    if return_metadata:
        meta = {"cache_path": stats_path, "weeks": weeks, "season": season, "teams": team_codes}
    if as_df:
        data_out = feature_df.set_index("team")
    else:
        data_out = {
            row["team"]: TeamFeatures(
                team=row["team"],
                offensive_epa_per_play=float(row["offensive_epa_per_play"]),
                defensive_epa_per_play=float(row["defensive_epa_per_play"]),
                success_rate_off=float(row["success_rate_off"]),
                success_rate_def=float(row["success_rate_def"]),
                turnover_rate=float(row["turnover_rate"]),
                pace=float(row["pace"]),
            )
            for _, row in feature_df.iterrows()
        }

    if return_metadata:
        return data_out, meta
    return data_out


class NFLFastRClient:
    """Tiny stub mirroring the previous interface."""

    def __init__(self, cache_dir: str | Path = CACHE_DIR, lookback_weeks: int = 6) -> None:
        self._cache_dir = cache_dir
        self._lookback = lookback_weeks

    def get_team_features(
        self,
        teams: Sequence[str],
        date: str | None = None,
        *,
        as_df: bool = True,
    ) -> pd.DataFrame | dict[str, TeamFeatures]:
        return get_team_features(
            teams,
            weeks=self._lookback,
            date=date,
            cache_dir=self._cache_dir,
            as_df=as_df,
        )

    def close(self) -> None:  # pragma: no cover - compatibility
        return None

    def __enter__(self) -> "NFLFastRClient":  # pragma: no cover
        return self

    def __exit__(self, *exc_info) -> None:  # pragma: no cover
        return None


def _resolve_cache_path(cache_dir: str | Path, date: str | None) -> Path | None:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return None
    if date:
        expected = cache_dir / f"team_features_{date}.csv"
        if expected.exists():
            return expected
    csvs = sorted(cache_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def _filter_and_fill(df: pd.DataFrame, teams: Sequence[str]) -> pd.DataFrame:
    needed = {
        "team",
        "epa_off",
        "epa_def",
        "success_off",
        "success_def",
        "turnover_rate",
        "pace",
    }
    present = [col for col in df.columns if col in needed]
    if set(present) != needed:
        return _neutral_features(teams)
    filtered = df[df["team"].isin(teams)].copy()
    if filtered.empty:
        return _neutral_features(teams)
    return filtered


def _neutral_features(teams: Sequence[str]) -> pd.DataFrame:
    rows = []
    for team in teams:
        jitter = _jitter(team)
        rows.append(
            {
                "team": team,
                "offensive_epa_per_play": 0.0 + jitter * 0.01,
                "defensive_epa_per_play": 0.0 - jitter * 0.01,
                "success_rate_off": 0.5 + jitter * 0.02,
                "success_rate_def": 0.5 - jitter * 0.02,
                "turnover_rate": 0.02 + abs(jitter) * 0.005,
                "pace": 60 + jitter * 2,
            }
        )
    return pd.DataFrame(rows)


def _jitter(team: str) -> float:
    if not team:
        return 0.0
    value = sum(ord(ch) for ch in team)
    return math.sin(value) * 0.1


__all__ = ["NFLFastRClient", "TeamFeatures", "get_team_features"]
