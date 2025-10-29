"""Utilities for loading completed NFL game results."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

from .teams import normalize_team

logger = logging.getLogger(__name__)

_ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
_DEFAULT_CACHE = Path("data_cache") / "espn_scoreboard"


def load_actuals(
    dates: Sequence[str] | None = None,
    *,
    game_ids: Sequence[str] | None = None,
    cache_dir: str | Path = _DEFAULT_CACHE,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Return completed game results for the supplied slate."""

    date_tokens = _build_date_set(dates, game_ids)
    if not date_tokens:
        return _empty_actuals_frame()

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    session = session or requests.Session()
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for date in sorted(date_tokens):
        data = _fetch_scoreboard(date, cache_path, session)
        if not data:
            continue
        parsed = _extract_rows(data, default_date=date)
        for row in parsed:
            game_id = row["game_id"]
            if game_id in seen_ids:
                continue
            seen_ids.add(game_id)
            rows.append(row)

    if not rows:
        return _empty_actuals_frame()
    df = pd.DataFrame(rows)
    if "kickoff" in df.columns:
        df = df.drop(columns=["kickoff"])
    df = df.sort_values(["kickoff_date", "game_id"]).reset_index(drop=True)
    return df
def _build_date_set(
    dates: Sequence[str] | None,
    game_ids: Sequence[str] | None,
) -> set[str]:
    date_tokens: set[str] = set()
    if dates:
        for date in dates:
            token = _normalize_date_token(date)
            if token:
                date_tokens.add(token)
    if game_ids:
        for game_id in game_ids:
            token = _extract_date_from_game_id(game_id)
            if token:
                date_tokens.add(token)
    return date_tokens


def _normalize_date_token(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass
    cleaned = text.replace("/", "-")
    try:
        dt = datetime.strptime(cleaned, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass
    if len(cleaned) == 8 and cleaned.isdigit():
        try:
            dt = datetime.strptime(cleaned, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def _extract_date_from_game_id(game_id: str) -> str | None:
    tail = game_id[-10:]
    try:
        dt = datetime.strptime(tail, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass
    tail = game_id[-8:]
    if tail.isdigit():
        try:
            dt = datetime.strptime(tail, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def _fetch_scoreboard(
    date: str,
    cache_dir: Path,
    session: requests.Session,
) -> dict | None:
    slug = date.replace("-", "")
    cache_file = cache_dir / f"scoreboard_{slug}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache_file.unlink(missing_ok=True)

    url = f"{_ESPN_SCOREBOARD_URL}?dates={slug}"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:  # pragma: no cover - network failure
        logger.warning("ESPN scoreboard request failed for %s: %s", date, exc)
        return None

    try:
        cache_file.write_text(json.dumps(data), encoding="utf-8")
    except OSError:  # pragma: no cover - cache write failure
        logger.debug("Unable to write scoreboard cache for %s", date, exc_info=True)
    return data


def _extract_rows(data: dict, *, default_date: str) -> Iterable[dict[str, object]]:
    events = data.get("events") or []
    for event in events:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        status = comp.get("status", {}).get("type", {})
        state = str(status.get("state", "")).lower()
        completed = bool(status.get("completed")) or state in {"post", "final"}
        if not completed:
            continue

        competitors = comp.get("competitors") or []
        home = next((team for team in competitors if team.get("homeAway") == "home"), None)
        away = next((team for team in competitors if team.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        home_code = _resolve_team_code(home)
        away_code = _resolve_team_code(away)
        if not home_code or not away_code:
            continue

        try:
            home_score = float(home.get("score"))
            away_score = float(away.get("score"))
        except (TypeError, ValueError):
            continue
        if math.isnan(home_score) or math.isnan(away_score):
            continue

        kickoff_iso = comp.get("date") or event.get("date") or f"{default_date}T00:00:00Z"
        kickoff_date = _extract_date_from_iso(kickoff_iso) or default_date

        total_points = home_score + away_score
        margin = home_score - away_score
        outcome = "push"
        home_win: bool | None = None
        if margin > 0:
            outcome = "home"
            home_win = True
        elif margin < 0:
            outcome = "away"
            home_win = False

        record = {
            "game_id": f"{away_code}@{home_code}-{kickoff_date}",
            "kickoff": kickoff_iso,
            "kickoff_date": kickoff_date,
            "home_code": home_code,
            "away_code": away_code,
            "home_score": home_score,
            "away_score": away_score,
            "final_total": total_points,
            "margin": margin,
            "ml_winner": outcome,
            "home_win": home_win,
        }
        yield record
def _resolve_team_code(entry: dict) -> str | None:
    team = entry.get("team") or {}
    candidates = [
        team.get("abbreviation"),
        team.get("shortDisplayName"),
        team.get("displayName"),
        team.get("name"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return normalize_team(candidate)
        except ValueError:
            continue
    return None


def _extract_date_from_iso(value: str) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _empty_actuals_frame() -> pd.DataFrame:
    columns = [
        "game_id",
        "kickoff",
        "kickoff_date",
        "home_code",
        "away_code",
        "home_score",
        "away_score",
        "final_total",
        "margin",
        "ml_winner",
        "home_win",
    ]
    return pd.DataFrame(columns=columns)


__all__ = ["load_actuals"]
