"""Thin client for interacting with The Odds API."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import requests

from .config import OddsAPIConfig
from .teams import GameRequest, build_game_lookup, canonical_name, normalize_team

logger = logging.getLogger(__name__)

_MARKET_PARAM_MAP = {
    "ml": "h2h",
    "moneyline": "h2h",
    "h2h": "h2h",
    "spread": "spreads",
    "spreads": "spreads",
    "total": "totals",
    "totals": "totals",
}
_MARKET_CANONICAL = {
    "h2h": "ml",
    "spreads": "spread",
    "totals": "total",
}


@dataclass(frozen=True)
class NormalizedLine:
    """Normalized view of an individual line offering."""

    game_id: str
    game_label: str
    kickoff: str
    away: str
    home: str
    away_code: str
    home_code: str
    market: str
    side: str
    book: str
    price: int
    point: float | None
    last_update: str


class OddsAPIError(RuntimeError):
    """Raised when The Odds API responds with an error."""


class OddsAPIClient:
    """Simple HTTP client for The Odds API."""

    def __init__(
        self,
        config: OddsAPIConfig,
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._config = config
        self._timeout = timeout
        self._session = session or requests.Session()

    def fetch_odds(self, matchups: Iterable[tuple[str, str]]) -> dict[str, list[Mapping]]:
        """Fetch raw odds payloads for provided matchups (legacy helper)."""

        normalized_keys = {
            _normalize_matchup(matchup): f"{matchup[0]} @ {matchup[1]}"
            for matchup in matchups
        }
        logger.debug("Requesting odds for %s", list(normalized_keys.values()))

        if not self._config.api_key:
            logger.info("ODDS_API_KEY not set; skipping live odds request.")
            return {label: [] for label in normalized_keys.values()}

        params: MutableMapping[str, str] = {
            "apiKey": self._config.api_key,
            "regions": self._config.regions,
            "markets": self._config.markets,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        url = f"{self._config.base_url}/sports/{self._config.sport_key}/odds"

        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - convenience
            logger.debug("The Odds API request failed: %s", exc, exc_info=True)
            raise OddsAPIError("Unable to reach The Odds API") from exc

        payload = response.json()
        results: dict[str, list[Mapping]] = {label: [] for label in normalized_keys.values()}

        for game in payload:
            key = _normalize_matchup((game.get("away_team", ""), game.get("home_team", "")))
            label = normalized_keys.get(key)
            if not label:
                continue
            results[label].append(game)

        return results

    def fetch_lines(
        self,
        games: Sequence[GameRequest],
        *,
        markets: Sequence[str] | None = None,
        date: str | None = None,
        cache_dir: str | Path | None = None,
        max_cache_age_hours: int = 6,
    ) -> list[NormalizedLine]:
        """Fetch normalized line data for the requested games."""

        if not games:
            return []

        use_live = bool(self._config.api_key)
        if not use_live:
            logger.warning("ODDS_API_KEY not set; attempting to use cached odds only.")

        market_params = _select_markets(markets)
        game_lookup = build_game_lookup(games)
        results: list[NormalizedLine] = []

        for market in market_params:
            payload = self._load_market_payload(
                market,
                date=date,
                cache_dir=cache_dir,
                max_cache_age_hours=max_cache_age_hours,
                allow_live=use_live,
            )
            if not payload:
                continue
            results.extend(self._normalize_market_payload(payload, market, game_lookup))

        return results

    def _load_market_payload(
        self,
        market: str,
        *,
        date: str | None,
        cache_dir: str | Path | None,
        max_cache_age_hours: int,
        allow_live: bool,
    ) -> list[Mapping]:
        cache_file: Path | None = None
        cache_data: list[Mapping] | None = None
        if cache_dir:
            cache_base = Path(cache_dir)
            cache_base.mkdir(parents=True, exist_ok=True)
            cache_file = cache_base / f"{market}.json"
            if cache_file.exists():
                try:
                    cache_payload = json.loads(cache_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    cache_payload = None
                if cache_payload is not None:
                    modified = datetime.fromtimestamp(
                        cache_file.stat().st_mtime, tz=timezone.utc
                    )
                    age = datetime.now(timezone.utc) - modified
                    if age <= timedelta(hours=max_cache_age_hours) or not allow_live:
                        return cache_payload
                    cache_data = cache_payload

        if not allow_live:
            return cache_data or []

        params: MutableMapping[str, str] = {
            "apiKey": self._config.api_key,
            "regions": self._config.regions,
            "markets": market,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if date:
            params["date"] = date

        url = f"{self._config.base_url}/sports/{self._config.sport_key}/odds"
        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("The Odds API request failed for %s: %s", market, exc)
            if cache_data is not None:
                return cache_data
            return []

        payload = response.json()
        if cache_file:
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _normalize_market_payload(
        self,
        payload: Sequence[Mapping],
        market: str,
        game_lookup: Mapping[tuple[str, str], GameRequest],
    ) -> list[NormalizedLine]:
        normalized: list[NormalizedLine] = []
        canonical_market = _MARKET_CANONICAL.get(market, market)

        for event in payload:
            away_raw = event.get("away_team", "")
            home_raw = event.get("home_team", "")
            away = canonical_name(away_raw)
            home = canonical_name(home_raw)
            key = (away, home)
            game = game_lookup.get(key)

            if not game:
                continue

            kickoff_raw = str(event.get("commence_time", "")) or ""
            kickoff_iso = kickoff_raw
            kickoff_date = ""
            if kickoff_raw:
                try:
                    kickoff_dt = datetime.fromisoformat(kickoff_raw.replace("Z", "+00:00"))
                    kickoff_iso = kickoff_dt.isoformat()
                    kickoff_date = kickoff_dt.strftime("%Y-%m-%d")
                except ValueError:
                    kickoff_date = kickoff_raw[:10]
            if kickoff_date:
                game_id = f"{game.away_code}@{game.home_code}-{kickoff_date}"
            else:
                fallback = str(event.get("id", ""))
                game_id = fallback or f"{game.away_code}@{game.home_code}"

            label = game.label

            for bookmaker in event.get("bookmakers", []):
                book_title = bookmaker.get("title") or bookmaker.get("key") or "Unknown"
                last_update = bookmaker.get("last_update") or ""

                for market_data in bookmaker.get("markets", []):
                    if market_data.get("key") != market:
                        continue
                    for outcome in market_data.get("outcomes", []):
                        line = _line_from_outcome(
                            outcome,
                            canonical_market,
                            game_id=game_id,
                            game_label=label,
                            kickoff=kickoff_iso,
                            away=away,
                            home=home,
                            away_code=game.away_code,
                            home_code=game.home_code,
                            book=book_title,
                            last_update=last_update,
                        )
                        if line:
                            normalized.append(line)

        return normalized

    def close(self) -> None:
        """Close the underlying HTTP session."""

        self._session.close()

    def __enter__(self) -> "OddsAPIClient":  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc_info) -> None:  # pragma: no cover - convenience
        self.close()


def list_games(
    date: str,
    *,
    config: OddsAPIConfig | None = None,
    cache_dir: str | Path = "data_cache/odds",
    stale_hours: int = 6,
) -> list[str]:
    """Return normalized AWAY@HOME tokens for the given date."""

    config = config or OddsAPIConfig(api_key=os.getenv("ODDS_API_KEY"))
    if not config.api_key:
        logger.warning("ODDS_API_KEY not set; cannot fetch slate for %s", date)
        return []

    slug = date.replace("-", "")
    base_dir = Path(cache_dir) / slug
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_file = base_dir / "events.json"

    data: list[dict[str, object]] | None = None
    now = datetime.now(timezone.utc)
    if cache_file.exists():
        modified = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
        if now - modified <= timedelta(hours=stale_hours):
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = None

    if data is None:
        params = {
            "apiKey": config.api_key,
            "dateFormat": "iso",
            "date": date,
        }
        url = f"{config.base_url}/sports/{config.sport_key}/events"
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            cache_file.write_text(json.dumps(data), encoding="utf-8")
        except requests.RequestException as exc:
            logger.warning("Failed to fetch slate for %s: %s", date, exc)
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    data = None
            if data is None:
                return []

    tokens: list[str] = []
    seen: set[str] = set()
    for event in data or []:
        away = event.get("away_team", "")
        home = event.get("home_team", "")
        try:
            away_code = normalize_team(str(away))
            home_code = normalize_team(str(home))
        except ValueError:
            logger.warning("Skipping event with unsupported teams: %s @ %s", away, home)
            continue
        token = f"{away_code}@{home_code}"
        if token not in seen:
            tokens.append(token)
            seen.add(token)

    return tokens


def list_games_window(
    start_date: str,
    *,
    days: int = 7,
    config: OddsAPIConfig | None = None,
    cache_dir: str | Path = "data_cache/odds",
    stale_hours: int = 6,
) -> dict[str, list[str]]:
    """Return AWAY@HOME tokens grouped by date for the requested window."""

    if days <= 0:
        return {}

    try:
        base_date = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError("start_date must be in YYYY-MM-DD format") from exc

    shared_config = config or OddsAPIConfig(api_key=os.getenv("ODDS_API_KEY"))
    if not shared_config.api_key:
        logger.warning(
            "ODDS_API_KEY not set; cannot fetch slate window starting %s",
            start_date,
        )
        return {}

    results: dict[str, list[str]] = {}
    for offset in range(days):
        current_date = (base_date + timedelta(days=offset)).strftime("%Y-%m-%d")
        tokens = list_games(
            current_date,
            config=shared_config,
            cache_dir=cache_dir,
            stale_hours=stale_hours,
        )
        if tokens:
            results[current_date] = tokens

    return results

def _normalize_matchup(matchup: tuple[str, str]) -> tuple[str, str]:
    away, home = matchup
    return away.strip().lower(), home.strip().lower()


def _select_markets(markets: Sequence[str] | None) -> list[str]:
    if not markets:
        markets = ("ml", "spread", "total")

    result: list[str] = []
    for market in markets:
        mapped = _MARKET_PARAM_MAP.get(market.lower())
        if mapped and mapped not in result:
            result.append(mapped)
    return result


def _line_from_outcome(
    outcome: Mapping,
    market: str,
    *,
    game_id: str,
    game_label: str,
    kickoff: str,
    away: str,
    home: str,
    away_code: str,
    home_code: str,
    book: str,
    last_update: str,
) -> NormalizedLine | None:
    price_raw = outcome.get("price")
    if price_raw in (None, ""):
        return None

    try:
        price = int(price_raw)
    except (TypeError, ValueError):  # pragma: no cover - inconsistent API data
        return None

    side: str | None
    point: float | None = None

    if market == "ml":
        name = canonical_name(outcome.get("name", ""))
        if name == away:
            side = "away"
        elif name == home:
            side = "home"
        else:
            return None
    elif market == "spread":
        name = canonical_name(outcome.get("name", ""))
        if name == away:
            side = "away"
        elif name == home:
            side = "home"
        else:
            return None
        point_raw = outcome.get("point")
        if point_raw in (None, ""):
            return None
        point = float(point_raw)
    elif market == "total":
        name = str(outcome.get("name", "")).lower()
        if name.startswith("over"):
            side = "over"
        elif name.startswith("under"):
            side = "under"
        else:
            return None
        point_raw = outcome.get("point")
        if point_raw in (None, ""):
            return None
        point = float(point_raw)
    else:
        return None

    return NormalizedLine(
        game_id=game_id,
        game_label=game_label,
        kickoff=kickoff,
        away=away,
        home=home,
        away_code=away_code,
        home_code=home_code,
        market=market,
        side=side,
        book=book,
        price=price,
        point=point,
        last_update=last_update,
    )


__all__ = [
    "list_games",
    "list_games_window",
    "NormalizedLine",
    "OddsAPIClient",
    "OddsAPIConfig",
    "OddsAPIError",
]

