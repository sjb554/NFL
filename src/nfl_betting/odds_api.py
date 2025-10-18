"""Thin client for interacting with The Odds API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import requests

from .config import OddsAPIConfig
from .teams import GameRequest, build_game_lookup, canonical_name

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
    away: str
    home: str
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

        if not self._config.api_key:
            logger.warning("ODDS_API_KEY not set; skipping live odds fetch.")
            return []

        market_params = _select_markets(markets)
        game_lookup = build_game_lookup(games)
        results: list[NormalizedLine] = []

        for market in market_params:
            payload = self._load_market_payload(
                market,
                date=date,
                cache_dir=cache_dir,
                max_cache_age_hours=max_cache_age_hours,
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
    ) -> list[Mapping]:
        cache_file: Path | None = None
        if cache_dir:
            cache_base = Path(cache_dir)
            cache_base.mkdir(parents=True, exist_ok=True)
            cache_file = cache_base / f"{market}.json"
            if cache_file.exists():
                modified = datetime.fromtimestamp(
                    cache_file.stat().st_mtime, tz=timezone.utc
                )
                age = datetime.now(timezone.utc) - modified
                if age <= timedelta(hours=max_cache_age_hours):
                    try:
                        return json.loads(cache_file.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:  # pragma: no cover - fallback
                        logger.debug("Cache decode failed for %s", cache_file)

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
            game_id = str(event.get("id", ""))
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
                            away=away,
                            home=home,
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
    away: str,
    home: str,
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
        away=away,
        home=home,
        market=market,
        side=side,
        book=book,
        price=price,
        point=point,
        last_update=last_update,
    )


__all__ = [
    "NormalizedLine",
    "OddsAPIClient",
    "OddsAPIConfig",
    "OddsAPIError",
]
