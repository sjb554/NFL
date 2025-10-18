"""Team mapping utilities for odds lookups."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional

_NORMALIZER = re.compile(r"[^a-z0-9]")


def _normalize(value: str) -> str:
    return _NORMALIZER.sub("", value.lower())


_TEAM_ABBREVIATIONS: dict[str, str] = {
    "TB": "Tampa Bay Buccaneers",
    "DET": "Detroit Lions",
    "HOU": "Houston Texans",
    "SEA": "Seattle Seahawks",
}

_EXTRA_ALIASES = {
    "tampa bay bucs": "Tampa Bay Buccaneers",
    "detroit": "Detroit Lions",
    "houston": "Houston Texans",
    "seattle": "Seattle Seahawks",
}

_TEAM_ALIASES: dict[str, str] = {
    **{_normalize(name): name for name in _TEAM_ABBREVIATIONS.values()},
    **{_normalize(alias): canonical for alias, canonical in _EXTRA_ALIASES.items()},
}


@dataclass(frozen=True)
class GameRequest:
    """Represents a requested matchup using canonical team names."""

    away: str
    home: str

    @property
    def label(self) -> str:
        return f"{self.away} @ {self.home}"

    @property
    def key(self) -> tuple[str, str]:
        return canonical_name(self.away), canonical_name(self.home)

    @property
    def away_code(self) -> Optional[str]:
        return code_from_name(self.away)

    @property
    def home_code(self) -> Optional[str]:
        return code_from_name(self.home)


def parse_games_argument(arg: str) -> list[GameRequest]:
    """Parse a comma-separated AWAY@HOME string into GameRequest objects."""

    if not arg:
        raise ValueError("games argument cannot be empty")

    games: List[GameRequest] = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        if "@" not in token:
            raise ValueError(f"Invalid game token '{token}'. Expected AWAY@HOME format.")
        away_code, home_code = [part.strip().upper() for part in token.split("@", 1)]
        away_name = name_from_code(away_code)
        home_name = name_from_code(home_code)
        games.append(GameRequest(away=away_name, home=home_name))

    if not games:
        raise ValueError("No valid games provided.")

    return games


def name_from_code(code: str) -> str:
    """Return the canonical team name for a two or three letter code."""

    try:
        return _TEAM_ABBREVIATIONS[code]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported team code '{code}'.") from exc


def canonical_name(name: str) -> str:
    """Return the canonical form for a team name or alias."""

    normalized = _normalize(name)
    return _TEAM_ALIASES.get(normalized, name.strip())


def code_from_name(name: str) -> Optional[str]:
    """Return the team code for a provided team name or alias."""

    normalized = _normalize(name)
    for code, full_name in _TEAM_ABBREVIATIONS.items():
        if _normalize(full_name) == normalized:
            return code
    alias_target = _TEAM_ALIASES.get(normalized)
    if not alias_target:
        return None
    for code, full_name in _TEAM_ABBREVIATIONS.items():
        if full_name == alias_target:
            return code
    return None


def build_game_lookup(games: Iterable[GameRequest]) -> dict[tuple[str, str], GameRequest]:
    """Build a lookup of canonical (away, home) pairs to requested games."""

    return {game.key: game for game in games}


__all__ = [
    "GameRequest",
    "build_game_lookup",
    "canonical_name",
    "code_from_name",
    "name_from_code",
    "parse_games_argument",
]
