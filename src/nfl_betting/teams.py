"""Team mapping utilities for odds lookups."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List

_NORMALIZER = re.compile(r"[^a-z0-9]")

CANONICAL: dict[str, str] = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

ALIASES: dict[str, str] = {
    "ARZ": "ARI",
    "CRD": "ARI",
    "JAC": "JAX",
    "JAGS": "JAX",
    "NOR": "NO",
    "NOL": "NO",
    "LA": "LAR",
    "SD": "LAC",
    "OAK": "LV",
    "WFT": "WAS",
    "WSH": "WAS",
    "NWE": "NE",
    "GNB": "GB",
    "KAN": "KC",
    "SFO": "SF",
    "TAM": "TB",
    "RAV": "BAL",
    "HTX": "HOU",
    "CLT": "IND",
    "RAI": "LV",
    "RAM": "LAR",
    "CHR": "LAC",
    "SEAHAWKS": "SEA",
    "SEATTLE": "SEA",
    "TEXANS": "HOU",
    "BUCCANEERS": "TB",
    "BUCS": "TB",
    "SAINTS": "NO",
    "PACKERS": "GB",
    "JETS": "NYJ",
    "GIANTS": "NYG",
    "49ERS": "SF",
    "NINERS": "SF",
    "PATS": "NE",
    "PATRIOTS": "NE",
    "VIKINGS": "MIN",
    "DOLPHINS": "MIA",
    "BILLS": "BUF",
    "BEARS": "CHI",
    "BENGALS": "CIN",
    "BROWNS": "CLE",
    "COWBOYS": "DAL",
    "BRONCOS": "DEN",
    "LIONS": "DET",
    "TITANS": "TEN",
    "EAGLES": "PHI",
    "STEELERS": "PIT",
    "RAVENS": "BAL",
    "CHARGERS": "LAC",
    "RAMS": "LAR",
    "RAIDERS": "LV",
    "COMMANDERS": "WAS",
    "JAGUARS": "JAX",
    "CARDINALS": "ARI",
    "FALCONS": "ATL",
    "COLTS": "IND",
    "CHIEFS": "KC",
}

ALIASES_LOOKUP: dict[str, str] = {
    _NORMALIZER.sub("", key.lower()): value for key, value in ALIASES.items()
}

for code, name in CANONICAL.items():
    ALIASES_LOOKUP[_NORMALIZER.sub("", name.lower())] = code


@dataclass(frozen=True)
class GameRequest:
    away_code: str
    home_code: str

    @property
    def away(self) -> str:
        return CANONICAL[self.away_code]

    @property
    def home(self) -> str:
        return CANONICAL[self.home_code]

    @property
    def label(self) -> str:
        return f"{self.away} @ {self.home}"

    @property
    def key(self) -> tuple[str, str]:
        return (self.away, self.home)


def normalize_team(value: str) -> str:
    if not value:
        raise ValueError("Team value cannot be empty.")
    code = value.strip().upper()
    if code in CANONICAL:
        return code
    normalized = _NORMALIZER.sub("", value.lower())
    if normalized in ALIASES_LOOKUP:
        return ALIASES_LOOKUP[normalized]
    raise ValueError(f"Unsupported team code: {value}")


def parse_games_argument(arg: str) -> list[GameRequest]:
    if not arg:
        raise ValueError("games argument cannot be empty")

    games: List[GameRequest] = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        if "@" not in token:
            raise ValueError(f"Invalid game token '{token}'. Expected AWAY@HOME format.")
        away_raw, home_raw = [part.strip() for part in token.split("@", 1)]
        away_code = normalize_team(away_raw)
        home_code = normalize_team(home_raw)
        games.append(GameRequest(away_code=away_code, home_code=home_code))

    if not games:
        raise ValueError("No valid games provided.")

    return games


def name_from_code(code: str) -> str:
    canonical_code = normalize_team(code)
    return CANONICAL[canonical_code]


def canonical_name(name: str) -> str:
    code = normalize_team(name)
    return CANONICAL[code]


def code_from_name(name: str) -> str:
    return normalize_team(name)


def build_game_lookup(games: Iterable[GameRequest]) -> dict[tuple[str, str], GameRequest]:
    return {game.key: game for game in games}


__all__ = [
    "CANONICAL",
    "ALIASES",
    "GameRequest",
    "build_game_lookup",
    "canonical_name",
    "code_from_name",
    "name_from_code",
    "normalize_team",
    "parse_games_argument",
]
