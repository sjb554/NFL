"""Edge calculations and simple recommendations.""" 

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

MIN_PRICE = -10000
MAX_PRICE = 10000


def fair_line_from_prob(prob: float) -> int:
    """Convert a win probability to American odds."""

    prob = max(min(prob, 0.9999), 0.0001)
    if prob >= 0.5:
        odds = -100 * prob / (1 - prob)
    else:
        odds = 100 * (1 - prob) / prob
    return int(round(odds))


def edge(model_p: float, market_p: float) -> float:
    """Return the probability edge versus the market implied probability."""

    return model_p - market_p


def kelly_stake(
    bankroll: float,
    b: float,
    p: float,
    q: float,
    *,
    fraction: float = 0.25,
    max_pct: float = 0.02,
) -> float:
    """Fractional Kelly sizing capped at a maximum bankroll percentage."""

    if bankroll <= 0 or b <= 0:
        return 0.0
    kelly = (b * p - q) / b
    if kelly <= 0:
        return 0.0
    stake = bankroll * fraction * kelly
    cap = bankroll * max_pct
    return float(min(stake, cap))


def recommend_bets(
    odds_rows: Sequence[Mapping[str, object]],
    model_probs: Mapping[tuple[str, str, str], float],
    *,
    bankroll: float,
    min_edge: float,
    fraction: float,
    max_pct: float,
) -> list[dict[str, object]]:
    """Return recommended bets that clear the minimum edge threshold."""

    recommendations: list[dict[str, object]] = []

    for row in odds_rows:
        key = (row["game"], row["market"], row["side"])
        model_p = model_probs.get(key)
        if model_p is None:
            continue

        price = int(row["price"])
        market_p = _american_to_prob(price)
        ev = edge(model_p, market_p)
        if ev <= min_edge:
            continue

        decimal_odds = _american_to_decimal(price)
        stake = kelly_stake(
            bankroll,
            decimal_odds - 1.0,
            model_p,
            1 - model_p,
            fraction=fraction,
            max_pct=max_pct,
        )
        if stake <= 0:
            continue

        recommendations.append(
            {
                "game": row["game"],
                "market": row["market"],
                "side": row["side"],
                "line": row.get("line"),
                "price": price,
                "book": row.get("book", ""),
                "model_prob": model_p,
                "implied_prob": market_p,
                "ev_pct": ev,
                "stake": stake,
                "fair_price": fair_line_from_prob(model_p),
                "last_update": row.get("last_update", ""),
            }
        )

    recommendations.sort(key=lambda item: item["ev_pct"], reverse=True)
    return recommendations


def _american_to_prob(price: int) -> float:
    price = max(min(price, MAX_PRICE), MIN_PRICE)
    if price >= 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def _american_to_decimal(price: int) -> float:
    price = max(min(price, MAX_PRICE), MIN_PRICE)
    if price >= 0:
        return 1 + price / 100.0
    return 1 + 100.0 / (-price)


__all__ = [
    "edge",
    "fair_line_from_prob",
    "kelly_stake",
    "recommend_bets",
]

