"""Command-line entry point for nfl-betting utilities."""

from __future__ import annotations

import argparse
import csv
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import requests
from dotenv import load_dotenv

from .config import AppConfig, load_config
from .models import predict_home_margin, predict_home_win_prob, predict_total_points
from .nflfastr import NFLFastRClient, TeamFeatures, get_team_features
from .odds_api import NormalizedLine, OddsAPIClient
from .pricing.edge import recommend_bets
from .teams import GameRequest, code_from_name, parse_games_argument

DEFAULT_MATCHUPS: list[tuple[str, str]] = [
    ("Tampa Bay Buccaneers", "Detroit Lions"),
    ("Houston Texans", "Seattle Seahawks"),
]
DEFAULT_GAMES_ARG = "TB@DET,HOU@SEA"
DEFAULT_BANKROLL = 1000.0
SPREAD_SIGMA = 6.0
TOTAL_SIGMA = 8.0
MARKET_PRINT_LABELS = {"ml": "Moneyline", "spread": "Spread", "total": "Total"}
MARKET_SIDE_ORDER = {
    "ml": ("away", "home"),
    "spread": ("away", "home"),
    "total": ("over", "under"),
}
MARKET_ORDER = ("ml", "spread", "total")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NFL betting helpers")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Attempt live data fetches instead of running the dry-run stub.",
    )
    parser.add_argument(
        "--lines",
        action="store_true",
        help="Fetch odds and print the best-line report for the supplied games.",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Fetch odds, run placeholder models, and print recommended bets.",
    )
    parser.add_argument(
        "--games",
        default=DEFAULT_GAMES_ARG,
        help="Comma-separated games in AWAY@HOME format for --lines/--recommend.",
    )
    parser.add_argument(
        "--date",
        help="Optional odds date in YYYY-MM-DD format (used with --lines/--recommend).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    modes = [args.fetch, args.lines, args.recommend]
    if sum(1 for flag in modes if flag) > 1:
        parser.error("Pick only one of --fetch, --lines, or --recommend.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    load_dotenv()
    config = load_config()

    if args.recommend:
        return run_recommend_flow(
            config=config,
            games_arg=args.games,
            requested_date=args.date,
        )

    if args.lines:
        return run_lines_flow(
            config=config,
            games_arg=args.games,
            requested_date=args.date,
        )

    if not args.fetch:
        print("everything wired correctly")
        return 0

    run_fetch(config)
    return 0


def run_fetch(config: AppConfig) -> None:
    """Execute the wiring fetch to confirm modules talk to each other."""

    logging.info("Starting live fetch with configured providers.")

    with OddsAPIClient(config.odds_api) as odds_client:
        odds = odds_client.fetch_odds(DEFAULT_MATCHUPS)
        for label, entries in odds.items():
            logging.info("Odds for %s: %d markets", label, len(entries))

    try:
        with NFLFastRClient(config.nflfastr) as fastr_client:
            stats = fastr_client.fetch_latest_team_stats(limit=4)
            for stat in stats:
                logging.info(
                    "nflfastR team=%s season=%s games=%s points_for=%.1f",
                    stat.team,
                    stat.season,
                    stat.games,
                    stat.points_for,
                )
    except requests.RequestException as exc:
        logging.warning("Unable to fetch nflfastR data: %s", exc)


def run_lines_flow(config: AppConfig, games_arg: str, requested_date: str | None) -> int:
    """Run the odds fetch + best-line report flow."""

    games = _parse_games(games_arg)
    if not games:
        return 1

    requested_markets = {_canonical_market_name(m) for m in config.settings.markets} or set(MARKET_ORDER)

    lines, date_str, date_slug = _retrieve_lines(config, games, requested_date)
    if not lines:
        print("No odds available (API key/rate limit)?")
        return 0

    best_rows = _compile_best_lines(lines, games, requested_markets)
    if not best_rows:
        print("No odds available (API key/rate limit)?")
        return 0

    _print_best_line_table(best_rows, games)
    _write_lines_csv(best_rows, date_slug)
    return 0


def run_recommend_flow(config: AppConfig, games_arg: str, requested_date: str | None) -> int:
    """Run odds + features + placeholder models to produce recommendations."""

    games = _parse_games(games_arg)
    if not games:
        return 1

    season = max(config.settings.train_seasons or [datetime.now(timezone.utc).year])
    weeks = max(config.settings.lookback_weeks, 1)
    features = get_team_features(weeks=weeks, season=season)
    if not features:
        print("Insufficient data to model (nflfastR weekly stats missing).")
        return 0

    lines, date_str, date_slug = _retrieve_lines(config, games, requested_date)
    if not lines:
        print("No odds available (API key/rate limit)?")
        return 0

    requested_markets = {_canonical_market_name(m) for m in config.settings.markets} or set(MARKET_ORDER)
    best_rows = _compile_best_lines(lines, games, requested_markets)
    if not best_rows:
        print("No odds available (API key/rate limit)?")
        return 0

    league_avg = features.get("LEAGUE_AVG")
    game_context = _build_game_context(games, features, league_avg)
    if not game_context:
        print("Insufficient data to model (team features missing).")
        return 0

    odds_rows: list[dict[str, object]] = []
    model_probs: dict[tuple[str, str, str], float] = {}

    for game, market, side, line in best_rows:
        context = game_context.get(game.label)
        if not context:
            continue

        key = (game.label, market, side)
        if market == "ml":
            home_prob = context["home_prob"]
            model_prob = home_prob if side == "home" else 1 - home_prob
        elif market == "spread" and line.point is not None:
            model_prob = _spread_cover_probability(
                context["expected_margin"],
                float(line.point),
                side,
            )
        elif market == "total" and line.point is not None:
            model_prob = _total_probability(
                context["predicted_total"],
                float(line.point),
                side,
            )
        else:
            continue

        model_probs[key] = model_prob
        odds_rows.append(
            {
                "game": game.label,
                "market": market,
                "side": side,
                "price": line.price,
                "book": line.book,
                "line": _csv_line_value(line),
                "last_update": line.last_update,
            }
        )

    if not odds_rows:
        print("Insufficient data to model (no lines matched).")
        return 0

    recommendations = recommend_bets(
        odds_rows,
        model_probs,
        bankroll=DEFAULT_BANKROLL,
        min_edge=config.settings.min_ev,
        fraction=config.settings.kelly_fraction,
        max_pct=config.settings.max_stake_pct,
    )

    if not recommendations:
        print("No actionable edges (EV <= {0:.0%}).".format(config.settings.min_ev))
        return 0

    _print_recommendations_table(recommendations)
    _write_recommendations_csv(recommendations, date_slug)
    return 0


def _parse_games(arg: str) -> list[GameRequest]:
    try:
        return parse_games_argument(arg)
    except ValueError as exc:
        logging.error("%s", exc)
        return []


def _retrieve_lines(
    config: AppConfig,
    games: Sequence[GameRequest],
    requested_date: str | None,
) -> tuple[list[NormalizedLine], str, str]:
    date_str = requested_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_slug = date_str.replace("-", "")
    cache_dir = Path("data_cache") / "odds" / date_slug

    with OddsAPIClient(config.odds_api) as odds_client:
        lines = odds_client.fetch_lines(
            games,
            markets=config.settings.markets,
            date=date_str,
            cache_dir=cache_dir,
        )
    return lines, date_str, date_slug


def _build_game_context(
    games: Sequence[GameRequest],
    features: Mapping[str, TeamFeatures],
    league_avg: TeamFeatures | None,
) -> dict[str, dict[str, float]]:
    context: dict[str, dict[str, float]] = {}
    for game in games:
        away_code = game.away_code
        home_code = game.home_code
        if not away_code or not home_code:
            logging.warning("Missing team code for matchup %s", game.label)
            continue

        home_feat = features.get(home_code) or league_avg
        away_feat = features.get(away_code) or league_avg
        if not home_feat or not away_feat:
            logging.warning("Missing features for matchup %s", game.label)
            continue

        home_strength = _team_strength(home_feat)
        away_strength = _team_strength(away_feat)
        home_prob = predict_home_win_prob(home_strength, away_strength)
        expected_margin = predict_home_margin(home_strength, away_strength)
        predicted_total = predict_total_points(
            home_feat.pace,
            away_feat.pace,
            home_feat.offensive_epa_per_play,
            away_feat.offensive_epa_per_play,
            home_feat.defensive_epa_per_play,
            away_feat.defensive_epa_per_play,
        )

        context[game.label] = {
            "home_prob": float(_clamp(home_prob, 0.01, 0.99)),
            "expected_margin": float(expected_margin),
            "predicted_total": float(predicted_total),
        }

    return context


def _compile_best_lines(
    lines: Sequence[NormalizedLine],
    games: Sequence[GameRequest],
    requested_markets: set[str],
) -> list[tuple[GameRequest, str, str, NormalizedLine]]:
    grouped: dict[tuple[str, str], list[NormalizedLine]] = defaultdict(list)
    for line in lines:
        grouped[(line.away, line.home)].append(line)

    ordered_markets = [m for m in MARKET_ORDER if m in requested_markets] or list(MARKET_ORDER)

    selections: list[tuple[GameRequest, str, str, NormalizedLine]] = []
    for game in games:
        game_lines = grouped.get(game.key, [])
        if not game_lines:
            continue
        for market in ordered_markets:
            for side in MARKET_SIDE_ORDER.get(market, ()):  # pragma: no branch - defensive
                best = _best_line(game_lines, market, side)
                if best:
                    selections.append((game, market, side, best))
    return selections


def _best_line(lines: Sequence[NormalizedLine], market: str, side: str) -> NormalizedLine | None:
    candidates = [line for line in lines if line.market == market and line.side == side]
    if not candidates:
        return None

    def priority(line: NormalizedLine) -> tuple[float, int]:
        point = line.point if line.point is not None else 0.0
        if market == "ml":
            return (0.0, line.price)
        if market == "spread":
            return (point, line.price)
        if market == "total" and side == "over":
            return (-point, line.price)
        if market == "total" and side == "under":
            return (point, line.price)
        return (point, line.price)

    return max(candidates, key=priority)


def _print_best_line_table(
    selections: Sequence[tuple[GameRequest, str, str, NormalizedLine]],
    games: Sequence[GameRequest],
) -> None:
    rows_by_game: dict[str, list[tuple[str, str, NormalizedLine]]] = defaultdict(list)
    for game, market, side, line in selections:
        rows_by_game[game.label].append((market, side, line))

    for game in games:
        label = game.label
        rows = rows_by_game.get(label, [])
        print(f"\n{label}")
        print("-" * len(label))
        if not rows:
            print("No odds available (API key/rate limit)?")
            continue

        header = f"{'Market':<10} {'Selection':<28} {'Line':>8} {'Price':>6} {'Book':<18} {'Last Update'}"
        print(header)
        print("-" * len(header))
        for market, side, line in rows:
            market_label = MARKET_PRINT_LABELS.get(market, market.title())
            selection_label = _format_selection_label(line, side)
            line_value = _format_line_value(line)
            price_value = f"{line.price:+d}"
            last_update = _format_last_update(line.last_update)
            print(
                f"{market_label:<10} {selection_label:<28} {line_value:>8} {price_value:>6} "
                f"{line.book:<18} {last_update}"
            )


def _print_recommendations_table(recommendations: Sequence[Mapping[str, object]]) -> None:
    header = (
        f"{'Game':<28} {'Market':<8} {'Side':<6} {'Line':>8} {'Price':>6} "
        f"{'Model%':>8} {'Implied%':>9} {'Edge%':>7} {'Stake':>8} {'Book':<14} {'Fair':>6}"
    )
    print("\nRecommendations")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for rec in recommendations:
        model_pct = rec['model_prob'] * 100
        implied_pct = rec['implied_prob'] * 100
        edge_pct = rec['ev_pct'] * 100
        stake_value = f"${rec['stake']:.2f}"
        print(
            f"{rec['game']:<28} {rec['market']:<8} {rec['side']:<6} {str(rec['line'] or '--'):>8} "
            f"{rec['price']:+6d} {model_pct:>7.2f}% {implied_pct:>8.2f}% {edge_pct:>6.2f}% "
            f"{stake_value:>8} {str(rec['book']):<14} {rec['fair_price']:>6d}"
        )


def _write_lines_csv(
    selections: Sequence[tuple[GameRequest, str, str, NormalizedLine]],
    date_slug: str,
) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"lines_{date_slug}.csv"

    fieldnames = ["game", "market", "side", "line", "price", "book", "last_update"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for game, market, side, line in selections:
            writer.writerow(
                {
                    "game": game.label,
                    "market": market,
                    "side": side,
                    "line": _csv_line_value(line),
                    "price": f"{line.price:+d}",
                    "book": line.book,
                    "last_update": line.last_update,
                }
            )


def _write_recommendations_csv(
    recommendations: Sequence[Mapping[str, object]],
    date_slug: str,
) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"recommend_{date_slug}.csv"

    fieldnames = [
        "game",
        "market",
        "side",
        "line",
        "price",
        "book",
        "model_prob",
        "implied_prob",
        "ev_pct",
        "stake",
        "fair_price",
        "last_update",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in recommendations:
            writer.writerow(
                {
                    "game": rec["game"],
                    "market": rec["market"],
                    "side": rec["side"],
                    "line": rec.get("line", ""),
                    "price": rec["price"],
                    "book": rec.get("book", ""),
                    "model_prob": round(rec["model_prob"], 4),
                    "implied_prob": round(rec["implied_prob"], 4),
                    "ev_pct": round(rec["ev_pct"], 4),
                    "stake": round(rec["stake"], 2),
                    "fair_price": rec["fair_price"],
                    "last_update": rec.get("last_update", ""),
                }
            )


def _format_selection_label(line: NormalizedLine, side: str) -> str:
    if line.market in {"ml", "spread"}:
        prefix = "Away" if side == "away" else "Home"
        team = line.away if side == "away" else line.home
        return f"{prefix} ({team})"
    return side.capitalize()


def _format_line_value(line: NormalizedLine) -> str:
    if line.point is None:
        return "--"
    if line.market == "spread":
        return f"{line.point:+g}"
    return f"{line.point:g}"


def _csv_line_value(line: NormalizedLine) -> str:
    if line.point is None:
        return ""
    if line.market == "spread":
        return f"{line.point:+g}"
    return f"{line.point:g}"


def _format_last_update(value: str) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def _canonical_market_name(market: str) -> str:
    lowered = market.lower()
    if lowered in {"ml", "moneyline", "h2h"}:
        return "ml"
    if lowered in {"spread", "spreads"}:
        return "spread"
    if lowered in {"total", "totals"}:
        return "total"
    return lowered


def _team_strength(features: TeamFeatures) -> float:
    """Derive a crude strength score from feature placeholders."""

    return (
        features.offensive_epa_per_play
        + features.defensive_epa_per_play
        + (features.success_rate_off - 0.5)
        + (features.success_rate_def - 0.5)
        - features.turnover_rate
    )


def _spread_cover_probability(expected_margin: float, line_point: float, side: str) -> float:
    team_margin = expected_margin if side == "home" else -expected_margin
    delta = (team_margin + line_point) / SPREAD_SIGMA
    return _sigmoid(delta)


def _total_probability(predicted_total: float, line_point: float, side: str) -> float:
    delta = (predicted_total - line_point) / TOTAL_SIGMA
    if side == "under":
        delta = -delta
    return _sigmoid(delta)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
