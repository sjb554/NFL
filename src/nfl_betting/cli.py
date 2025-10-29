"""Command-line entry point for nfl-betting utilities."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import json
import os
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover - Windows fallback
    try:
        from backports.zoneinfo import ZoneInfo  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        ZoneInfo = None  # type: ignore

import pandas as pd
import requests
from dotenv import load_dotenv

from .actuals import load_actuals
from .config import AppConfig, load_config
from .models import predict_home_margin, predict_home_win_prob, predict_total_points
from .nflfastr import NFLFastRClient, TeamFeatures, get_team_features
from .odds_api import NormalizedLine, OddsAPIClient, list_games, list_games_window
from .pricing.edge import ev_from_american, recommend_bets
from .report import (
    copy_latest_reports_to_site,
    render_html,
    render_markdown,
    write_report,
)
from .teams import GameRequest, parse_games_argument

DEFAULT_MATCHUPS: list[tuple[str, str]] = [
    ("Tampa Bay Buccaneers", "Detroit Lions"),
    ("Houston Texans", "Seattle Seahawks"),
]
DEFAULT_GAMES_ARG = "TB@DET,HOU@SEA"
SPREAD_SIGMA = 6.0
TOTAL_SIGMA = 14.0
MARKET_PRINT_LABELS = {"ml": "Moneyline", "spread": "Spread", "total": "Total"}
MARKET_SIDE_ORDER = {
    "ml": ("away", "home"),
    "spread": ("away", "home"),
    "total": ("over", "under"),
}
MARKET_ORDER = ("ml", "spread", "total")
MARKET_COMPACT_LABELS = {"ml": "ML", "spread": "Spread", "total": "Total"}
BEST_LINE_FIELDS = ("Side", "Line", "Price", "Book")
VEGAS_LOG_DIR = Path("artifacts")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NFL betting helpers")
    parser.add_argument("--fetch", action="store_true", help="Attempt live data fetches instead of running the dry-run stub.")
    parser.add_argument("--lines", action="store_true", help="Fetch odds and print the best-line report for the supplied games.")
    parser.add_argument("--recommend", action="store_true", help="Fetch odds, run placeholder models, and print recommended bets.")
    parser.add_argument("--report", action="store_true", help="Generate Markdown + HTML report for the selected flow.")
    parser.add_argument("--publish", action="store_true", help="Copy the generated HTML report to docs/index.html (for GitHub Pages).")
    parser.add_argument("--open", dest="open_report", action="store_true", help="Open the generated HTML report in the default browser.")
    parser.add_argument("--vegas", action="store_true", help="Run the entire pipeline end-to-end with Vegas guardrails.")
    parser.add_argument("--games", default=DEFAULT_GAMES_ARG, help="Comma-separated games in AWAY@HOME format for --lines/--recommend/--vegas.")
    parser.add_argument(
        "--slate",
        choices=["today", "sunday", "date", "week"],
        help="Auto-populate the NFL slate for the given day or week window.",
    )
    parser.add_argument(
        "--week-days",
        type=int,
        default=7,
        help="Number of days to include when using --slate week (default: 7).",
    )
    parser.add_argument("--export-games", action="store_true", help="Write the resolved slate to reports/games_<date>.txt.")
    parser.add_argument("--date", help="Optional odds date in YYYY-MM-DD format (used with --lines/--recommend/--vegas).")
    parser.add_argument("--bankroll", type=float, help="Override bankroll when running --recommend/--vegas.")
    parser.add_argument("--kelly", type=float, help="Override Kelly fraction (0-1) when running --recommend/--vegas.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    mode_flags = [args.fetch, args.lines, args.recommend, args.vegas]
    if sum(1 for flag in mode_flags if flag) > 1:
        parser.error("Pick only one of --fetch, --lines, --recommend, or --vegas.")

    if args.publish and not (args.report or args.vegas):
        parser.error("--publish requires --report or --vegas.")
    if args.open_report and not (args.report or args.vegas):
        parser.error("--open requires --report or --vegas.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    load_dotenv()
    config = load_config()
    resolved_games = args.games
    resolved_dates: list[str] = [args.date] if args.date else []

    if args.slate:
        if args.slate == "date" and not args.date:
            parser.error("--slate date requires --date YYYY-MM-DD")
        if args.slate == "week":
            start_date = _resolve_slate_date("week", args.date)
            days = max(args.week_days, 1)
            window = list_games_window(
                start_date,
                config=config.odds_api,
                days=days,
            )
            if not window:
                print(
                    f"No games available for the window starting {start_date} (check API key or schedule)."
                )
                return 0
            ordered_dates = sorted(window.keys())
            resolved_dates = ordered_dates
            games_seen: set[str] = set()
            games_list: list[str] = []
            for date_key in ordered_dates:
                for token in window[date_key]:
                    if token not in games_seen:
                        games_list.append(token)
                        games_seen.add(token)
            resolved_games = ",".join(games_list)
            if ordered_dates:
                args.date = ordered_dates[0]
            if args.export_games:
                start_slug = ordered_dates[0].replace("-", "")
                end_slug = ordered_dates[-1].replace("-", "")
                export_path = Path("reports") / f"games_{start_slug}_{end_slug}.txt"
                export_lines: list[str] = []
                for date_key in ordered_dates:
                    export_lines.append(f"# {date_key}")
                    export_lines.extend(window[date_key])
                    export_lines.append("")
                export_path.parent.mkdir(parents=True, exist_ok=True)
                export_path.write_text("\n".join(export_lines).strip() + "\n", encoding="utf-8")
                print(f"Exported games list to {export_path}")
        else:
            resolved_date = _resolve_slate_date(args.slate, args.date)
            games_list = list_games(
                resolved_date,
                config=config.odds_api,
            )
            if not games_list:
                print(
                    f"No games available for {resolved_date} (check API key or schedule)."
                )
                return 0
            resolved_games = ",".join(games_list)
            resolved_dates = [resolved_date]
            args.date = resolved_date
            if args.export_games:
                export_path = Path("reports") / f"games_{resolved_date.replace('-', '')}.txt"
                export_path.parent.mkdir(parents=True, exist_ok=True)
                export_path.write_text("\n".join(games_list) + "\n", encoding="utf-8")
                print(f"Exported games list to {export_path}")
    elif args.export_games:
        parser.error("--export-games requires --slate")


    args.games = resolved_games
    requested_dates = resolved_dates if resolved_dates else None

    report_opts = {
        "report": args.report,
        "publish": args.publish,
        "open": args.open_report,
    }

    bankroll_override = args.bankroll
    kelly_override = args.kelly
    vegas_options: dict[str, object] | None = None

    if args.vegas:
        args.recommend = True
        report_opts["report"] = True
        if args.publish:
            report_opts["publish"] = True
        if args.open_report:
            report_opts["open"] = True
        if not requested_dates:
            next_monday = _next_monday_pt().strftime("%Y-%m-%d")
            requested_dates = [next_monday]
            args.date = next_monday
        else:
            args.date = requested_dates[0]
        bankroll_override = bankroll_override or config.vegas.default_bankroll
        vegas_options = {
            "max_total_stake_pct": config.vegas.max_total_stake_pct,
            "min_books": config.vegas.min_books_per_selection,
            "stale_minutes": config.vegas.stale_minutes_ok,
            "default_bankroll": config.vegas.default_bankroll,
        }
        if requested_dates:
            if len(requested_dates) > 1:
                logging.info(
                    "Vegas mode running for %s to %s",
                    requested_dates[0],
                    requested_dates[-1],
                )
            else:
                logging.info("Vegas mode running for %s", requested_dates[0])
        else:
            logging.info("Vegas mode running without a fixed slate date")

    if args.recommend and bankroll_override is not None and bankroll_override <= 0:
        parser.error("--bankroll must be positive.")
    if args.recommend and kelly_override is not None and not (0 < kelly_override <= 1):
        parser.error("--kelly must be between 0 and 1.")

    if args.vegas and not args.games:
        args.games = DEFAULT_GAMES_ARG_GAMES_ARG

    if not (args.fetch or args.lines or args.recommend or args.vegas):
        # default dry-run
        print("everything wired correctly")
        return 0

    if args.fetch:
        run_fetch(config)
        return 0

    if args.lines and not args.recommend:
        return run_lines_flow(
            config=config,
            games_arg=args.games,
            requested_dates=requested_dates,
            report_opts=report_opts,
        )

    return run_recommend_flow(
        config=config,
        games_arg=args.games,
        requested_dates=requested_dates,
        report_opts=report_opts,
        bankroll_override=bankroll_override,
        kelly_override=kelly_override,
        vegas_mode=args.vegas,
        vegas_options=vegas_options,
    )



def _next_monday_pt(now: datetime | None = None, *, as_date_str: bool = False):
    """Return the next Monday in Pacific Time (start-of-day)."""
    tz = ZoneInfo('America/Los_Angeles') if ZoneInfo else timezone(timedelta(hours=-8))
    current = now.astimezone(tz) if now else datetime.now(tz)
    days_ahead = (7 - current.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    target = (current + timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)
    return target.date().isoformat() if as_date_str else target

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


def run_lines_flow(
    config: AppConfig,
    games_arg: str,
    requested_dates: Sequence[str] | None,
    report_opts: Mapping[str, bool],
) -> int:
    games = _parse_games(games_arg)
    if not games:
        return 1

    requested_markets = {_canonical_market_name(m) for m in config.settings.markets} or set(MARKET_ORDER)

    lines, date_label, date_slug, cache_dir, slate_dates = _retrieve_lines(
        config,
        games,
        requested_dates,
    )
    if not lines:
        print("No odds available (API key/rate limit)?")
        return 0

    best_rows = _compile_best_lines(lines, games, requested_markets)
    if not best_rows:
        print("No odds available (API key/rate limit)?")
        return 0

    _print_best_line_table(best_rows, games)
    _write_lines_csv(best_rows, date_slug)

    if report_opts.get("report"):
        best_df = _best_lines_dataframe(best_rows)
        context = _build_report_context(
            date_str=date_label,
            date_slug=date_slug,
            games=games,
            config=config,
            odds_cache=cache_dir,
            features_metadata=None,
            vegas_meta=None,
            safety_notes=None,
            slate_dates=slate_dates,
        )
        paths = _generate_report(
            best_lines_df=best_df,
            recs_df=None,
            context=context,
            publish=report_opts.get("publish", False),
            open_report=report_opts.get("open", False),
        )
        if report_opts.get("publish"):
            copied = copy_latest_reports_to_site()
            for label, dest in copied.items():
                print(f"Copied {label} CSV to {dest}")

    return 0


def run_recommend_flow(
    config: AppConfig,
    games_arg: str,
    requested_dates: Sequence[str] | None,
    report_opts: Mapping[str, bool],
    *,
    bankroll_override: float | None = None,
    kelly_override: float | None = None,
    vegas_mode: bool = False,
    vegas_options: Mapping[str, object] | None = None,
) -> int:
    games = _parse_games(games_arg)
    if not games:
        return 1

    season = max(config.settings.train_seasons or [datetime.now(timezone.utc).year])
    weeks = max(config.settings.lookback_weeks, 1)
    feature_teams: list[str] = []
    for game in games:
        if game.away_code:
            feature_teams.append(game.away_code)
        feature_teams.append(game.away)
        if game.home_code:
            feature_teams.append(game.home_code)
        feature_teams.append(game.home)
    feature_teams = list(dict.fromkeys(ft for ft in feature_teams if ft))
    features, features_meta = get_team_features(
        feature_teams,
        weeks=weeks,
        season=season,
        return_metadata=True,
    )
    if not features:
        print("Insufficient data to model (nflfastR weekly stats missing).")
        return 0

    lines, date_label, date_slug, cache_dir, slate_dates = _retrieve_lines(
        config,
        games,
        requested_dates,
    )
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

    odds_rows, model_probs, game_entries = _prepare_model_inputs(best_rows, game_context)

    total_rows = len(odds_rows)
    sanity_flagged = 0
    sanity_note: str | None = None
    filtered_odds: list[dict[str, object]] = []
    filtered_model_probs: dict[tuple[str, str, str], float] = {}

    clamp_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in odds_rows:
        key = (row['game_id'], row['market'], row['side'])
        prob = model_probs.get(key)
        price = int(row['price'])
        market = str(row.get('market', ''))
        lower, upper = (0.001, 0.999) if market in {'spread', 'total'} else (0.01, 0.99)
        reason: str | None = None
        if prob is None or math.isnan(prob) or not (lower <= prob <= upper):
            reason = 'probability_out_of_bounds'
        elif market == 'ml' and (prob >= 0.95 or prob <= 0.05):
            reason = 'extreme_probability'
        elif market == 'ml' and prob > 0.75 and price >= 100:
            reason = 'high_prob_plus_price'
        elif market == 'ml' and prob < 0.25 and price <= -200:
            reason = 'low_prob_heavy_favorite'
        if reason:
            sanity_flagged += 1
            clamp_counts[market][reason] += 1
            continue
        filtered_odds.append(row)
        filtered_model_probs[key] = prob

    if filtered_odds:
        odds_rows = filtered_odds
        model_probs = filtered_model_probs

    debug_clamp_counts(clamp_counts)

    if total_rows and sanity_flagged / total_rows >= 0.10:
        raise RuntimeError('model probabilities missing/misaligned')

    if sanity_flagged:
        sanity_note = f"Sanity filters dropped {sanity_flagged} of {total_rows} odds rows ({sanity_flagged / total_rows:.1%})."

    bankroll = bankroll_override if bankroll_override is not None else config.vegas.default_bankroll
    kelly_fraction = kelly_override if kelly_override is not None else config.settings.kelly_fraction

    recommendations = recommend_bets(
        odds_rows,
        model_probs,
        bankroll=bankroll,
        min_edge=config.settings.min_ev,
        fraction=kelly_fraction,
        max_pct=config.settings.max_stake_pct,
    )

    out_dir = Path('out')
    out_dir.mkdir(parents=True, exist_ok=True)
    reco_df = pd.DataFrame(recommendations).copy()
    desired_cols = [
        'game_id',
        'game',
        'kickoff',
        'market',
        'side',
        'line',
        'price',
        'book',
        'model_prob',
        'implied_prob',
        'ev_pct',
        'stake',
        'fair_price',
        'last_update',
    ]
    string_defaults = {
        'game_id': '',
        'game': '',
        'kickoff': '',
        'market': '',
        'side': '',
        'line': '',
        'book': '',
        'last_update': '',
    }
    numeric_defaults = {
        'price': 0,
        'model_prob': 0.0,
        'implied_prob': 0.0,
        'ev_pct': 0.0,
        'stake': 0.0,
        'fair_price': 0,
    }
    for col in desired_cols:
        if col not in reco_df.columns:
            if col in string_defaults:
                reco_df[col] = string_defaults[col]
            elif col in numeric_defaults:
                reco_df[col] = numeric_defaults[col]
            else:
                reco_df[col] = ''
    reco_df = reco_df[desired_cols]
    reco_df.to_csv(out_dir / 'reco_table.csv', index=False)

    game_ids = [rec.get('game_id') for rec in recommendations if rec.get('game_id')]
    results_df = load_actuals(
        dates=slate_dates or [date_label],
        game_ids=game_ids,
    )
    accuracy_df = _build_accuracy_table(recommendations, results_df)
    _write_accuracy_csv(accuracy_df, out_dir, date_slug)
    _write_summary(reco_df, accuracy_df, out_dir, date_slug)

    run_time = datetime.now(timezone.utc)

    vegas_summary: dict[str, object] | None = None
    safety_notes: list[str] = []
    if sanity_note:
        safety_notes.append(sanity_note)
    skipped_bets: list[tuple[Mapping[str, object], list[str]]] = []

    if vegas_mode:
        guardrails = vegas_options or {}
        max_total_pct = float(guardrails.get("max_total_stake_pct", 0.05))
        min_books = int(guardrails.get("min_books", 3))
        stale_minutes = int(guardrails.get("stale_minutes", 60))
        safety_notes.append(
            f"Guardrails in effect: total stake cap {max_total_pct:.1%}, minimum {min_books} books per side, odds newer than {stale_minutes} minutes."
        )
        book_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        best_line_lookup: dict[tuple[str, str, str], NormalizedLine] = {}
        for game, market, side, line in best_rows:
            key = (game.label, market, side)
            best_line_lookup[key] = line
        for line in lines:
            key = (line.game_label, line.market, line.side)
            book_sets[key].add(line.book)

        filtered: list[dict[str, object]] = []
        for rec in recommendations:
            key = (rec["game"], rec["market"], rec["side"])
            reasons: list[str] = []
            book_count = len(book_sets.get(key, set()))
            if min_books and book_count < min_books:
                reasons.append(f"liquidity < {min_books} (have {book_count})")
            line = best_line_lookup.get(key)
            minutes_old = _minutes_since(line.last_update if line else None, run_time)
            if minutes_old is not None and minutes_old > stale_minutes:
                reasons.append(f"stale {minutes_old:.0f} min")
            if reasons:
                skipped_bets.append((rec, reasons))
                continue
            filtered.append(rec)

        recommendations = filtered

        total_stake = sum(rec["stake"] for rec in recommendations)
        cap = max_total_pct * bankroll
        if recommendations and total_stake > cap > 0:
            scale = cap / total_stake
            safety_notes.append(f"Stakes scaled to cap total at {max_total_pct:.1%} of bankroll.")
            for rec in recommendations:
                rec["stake"] = round(rec["stake"] * scale, 2)
            total_stake = sum(rec["stake"] for rec in recommendations)
        else:
            total_stake = round(total_stake, 2)

        expected_ev = sum(rec["ev_pct"] * rec["stake"] for rec in recommendations)
        expected_ev_pct = (expected_ev / bankroll) if bankroll else 0.0

        vegas_summary = {
            "timestamp": _format_vegas_timestamp(run_time),
            "bankroll": bankroll,
            "kelly": kelly_fraction,
            "total_stake": total_stake,
            "expected_ev_pct": expected_ev_pct,
            "skipped": len(skipped_bets),
        }

        if skipped_bets:
            safety_notes.append("Some bets skipped due to liquidity/stale odds (see run log).")

        log_path = _write_vegas_log(
            date_str=date_label,
            date_slug=date_slug,
            run_time=run_time,
            bankroll=bankroll,
            kelly=kelly_fraction,
            guardrails={
                "max_total_stake_pct": max_total_pct,
                "min_books": min_books,
                "stale_minutes": stale_minutes,
            },
            kept=recommendations,
            skipped=skipped_bets,
        )

        _print_vegas_summary(
            date_str=date_label,
            games=games,
            bankroll=bankroll,
            kelly=kelly_fraction,
            bets=recommendations,
            expected_ev_pct=expected_ev_pct,
            log_path=log_path,
        )

    if not recommendations:
        print("No actionable edges after guardrails.")
        if vegas_mode and skipped_bets:
            print("See vegas log for details: artifacts directory")
        if report_opts.get("report"):
            best_df = _best_lines_dataframe(best_rows)
            context = _build_report_context(
                date_str=date_label,
                date_slug=date_slug,
                games=games,
                config=config,
                odds_cache=cache_dir,
                features_metadata=features_meta,
                vegas_meta=vegas_summary,
                safety_notes=safety_notes,
                slate_dates=slate_dates,
            )
            paths = _generate_report(
                best_lines_df=best_df,
                recs_df=pd.DataFrame(),
                context=context,
                publish=report_opts.get("publish", False),
                open_report=report_opts.get("open", False),
            )
            if report_opts.get("publish"):
                copied = copy_latest_reports_to_site()
                for label, dest in copied.items():
                    print(f"Copied {label} CSV to {dest}")
        return 0

    _print_recommendations_table(recommendations)
    _write_recommendations_csv(recommendations, date_slug)

    if report_opts.get("report"):
        best_df = _apply_recommendations_to_best_df(
            _best_lines_dataframe(best_rows),
            recommendations,
        )

        entry_lookup = {entry['label']: entry for entry in game_entries.values()}
        projected_scores: list[str] = []
        for game_label in best_df['Game']:
            entry = entry_lookup.get(game_label)
            projected_scores.append(_format_projected_score(entry) if entry else '')

        last_update_val = None
        if 'Last Update' in best_df.columns:
            updates = [
                _parse_last_update(str(value))
                for value in best_df['Last Update']
                if str(value).strip()
            ]
            updates = [dt for dt in updates if dt]
            if updates:
                last_update_val = max(updates)
            insert_at = list(best_df.columns).index('Last Update')
            best_df.insert(insert_at, 'Projected Score', projected_scores)
        else:
            best_df['Projected Score'] = projected_scores

        recs_df = _recommendations_dataframe(recommendations)
        context = _build_report_context(
            date_str=date_label,
            date_slug=date_slug,
            games=games,
            config=config,
            odds_cache=cache_dir,
            features_metadata=features_meta,
            vegas_meta=vegas_summary,
            safety_notes=safety_notes,
            bankroll_value=bankroll,
            kelly_value=kelly_fraction,
            slate_dates=slate_dates,
        )
        if last_update_val:
            context['last_update'] = last_update_val
        paths = _generate_report(
            best_lines_df=best_df,
            recs_df=recs_df,
            context=context,
            publish=report_opts.get("publish", False),
            open_report=report_opts.get("open", False),
        )
        if report_opts.get("publish"):
            copied = copy_latest_reports_to_site()
            for label, dest in copied.items():
                print(f"Copied {label} CSV to {dest}")

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
    requested_dates: Sequence[str] | None,
) -> tuple[list[NormalizedLine], str, str, Path, list[str]]:
    parsed_dates: list[datetime] = []
    if requested_dates:
        for raw in requested_dates:
            if not raw:
                continue
            try:
                parsed_dates.append(datetime.strptime(raw, "%Y-%m-%d"))
            except ValueError:
                logging.warning("Skipping invalid odds date: %s", raw)
    if not parsed_dates:
        parsed_dates = [datetime.now(timezone.utc)]

    parsed_dates.sort()
    unique_dates: list[datetime] = []
    for dt in parsed_dates:
        if not unique_dates or unique_dates[-1] != dt:
            unique_dates.append(dt)
    date_tokens = [dt.strftime("%Y-%m-%d") for dt in unique_dates]
    first_token = date_tokens[0]

    if len(date_tokens) == 1:
        date_label = first_token
        date_slug = first_token.replace("-", "")
        cache_dir = Path("data_cache") / "odds" / date_slug
        with OddsAPIClient(config.odds_api) as odds_client:
            lines = odds_client.fetch_lines(
                games,
                markets=config.settings.markets,
                date=first_token,
                cache_dir=cache_dir,
            )
        return lines, date_label, date_slug, cache_dir, date_tokens

    last_token = date_tokens[-1]
    date_label = f"{first_token} to {last_token}"
    date_slug = f"{first_token.replace('-', '')}_{last_token.replace('-', '')}"
    cache_dir = Path("data_cache") / "odds" / date_slug

    aggregated: list[NormalizedLine] = []
    with OddsAPIClient(config.odds_api) as odds_client:
        for token in date_tokens:
            day_cache = cache_dir / token.replace("-", "")
            day_lines = odds_client.fetch_lines(
                games,
                markets=config.settings.markets,
                date=token,
                cache_dir=day_cache,
            )
            aggregated.extend(day_lines)

    deduped: dict[tuple[object, ...], NormalizedLine] = {}
    ordered: list[NormalizedLine] = []
    for line in aggregated:
        key = (
            line.game_label,
            line.market,
            line.side,
            line.book,
            line.price,
            line.point,
        )
        if key in deduped:
            continue
        deduped[key] = line
        ordered.append(line)

    return ordered, date_label, date_slug, cache_dir, date_tokens


def _build_game_context(
    games: Sequence[GameRequest],
    features: Mapping[str, TeamFeatures],
    league_avg: TeamFeatures | None,
) -> dict[str, dict[str, float]]:
    context: dict[str, dict[str, float]] = {}
    for game in games:
        away_feat = features.get(game.away_code or "") or league_avg
        home_feat = features.get(game.home_code or "") or league_avg
        if not away_feat or not home_feat:
            logging.warning("Missing features for matchup %s", game.label)
            continue

        home_strength = _team_strength(home_feat)
        away_strength = _team_strength(away_feat)
        expected_margin = predict_home_margin(home_strength, away_strength)
        raw_total = predict_total_points(
            home_feat.pace,
            away_feat.pace,
            home_feat.offensive_epa_per_play,
            away_feat.offensive_epa_per_play,
            home_feat.defensive_epa_per_play,
            away_feat.defensive_epa_per_play,
        )
        predicted_total = float(_clamp(raw_total, 20.0, 70.0))
        home_prob = 1.0 - _normal_cdf(0.0, mean=expected_margin, sigma=SPREAD_SIGMA)
        context[game.label] = {
            "home_prob": float(_clamp(home_prob, 0.01, 0.99)),
            "expected_margin": float(expected_margin),
            "predicted_total": predicted_total,
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
            f"{rec['game']:<28} {rec['market']:<8} {rec['side']:<6} {str(rec.get('line') or '--'):>8} "
            f"{rec['price']:+6d} {model_pct:>7.2f}% {implied_pct:>8.2f}% {edge_pct:>6.2f}% "
            f"{stake_value:>8} {str(rec.get('book', '')):<14} {rec['fair_price']:>6d}"
        )


def _write_lines_csv(
    selections: Sequence[tuple[GameRequest, str, str, NormalizedLine]],
    date_slug: str,
) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"lines_{date_slug}.csv"

    fieldnames = _best_line_columns()
    rows = _best_line_records(selections)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_recommendations_csv(
    recommendations: Sequence[Mapping[str, object]],
    date_slug: str,
) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"recommend_{date_slug}.csv"

    fieldnames = [
        "game_id",
        "game",
        "kickoff",
        "market",
        "side",
        "line",
        "price",
        "book",
        "model_prob",
        "implied_prob",
        "edge_prob",
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
                    "game_id": rec.get("game_id", ""),
                    "game": rec["game"],
                    "kickoff": rec.get("kickoff", ""),
                    "market": rec["market"],
                    "side": rec["side"],
                    "line": rec.get("line", ""),
                    "price": rec["price"],
                    "book": rec.get("book", ""),
                    "model_prob": round(rec["model_prob"], 4),
                    "implied_prob": round(rec["implied_prob"], 4),
                    "edge_prob": round(rec["model_prob"] - rec["implied_prob"], 4),
                    "ev_pct": round(rec["ev_pct"], 4),
                    "stake": round(rec["stake"], 2),
                    "fair_price": rec["fair_price"],
                    "last_update": rec.get("last_update", ""),
                }
            )







def _build_report_context(
    *,
    date_str: str,
    date_slug: str,
    games: Sequence[GameRequest],
    config: AppConfig,
    odds_cache: Path,
    features_metadata: Mapping[str, object] | None,
    vegas_meta: Mapping[str, object] | None,
    safety_notes: Sequence[str] | None,
    bankroll_value: float | None = None,
    kelly_value: float | None = None,
    slate_dates: Sequence[str] | None = None,
) -> dict[str, object]:
    run_time = datetime.now(timezone.utc)
    games_list = [game.label for game in games]
    features_cache = None
    season = None
    if features_metadata:
        features_cache = features_metadata.get('cache_path')
        season = features_metadata.get('season')
    if season is None:
        season = max(config.settings.train_seasons or [run_time.year])

    context: dict[str, object] = {
        'title': f"NFL Best Lines - {date_str}",
        'date': date_str,
        'date_slug': date_slug,
        'games': games_list,
        'run_time': run_time,
        'vegas_mode': bool(vegas_meta),
        'vegas_meta': vegas_meta or {},
        'safety_notes': list(safety_notes or []),
        'odds_cache': str(odds_cache),
        'features_cache': str(features_cache) if features_cache else 'N/A',
        'bankroll': float(bankroll_value if bankroll_value is not None else config.vegas.default_bankroll),
        'kelly_fraction': float(kelly_value if kelly_value is not None else config.settings.kelly_fraction),
        'max_stake_pct': float(config.settings.max_stake_pct),
        'min_ev': float(config.settings.min_ev),
        'lookback_weeks': int((features_metadata or {}).get('weeks') or config.settings.lookback_weeks),
        'season': int(season),
        'slate_dates': list(slate_dates or []),
        'footer': run_time.strftime('Generated %Y-%m-%d %H:%M UTC'),
    }
    return context


def _generate_report(
    *,
    best_lines_df: pd.DataFrame | None,
    recs_df: pd.DataFrame | None,
    context: Mapping[str, object],
    publish: bool,
    open_report: bool,
) -> dict[str, Path | None]:
    markdown = render_markdown(best_lines_df, recs_df, context)
    html = render_html(best_lines_df, recs_df, context)
    basename = f"mnf_{context.get('date_slug', datetime.now(timezone.utc).strftime('%Y%m%d'))}"
    paths = write_report(markdown, html, basename=basename, publish=publish)

    print(f"Markdown report: {paths['markdown']}")
    print(f"HTML report: {paths['html']}")
    if publish and paths.get('published'):
        print(f"Published HTML: {paths['published']}")

    if open_report and paths.get('html'):
        _open_html(paths['html'])

    return paths


def _open_html(path: Path) -> None:
    try:
        resolved = Path(path).resolve()
        webbrowser.open(resolved.as_uri())
    except Exception as exc:  # pragma: no cover - best effort
        logging.warning("Unable to open report %s: %s", path, exc)


def _apply_recommendations_to_best_df(df: pd.DataFrame, recommendations: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Overlay recommendation details onto the best-lines table."""

    rec_map: dict[tuple[str, str], Mapping[str, object]] = {}
    for rec in recommendations:
        game = str(rec.get('game', ''))
        market = str(rec.get('market', '')).lower()
        if not game or not market:
            continue
        rec_map[(game, market)] = rec

    picks_col: list[str] = []
    for idx, row in df.iterrows():
        game_label = row['Game']
        pick_segments: list[str] = []
        for market in MARKET_ORDER:
            rec = rec_map.get((game_label, market))
            if not rec:
                continue
            prefix = MARKET_COMPACT_LABELS.get(market, market.upper())
            side_token = str(rec.get('side', '')).lower()
            display_side = side_token.title()
            if market in {'ml', 'spread'}:
                teams = game_label.split(' @ ')
                if len(teams) == 2 and side_token in {'home', 'away'}:
                    display_side = 'Home ({})'.format(teams[1]) if side_token == 'home' else 'Away ({})'.format(teams[0])
            df.at[idx, f"{prefix} Side"] = display_side

            line_val = rec.get('line')
            df.at[idx, f"{prefix} Line"] = '--' if line_val in (None, '', '--') else str(line_val)

            price_val = rec.get('price')
            price_str = ''
            try:
                price_str = f"{int(price_val):+d}"
            except (TypeError, ValueError):
                if price_val not in (None, ''):
                    price_str = str(price_val)
            df.at[idx, f"{prefix} Price"] = price_str
            df.at[idx, f"{prefix} Book"] = str(rec.get('book', ''))

            last_update = rec.get('last_update')
            if last_update:
                df.at[idx, 'Last Update'] = _format_last_update(str(last_update))

            segment = prefix + ' ' + display_side
            if price_str:
                segment += f" {price_str}"
            book = rec.get('book')
            if book:
                segment += f" ({book})"
            pick_segments.append(segment)
        picks_col.append('; '.join(pick_segments))

    if picks_col:
        df['Picks'] = picks_col
    return df




def _best_lines_dataframe(
    selections: Sequence[tuple[GameRequest, str, str, NormalizedLine]]
) -> pd.DataFrame:
    columns = _best_line_columns()
    rows = _best_line_records(selections)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(rows, columns=columns)


def _best_line_columns() -> list[str]:
    columns = ["Game"]
    for market in MARKET_ORDER:
        prefix = MARKET_COMPACT_LABELS.get(market, market.upper())
        for field in BEST_LINE_FIELDS:
            columns.append(f"{prefix} {field}")
    columns.append("Last Update")
    return columns


def debug_clamp_counts(counts: Mapping[str, Mapping[str, int]]) -> None:
    total = sum(sum(reason_counts.values()) for reason_counts in counts.values())
    if not total:
        return
    print('Sanity filter drops:')
    for market in sorted(counts):
        for reason, count in sorted(counts[market].items()):
            print(f"  {market}: {reason} -> {count}")


def _prepare_model_inputs(
    best_rows: Sequence[tuple[GameRequest, str, str, NormalizedLine]],
    game_context: Mapping[str, Mapping[str, float]],
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], float], dict[str, dict[str, object]]]:
    """Build odds rows and model probabilities keyed by canonical game id."""

    game_entries: dict[str, dict[str, object]] = {}
    for game, market, side, line in best_rows:
        entry = game_entries.setdefault(
            line.game_id,
            {
                'label': line.game_label,
                'away_code': game.away_code,
                'home_code': game.home_code,
                'kickoff': line.kickoff,
                'home_ml_price': None,
                'away_ml_price': None,
                'total_points': {},
                'context': game_context.get(game.label),
            },
        )
        if entry.get('context') is None:
            entry['context'] = game_context.get(game.label)
        if market == 'ml':
            if side == 'home':
                entry['home_ml_price'] = line.price
            elif side == 'away':
                entry['away_ml_price'] = line.price
        if market == 'total' and line.point is not None:
            entry.setdefault('total_points', {})[side] = float(line.point)

    for game_id, entry in game_entries.items():
        context = entry.get('context')
        if not context:
            raise RuntimeError(f"Missing model context for {entry['label']}")
        p_home = float(_clamp(context.get('home_prob', 0.5), 0.0, 1.0))
        implied_home = None
        home_ml = entry.get('home_ml_price')
        away_ml = entry.get('away_ml_price')
        if home_ml is not None:
            implied_home = _implied_prob_from_price(int(home_ml))
        elif away_ml is not None:
            implied_home = 1.0 - _implied_prob_from_price(int(away_ml))
        if implied_home is not None:
            p_home = (p_home + float(_clamp(implied_home, 0.05, 0.95))) / 2.0
        p_home = float(_clamp(p_home, 0.05, 0.95))
        entry['p_home_win'] = p_home
        entry['p_away_win'] = 1.0 - p_home
        entry['expected_margin'] = float(context.get('expected_margin', 0.0))
        predicted_total = float(context.get('predicted_total', 0.0))
        totals = entry.get('total_points') or {}
        book_total = None
        if isinstance(totals, dict):
            over_pt = totals.get('over')
            under_pt = totals.get('under')
            if over_pt is not None and under_pt is not None:
                book_total = (over_pt + under_pt) / 2.0
            elif over_pt is not None:
                book_total = over_pt
            elif under_pt is not None:
                book_total = under_pt
        if book_total is not None:
            predicted_total = (predicted_total + float(book_total)) / 2.0
        entry['predicted_total'] = predicted_total
        home_pts = predicted_total * p_home
        away_pts = predicted_total - home_pts
        home_pts = float(_clamp(home_pts, 0.0, 70.0))
        away_pts = float(_clamp(away_pts, 0.0, 70.0))
        entry['projected_home_pts'] = home_pts
        entry['projected_away_pts'] = away_pts

        pick_side = 'home' if p_home >= 0.5 else 'away'
        home_ml_str = home_ml if home_ml is not None else '--'
        print(f"{entry['away_code']}@{entry['home_code']} | p_home={p_home:.3f} | home_ml={home_ml_str} | pick={pick_side}")

    odds_rows: list[dict[str, object]] = []
    model_probs: dict[tuple[str, str, str], float] = {}
    for game, market, side, line in best_rows:
        entry = game_entries[line.game_id]
        prob = _select_probability(entry, market, side, line.point)
        if prob is None:
            continue
        model_probs[(line.game_id, market, side)] = prob
        odds_rows.append(
            {
                'game': line.game_label,
                'game_id': line.game_id,
                'kickoff': entry['kickoff'],
                'market': market,
                'side': side,
                'price': line.price,
                'book': line.book,
                'line': _csv_line_value(line),
                'last_update': line.last_update,
                'p_home_win': entry['p_home_win'],
                'p_away_win': entry['p_away_win'],
            }
        )

    for entry in game_entries.values():
        p_home = entry.get('p_home_win')
        if p_home is None:
            raise RuntimeError(f"Missing p_home_win for {entry['label']}")
        total_prob = p_home + entry.get('p_away_win', 0.0)
        if not 0.98 <= total_prob <= 1.02:
            raise RuntimeError(
                f"Model probabilities misaligned for {entry['label']}: p_home={p_home:.3f}, p_away={entry.get('p_away_win', 0.0):.3f}"
            )

    return odds_rows, model_probs, game_entries


def _select_probability(entry: Mapping[str, object], market: str, side: str, line_point: float | None) -> float | None:
    """Return the model probability for a specific market/side."""

    if market == 'ml':
        if side == 'home':
            return float(_clamp(entry.get('p_home_win'), 0.01, 0.99))
        if side == 'away':
            return float(_clamp(entry.get('p_away_win'), 0.01, 0.99))
        return None

    if market == 'spread' and line_point is not None:
        prob = _spread_cover_probability(
            float(entry.get('expected_margin', 0.0)),
            float(line_point),
            side,
        )
        return float(_clamp(prob, 0.001, 0.999))

    if market == 'total' and line_point is not None:
        prob = _total_probability(
            float(entry.get('predicted_total', 0.0)),
            float(line_point),
            side,
        )
        return float(_clamp(prob, 0.001, 0.999))

    return None


def _canonical_game_id(away_code: str, home_code: str, commence_time: str) -> str:
    kickoff_raw = commence_time or ""
    if kickoff_raw:
        try:
            kickoff_dt = datetime.fromisoformat(kickoff_raw.replace("Z", "+00:00"))
            kickoff_date = kickoff_dt.strftime("%Y-%m-%d")
        except ValueError:
            kickoff_date = kickoff_raw[:10]
        if kickoff_date:
            return f"{away_code}@{home_code}-{kickoff_date}"
    return f"{away_code}@{home_code}"



def _implied_prob_from_price(price: int) -> float:
    if price >= 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def _decimal_from_price(price: int) -> float:
    if price >= 0:
        return 1 + price / 100.0
    return 1 + 100.0 / abs(price)


def _unit_return(price: int, won: bool) -> float:
    if not won:
        return -1.0
    if price >= 0:
        return price / 100.0
    return 100.0 / abs(price)


def _resolve_slate_date(mode: str, date_arg: str | None) -> str:
    """Return an ISO date for slate helpers."""

    base = datetime.now(timezone.utc).date()
    if date_arg:
        try:
            parsed = datetime.fromisoformat(date_arg.replace('Z', '+00:00'))
            base = parsed.date()
        except ValueError:
            try:
                base = datetime.strptime(date_arg, '%Y-%m-%d').date()
            except ValueError:
                try:
                    base = datetime.strptime(date_arg, '%Y%m%d').date()
                except ValueError:
                    pass
    mode = mode.lower()
    if mode == 'date':
        if not date_arg:
            raise ValueError('date mode requires --date YYYY-MM-DD')
        return base.strftime('%Y-%m-%d')
    if mode == 'today':
        return base.strftime('%Y-%m-%d')
    if mode in {'sunday', 'week'}:
        weekday = base.weekday()  # Monday=0, Sunday=6
        offset = (6 - weekday) % 7
        target = base + timedelta(days=offset)
        return target.strftime('%Y-%m-%d')
    return base.strftime('%Y-%m-%d')


def _parse_line_value(value: object) -> float | None:
    if value in (None, "", "--"):
        return None
    text = str(value).strip()
    if text.startswith('+'):
        text = text[1:]
    try:
        return float(text)
    except ValueError:
        return None


def _build_accuracy_table(
    recommendations: Sequence[Mapping[str, object]],
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    if not recommendations or results_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game",
                "kickoff",
                "market",
                "side",
                "line",
                "price",
                "model_prob",
                "implied_prob_from_odds",
                "edge",
                "ev",
                "pick_won",
                "stake",
                "home_score",
                "away_score",
                "final_total",
                "unit_return",
                "dollar_profit",
            ]
        )

    rec_df = pd.DataFrame(recommendations).copy()
    if "game_id" not in rec_df.columns:
        rec_df["game_id"] = ""
    rec_df["game_id"] = rec_df["game_id"].astype(str)
    for col in ("game", "kickoff", "market", "side", "line"):
        if col not in rec_df.columns:
            rec_df[col] = ""
    if "price" not in rec_df.columns:
        rec_df["price"] = 0
    rec_df["price"] = pd.to_numeric(rec_df["price"], errors='coerce').fillna(0).astype(int)
    if "model_prob" not in rec_df.columns:
        rec_df["model_prob"] = 0.5
    rec_df["model_prob"] = pd.to_numeric(rec_df["model_prob"], errors='coerce').fillna(0.5)
    if "stake" not in rec_df.columns:
        rec_df["stake"] = 1.0
    rec_df["stake"] = pd.to_numeric(rec_df["stake"], errors='coerce').fillna(1.0)
    results_df = results_df.copy()
    results_df["game_id"] = results_df["game_id"].astype(str)
    merged = rec_df.merge(results_df, on="game_id", how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game",
                "kickoff",
                "market",
                "side",
                "line",
                "price",
                "model_prob",
                "implied_prob_from_odds",
                "edge",
                "ev",
                "pick_won",
                "stake",
                "home_score",
                "away_score",
                "final_total",
                "unit_return",
                "dollar_profit",
            ]
        )

    def _resolve_pick(row: Mapping[str, object]) -> int | None:
        market = str(row.get("market"))
        side = str(row.get("side"))
        line_val = _parse_line_value(row.get("line"))
        home_score = float(row.get("home_score", 0.0))
        away_score = float(row.get("away_score", 0.0))

        if market == "ml":
            if side == "home":
                return 1 if home_score > away_score else 0
            if side == "away":
                return 1 if away_score > home_score else 0
            return None

        if market == "spread" and line_val is not None:
            if side == "home":
                adjusted = home_score + line_val - away_score
                if adjusted > 0:
                    return 1
                if adjusted < 0:
                    return 0
                return None
            if side == "away":
                adjusted = away_score + line_val - home_score
                if adjusted > 0:
                    return 1
                if adjusted < 0:
                    return 0
                return None
            return None

        if market == "total" and line_val is not None:
            total_score = home_score + away_score
            if side == "under":
                if total_score < line_val:
                    return 1
                if total_score > line_val:
                    return 0
                return None
            if side == "over":
                if total_score > line_val:
                    return 1
                if total_score < line_val:
                    return 0
                return None
        return None

    merged["pick_temp"] = merged.apply(_resolve_pick, axis=1)
    merged = merged.dropna(subset=["pick_temp"])
    if merged.empty:
        return merged.drop(columns=["pick_temp"]).copy()

    merged["pick_won"] = merged["pick_temp"].astype(int)
    merged = merged.drop(columns=["pick_temp"])
    merged["implied_prob_from_odds"] = merged["price"].astype(int).apply(_implied_prob_from_price)
    merged["edge"] = merged["model_prob"] - merged["implied_prob_from_odds"]
    merged["ev"] = merged.apply(lambda row: ev_from_american(float(row["model_prob"]), int(row["price"])), axis=1)
    merged["unit_return"] = merged.apply(lambda row: _unit_return(int(row["price"]), bool(row["pick_won"])), axis=1)
    merged["decimal_odds"] = merged["price"].astype(int).apply(_decimal_from_price)
    if "stake" not in merged.columns:
        merged["stake"] = 1.0
    merged["stake"] = pd.to_numeric(merged["stake"], errors='coerce').fillna(1.0)
    merged["dollar_profit"] = merged.apply(
        lambda row: row["stake"] * (row["decimal_odds"] - 1.0) if row["pick_won"] else -row["stake"],
        axis=1,
    )
    merged.drop(columns=["decimal_odds"], inplace=True)
    if "margin" not in merged.columns:
        merged["margin"] = merged["home_score"] - merged["away_score"]
    else:
        merged["margin"] = pd.to_numeric(merged["margin"], errors='coerce')

    if "ml_winner" not in merged.columns:
        merged["ml_winner"] = merged["margin"].apply(
            lambda value: "home" if pd.notna(value) and value > 0 else "away" if pd.notna(value) and value < 0 else "push" if pd.notna(value) else ""
        )
    else:
        merged["ml_winner"] = merged["ml_winner"].fillna("").astype(str)

    if "home_win" not in merged.columns:
        merged["home_win"] = merged["margin"].apply(
            lambda value: True if pd.notna(value) and value > 0 else False if pd.notna(value) and value < 0 else pd.NA
        )
    else:
        merged["home_win"] = merged["home_win"].where(~merged["home_win"].isna(), pd.NA)

    result = merged[
        [
            "game_id",
            "game",
            "kickoff",
            "market",
            "side",
            "line",
            "price",
            "model_prob",
            "implied_prob_from_odds",
            "edge",
            "ev",
            "pick_won",
            "stake",
            "home_score",
            "away_score",
            "final_total",
            "margin",
            "ml_winner",
            "home_win",
            "unit_return",
            "dollar_profit",
        ]
    ]
    result = result.rename(columns={"price": "odds", "model_prob": "p_model"})
    return result


def _write_accuracy_csv(df: pd.DataFrame, out_dir: Path, date_slug: str) -> Path:
    out_path = out_dir / f"accuracy_{date_slug}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _write_summary(
    reco_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    out_dir: Path,
    date_slug: str,
) -> Path:
    lines: list[str] = [f"# Summary for {date_slug}", ""]
    total_recs = len(reco_df)
    lines.append(f"- Recommendations generated: {total_recs}")
    scored = len(accuracy_df)
    lines.append(f"- Games scored: {scored}")

    if scored:
        overall_win = accuracy_df["pick_won"].mean()
        lines.append(f"- Overall win%: {overall_win:.1%}")

        for market_code, label in (("ml", "Moneyline"), ("spread", "Spread"), ("total", "Total")):
            subset = accuracy_df[accuracy_df["market"] == market_code]
            if not subset.empty:
                lines.append(f"- {label} win%: {subset['pick_won'].mean():.1%} ({len(subset)} picks)")

        avg_edge = accuracy_df["edge"].mean()
        avg_ev = accuracy_df["ev"].mean()
        lines.append(f"- Avg edge: {avg_edge:.2%}")
        lines.append(f"- Avg EV: {avg_ev:.2%}")

        flat_roi = accuracy_df["unit_return"].mean()
        lines.append(f"- Flat ROI: {flat_roi:.2%}")
        total_stake = accuracy_df["stake"].sum()
        total_profit = accuracy_df["dollar_profit"].sum()
        if total_stake > 0:
            lines.append(f"- Kelly ROI (stake-weighted): {total_profit / total_stake:.2%}")

        try:
            deciles = pd.qcut(accuracy_df["ev"], 10, labels=False, duplicates='drop')
            accuracy_copy = accuracy_df.copy()
            accuracy_copy["ev_decile"] = deciles
            lines.append("")
            lines.append("EV decile win%:")
            for decile in sorted(accuracy_copy["ev_decile"].dropna().unique()):
                subset = accuracy_copy[accuracy_copy["ev_decile"] == decile]
                lines.append(
                    f"  - Decile {int(decile) + 1}: {subset['pick_won'].mean():.1%} ({len(subset)} picks)"
                )
        except ValueError:
            pass
    else:
        lines.append("- No completed games with final scores yet.")

    out_path = out_dir / f"summary_{date_slug}.md"
    out_path.write_text("\n".join(lines).strip() + "\n", encoding='utf-8')
    return out_path



def _best_line_records(
    selections: Sequence[tuple[GameRequest, str, str, NormalizedLine]]
) -> list[dict[str, object]]:
    rows_by_game: dict[str, dict[str, object]] = {}
    last_updates: dict[str, datetime] = {}
    order: list[str] = []

    for game, market, side, line in selections:
        game_label = game.label
        if game_label not in rows_by_game:
            rows_by_game[game_label] = {"Game": game_label}
            order.append(game_label)

        prefix = MARKET_COMPACT_LABELS.get(market, market.upper())
        row = rows_by_game[game_label]
        row[f"{prefix} Side"] = _format_selection_label(line, side)
        row[f"{prefix} Line"] = _csv_line_value(line) or "--"
        row[f"{prefix} Price"] = f"{line.price:+d}"
        row[f"{prefix} Book"] = line.book

        parsed_update = _parse_last_update(line.last_update)
        if parsed_update:
            current = last_updates.get(game_label)
            if current is None or parsed_update > current:
                last_updates[game_label] = parsed_update

    records: list[dict[str, object]] = []
    for game_label in order:
        row = rows_by_game[game_label]
        for market in MARKET_ORDER:
            prefix = MARKET_COMPACT_LABELS.get(market, market.upper())
            row.setdefault(f"{prefix} Side", "")
            row.setdefault(f"{prefix} Line", "--")
            row.setdefault(f"{prefix} Price", "")
            row.setdefault(f"{prefix} Book", "")
        last_update = last_updates.get(game_label)
        row["Last Update"] = (
            _format_last_update(last_update.isoformat()) if last_update else ""
        )
        records.append(row)
    return records


def _parse_last_update(value: str | object) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _minutes_since(value: str | object, reference: datetime) -> float | None:
    timestamp = _parse_last_update(value)
    if not timestamp:
        return None
    ref = reference if reference.tzinfo else reference.replace(tzinfo=timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    delta = ref - timestamp.astimezone(ref.tzinfo)
    return delta.total_seconds() / 60.0


def _format_vegas_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def _format_kickoff_et(value: str | object) -> str:
    dt = _parse_last_update(value)
    if not dt:
        return str(value or '')
    try:
        tz = ZoneInfo('America/New_York') if ZoneInfo else timezone(timedelta(hours=-5))
    except Exception:
        tz = timezone(timedelta(hours=-5))
    dt_et = dt.astimezone(tz)
    return dt_et.strftime('%Y-%m-%d %I:%M %p ET')


def _write_vegas_log(
    *,
    date_str: str,
    date_slug: str,
    run_time: datetime,
    bankroll: float,
    kelly: float,
    guardrails: Mapping[str, object],
    kept: Sequence[Mapping[str, object]],
    skipped: Sequence[tuple[Mapping[str, object], Sequence[str]]],
) -> Path:
    VEGAS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = run_time.astimezone(timezone.utc) if run_time.tzinfo else run_time.replace(tzinfo=timezone.utc)
    filename = f"vegas_{date_slug}_{timestamp.strftime('%Y%m%dT%H%M%S')}.json"
    path = VEGAS_LOG_DIR / filename

    def _serialize(record: Mapping[str, object]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, datetime):
                value = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
                payload[key] = value.isoformat()
            else:
                payload[key] = value
        return payload

    log_payload = {
        'date': date_str,
        'date_slug': date_slug,
        'generated_at': timestamp.isoformat(),
        'bankroll': bankroll,
        'kelly_fraction': kelly,
        'guardrails': dict(guardrails),
        'kept': [_serialize(rec) for rec in kept],
        'skipped': [
            {'recommendation': _serialize(rec), 'reasons': list(reasons)}
            for rec, reasons in skipped
        ],
    }
    path.write_text(json.dumps(log_payload, indent=2), encoding='utf-8')
    return path


def _print_vegas_summary(
    *,
    date_str: str,
    games: Sequence[GameRequest],
    bankroll: float,
    kelly: float,
    bets: Sequence[Mapping[str, object]],
    expected_ev_pct: float,
    log_path: Path | None,
) -> None:
    print()
    print(f"Vegas Summary - {date_str}")
    print(f"Bankroll: ${bankroll:,.2f} | Kelly: {kelly:.2%}")
    if games:
        slate_overview = ', '.join(game.label for game in games)
        print(f"Slate: {slate_overview}")
    print(f"Kept {len(bets)} bet(s); expected EV {expected_ev_pct * 100:.2f}% of bankroll")
    for rec in bets:
        game = rec.get('game', '')
        market = str(rec.get('market', '')).upper()
        side = rec.get('side', '')
        line = rec.get('line', '')
        price = rec.get('price', '')
        book = rec.get('book', '')
        stake = rec.get('stake', '')
        print(f"  - {game} | {market} {side} {line} @ {price} ({book}) stake={stake}")
    if log_path:
        print(f"Log file: {log_path}")
    print()


def _recommendations_dataframe(
    recommendations: Sequence[Mapping[str, object]]
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for rec in recommendations:
        records.append(
            {
                "Game": rec["game"],
                "Game ID": rec.get("game_id", ""),
                "Kickoff": _format_kickoff_et(rec.get("kickoff", "")),
                "Market": rec["market"],
                "Side": rec["side"],
                "Model Prob": f"{rec['model_prob'] * 100:.2f}%",
                "Implied Prob": f"{rec['implied_prob'] * 100:.2f}%",
                "Edge %": f"{(rec['model_prob'] - rec['implied_prob']) * 100:.2f}%",
                "Book Odds": f"{rec['price']:+d}",
                "EV %": f"{rec['ev_pct'] * 100:.2f}%",
                "Stake $": f"${rec['stake']:.2f}",
                "Line": rec.get("line", ""),
                "Book": rec.get("book", ""),
                "Last Update": _format_last_update(rec.get("last_update", "")),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "Game",
            "Game ID",
            "Kickoff",
            "Market",
            "Side",
            "Model Prob",
            "Implied Prob",
            "Edge %",
            "Book Odds",
            "EV %",
            "Stake $",
            "Line",
            "Book",
            "Last Update",
        ],
    )
def _format_vegas_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')





def _format_projected_score(entry_or_home, away_pts: float | None = None) -> str:
    """Format projected scores as 'away-home'."""

    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if away_pts is None:
        entry = entry_or_home
        home_f = None
        away_f = None
        if isinstance(entry, dict):
            home_f = _to_float(entry.get('projected_home_pts'))
            away_f = _to_float(entry.get('projected_away_pts'))
            if home_f is None or away_f is None:
                total_f = _to_float(entry.get('predicted_total'))
                margin_f = _to_float(entry.get('expected_margin'))
                if total_f is not None and margin_f is not None:
                    home_f = (total_f + margin_f) / 2.0
                    away_f = total_f - home_f
            context = entry.get('context') if isinstance(entry.get('context'), dict) else {}
            if (home_f is None or away_f is None) and isinstance(context, dict):
                if home_f is None:
                    home_f = _to_float(context.get('projected_home_pts'))
                if away_f is None:
                    away_f = _to_float(context.get('projected_away_pts'))
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            home_f = _to_float(entry[0])
            away_f = _to_float(entry[1])
        else:
            home_f = _to_float(entry)
            away_f = _to_float(away_pts) if away_pts is not None else None
        if home_f is None or away_f is None:
            return ''
    else:
        home_f = _to_float(entry_or_home)
        away_f = _to_float(away_pts)
        if home_f is None or away_f is None:
            return ''

    return f"{away_f:.1f}-{home_f:.1f}"

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

    offense = features.offensive_epa_per_play * 50.0
    defense = -features.defensive_epa_per_play * 40.0
    success = (features.success_rate_off - features.success_rate_def) * 25.0
    turnover = -features.turnover_rate * 200.0
    pace_factor = (features.pace - 60.0) * 1.5
    return offense + defense + success + turnover + pace_factor


def _normal_cdf(value: float, *, mean: float, sigma: float) -> float:
    if sigma <= 0:
        sigma = 1.0
    z = (value - mean) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _spread_cover_probability(expected_margin: float, line_point: float, side: str) -> float:
    sigma = SPREAD_SIGMA if SPREAD_SIGMA > 0 else 1.0
    if side == "home":
        threshold = -line_point
        return 1.0 - _normal_cdf(threshold, mean=expected_margin, sigma=sigma)
    if side == "away":
        threshold = line_point
        return _normal_cdf(threshold, mean=expected_margin, sigma=sigma)
    return 0.5


def _total_probability(predicted_total: float, line_point: float, side: str) -> float:
    sigma = TOTAL_SIGMA if TOTAL_SIGMA > 0 else 1.0
    if side == "under":
        return _normal_cdf(line_point, mean=predicted_total, sigma=sigma)
    if side == "over":
        return 1.0 - _normal_cdf(line_point, mean=predicted_total, sigma=sigma)
    return 0.5


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

