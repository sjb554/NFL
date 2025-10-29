"""Reporting helpers for Markdown and HTML outputs."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
import shutil
from typing import Mapping

import pandas as pd


DROP_COLUMN_KEYS = {
    'last update',
}

def _normalize_header(name: str) -> str:
    return ' '.join(name.strip().lower().replace('_', ' ').split())

def _strip_columns(df: pd.DataFrame | None, *, keys: set[str]) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    normalized_map = {col: _normalize_header(col) for col in df.columns}
    to_drop = [col for col, norm in normalized_map.items() if norm in keys]
    if to_drop:
        return df.drop(columns=to_drop)
    return df



def render_markdown(
    best_lines_df: pd.DataFrame | None,
    recs_df: pd.DataFrame | None,
    context: Mapping[str, object],
) -> str:
    """Render the report in GitHub-flavoured Markdown."""

    lines: list[str] = []
    title = context.get("title", "MNF Report")
    run_time = _format_dt(context.get("run_time"))
    last_update_val = context.get("last_update")
    last_update = _format_dt(last_update_val) if last_update_val else None

    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Run time:** {run_time}")
    if last_update:
        lines.append(f"**Last update:** {last_update}")
    lines.append("")

    clean_best = _strip_columns(best_lines_df.copy() if best_lines_df is not None else None, keys=DROP_COLUMN_KEYS)
    clean_recs = _strip_columns(recs_df.copy() if recs_df is not None else None, keys=DROP_COLUMN_KEYS)

    if clean_best is not None and not clean_best.empty:
        lines.append("## Best Lines")
        lines.extend(_markdown_table(clean_best))
        lines.append("")

    if clean_recs is not None and not clean_recs.empty:
        lines.append("## Recommendations")
        lines.extend(_markdown_table(clean_recs))
        lines.append("")

    lines.append("## Method Notes")
    lines.extend(_method_notes(context))
    lines.append("")

    safety = context.get("safety_notes") or []
    if safety:
        lines.append("## Safety Rails")
        for note in safety:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Cache & Data Sources")
    lines.append(f"- Odds cache: `{context.get('odds_cache', 'N/A')}`")
    lines.append(f"- Team features cache: `{context.get('features_cache', 'N/A')}`")
    lines.append("")

    footer = context.get("footer")
    if footer:
        lines.append(f"_Generated: {footer}_")

    return "\n".join(lines).strip() + "\n"

def render_html(
    best_lines_df: pd.DataFrame | None,
    recs_df: pd.DataFrame | None,
    context: Mapping[str, object],
) -> str:
    """Render the report as a standalone HTML document (inline CSS)."""

    title = html.escape(context.get("title", "MNF Report"))
    run_time = html.escape(_format_dt(context.get("run_time")))
    badge_html = ""
    if context.get("vegas_mode"):
        meta = context.get("vegas_meta") or {}
        timestamp = html.escape(str(meta.get("timestamp", "")))
        if timestamp:
            badge_html = f"<span class='badge badge-vegas'>Vegas mode  - {timestamp}</span>"

    clean_best = _strip_columns(best_lines_df.copy() if best_lines_df is not None else None, keys=DROP_COLUMN_KEYS)
    clean_recs = _strip_columns(recs_df.copy() if recs_df is not None else None, keys=DROP_COLUMN_KEYS)

    sections: list[str] = []

    header = [f"<section><h1>{title}{(' ' + badge_html) if badge_html else ''}</h1><p>"]
    header.append(f"<strong>Run time:</strong> {run_time}")
    header.append("</p>")
    sections.append("".join(header))

    if clean_best is not None and not clean_best.empty:
        sections.append("<h2>Lines Over the Next 7 Days</h2>")
        sections.append(_html_best_lines_table(clean_best))

    if clean_recs is not None and not clean_recs.empty:
        sections.append("<h2>Recommendations</h2>")
        sections.append(_html_table(clean_recs))

    method_notes = "".join(f"<p>{html.escape(note)}</p>" for note in _method_notes(context))
    sections.append(f"<h2>Method Notes</h2>{method_notes}")

    safety = context.get("safety_notes") or []
    if safety:
        safety_html = "".join(f"<li>{html.escape(str(note))}</li>" for note in safety)
        sections.append(f"<h2>Safety Rails</h2><ul>{safety_html}</ul>")

    odds_cache = html.escape(str(context.get("odds_cache", "N/A")))
    features_cache = html.escape(str(context.get("features_cache", "N/A")))
    sections.append(
        "<h2>Cache &amp; Data Sources</h2>"
        f"<ul><li>Odds cache: <code>{odds_cache}</code></li>"
        f"<li>Team features cache: <code>{features_cache}</code></li></ul>"
    )

    footer = html.escape(str(context.get("footer", "")))
    if footer:
        sections.append(f"<footer><small>{footer}</small></footer>")

    body = "".join(sections)

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 2rem auto; max-width: 960px; color: #1f2933; background:#fff; }
    h1, h2 { color: #0b3d91; }
    .badge { display:inline-block; margin-left:0.75rem; padding:0.2rem 0.6rem; border-radius:999px; background:#0b3d91; color:#fff; font-size:0.75rem; }
    .badge-vegas { background:#0b3d91; color:#fff; }
    body.dark .badge-vegas { background:#3d8bfd; color:#04070f; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }
    th, td { border: 1px solid #d9e2ec; padding: 0.5rem; text-align: left; }
    th { background: #f0f4f8; }
    th.group-header { text-align: center; font-size: 0.9rem; }
    th.sub-header { font-size: 0.8rem; }
    code { background: #f0f4f8; padding: 0.1rem 0.25rem; border-radius: 4px; }
    footer { margin-top: 2rem; text-align: center; color: #52606d; }
    """

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{css}</style></head><body>{body}</body></html>"
    )


def write_report(
    markdown_str: str,
    html_str: str,
    *,
    out_dir: str | Path = "reports",
    basename: str,
    publish: bool = False,
) -> dict[str, Path | None]:
    """Persist the Markdown/HTML report and optionally copy to docs/."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    md_path = out_path / f"{basename}.md"
    html_path = out_path / f"{basename}.html"

    md_path.write_text(markdown_str, encoding="utf-8")
    html_path.write_text(html_str, encoding="utf-8")

    published_path: Path | None = None
    if publish:
        docs_dir = Path("docs")
        docs_dir.mkdir(parents=True, exist_ok=True)
        published_path = docs_dir / "index.html"
        published_path.write_text(html_str, encoding="utf-8")

    return {"markdown": md_path, "html": html_path, "published": published_path}


def copy_latest_reports_to_site(
    report_dir: str | Path = 'reports',
    site_dir: str | Path = 'site',
) -> dict[str, Path]:
    """Copy the latest CSV snapshots into site/data for the dashboard."""

    report_path = Path(report_dir)
    site_path = Path(site_dir)
    data_dir = site_path / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    def _latest(pattern: str) -> Path | None:
        candidates = sorted(
            report_path.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    copied: dict[str, Path] = {}

    lines_src = _latest('lines_*.csv')
    if lines_src:
        dest = data_dir / 'lines.csv'
        shutil.copy2(lines_src, dest)
        copied['lines'] = dest

    rec_src = _latest('recommend_*.csv')
    if rec_src:
        dest = data_dir / 'recommend.csv'
        shutil.copy2(rec_src, dest)
        copied['recommend'] = dest

    timestamp = datetime.utcnow().strftime('Dashboard data updated %Y-%m-%d %H:%M UTC')
    (data_dir / 'generated.txt').write_text(timestamp, encoding='utf-8')

    return copied


def _markdown_table(df: pd.DataFrame) -> list[str]:
    headers = [str(col) for col in df.columns]
    divider = ["---" for _ in headers]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(divider) + " |"]
    for _, row in df.iterrows():
        cells = [str(row[col]) for col in df.columns]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def _html_best_lines_table(df: pd.DataFrame) -> str:
    expected_markets = ["ML", "Spread", "Total"]
    fields = ["Side", "Line", "Price", "Book"]
    columns = list(df.columns)
    if not all(f"{market} {field}" in columns for market in expected_markets for field in fields):
        return _html_table(df)

    thead_top: list[str] = []
    thead_bottom: list[str] = []
    thead_top.append("<tr><th rowspan='2'>Game</th>")
    for market in expected_markets:
        thead_top.append(f"<th class='group-header' colspan='{len(fields)}'>{market}</th>")
    thead_top.append("<th rowspan='2'>Predicted Score</th></tr>")
    thead_bottom.append("<tr>")
    for _ in expected_markets:
        for field in fields:
            thead_bottom.append(f"<th class='sub-header'>{field}</th>")
    thead_bottom.append("</tr>")
    thead_html = "".join(thead_top + thead_bottom)

    body_rows: list[str] = []
    for _, row in df.iterrows():
        cells = [f"<td>{html.escape(str(row['Game']))}</td>"]
        for market in expected_markets:
            for field in fields:
                value = row.get(f"{market} {field}", "")
                cells.append(f"<td>{html.escape(str(value))}</td>")
        predicted = row.get('Projected Score', '')
        cells.append(f"<td>{html.escape(str(predicted))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    tbody_html = "".join(body_rows)
    return f"<table><thead>{thead_html}</thead><tbody>{tbody_html}</tbody></table>"


def _html_table(df: pd.DataFrame) -> str:
    thead = "".join(f"<th>{html.escape(str(col))}</th>" for col in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{html.escape(str(row[col]))}</td>" for col in df.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    tbody = "".join(body_rows)
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"


def _method_notes(context: Mapping[str, object]) -> list[str]:
    bankroll = float(context.get("bankroll", 0.0))
    kelly_fraction = float(context.get("kelly_fraction", 0.0))
    max_stake_pct = float(context.get("max_stake_pct", 0.0))
    min_ev = float(context.get("min_ev", 0.0))
    weeks = context.get("lookback_weeks")
    season = context.get("season")

    notes = [
        f"Bankroll ${bankroll:.2f} with fractional Kelly ({kelly_fraction:.2%}) capped at {max_stake_pct:.2%}.",
        f"Minimum edge threshold: {min_ev:.2%}.",
        f"Team metrics: last {weeks} weeks of season {season} (nflverse stats_team_week release).",
        "Odds fetched from The Odds API (US region, h2h/spread/total).",
    ]
    return notes


def _format_dt(value: object) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    return str(value or "N/A")




__all__ = ["render_markdown", "render_html", "write_report", "copy_latest_reports_to_site"]
