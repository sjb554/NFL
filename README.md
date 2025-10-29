# NFL Betting Helper

Phase 1 scaffolding for a small Python project that ties together The Odds API and nflfastR public data in support of NFL betting research. The focus is on a clean, Windows-friendly structure that will expand in later phases to include edge modeling and staking logic.

## Getting Started

1. Ensure Python 3.10+ is available on PATH (`python --version`).
2. Create and activate a virtual environment (Windows PowerShell example):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies either from the quick requirements file or in editable mode:
   ```powershell
   pip install -r requirements.txt
   # or
   pip install -e .
   ```
4. Copy `.env.example` to `.env` when you are ready to use a real Odds API key. Leaving it unset keeps the project in dry-run mode.
5. Adjust `settings.yaml` if you want to tweak the default modeling placeholders.

## Configuration Files

- `settings.yaml` holds scaffold defaults such as target markets and staking placeholders:
  ```yaml
  markets: ["ml", "spread", "total"]
  min_ev: 0.02
  kelly_fraction: 0.25
  max_stake_pct: 0.02
  lookback_weeks: 8
  train_seasons: [2022, 2023, 2024]
  max_total_stake_pct: 0.05
  min_books_per_selection: 3
  stale_minutes_ok: 60
  default_bankroll: 5000
  ```
- `.env.example` documents the `ODDS_API_KEY` variable. Copy it to `.env` and replace `YOUR_KEY` when you want live odds. `.env` is already ignored by git.

## Command-Line Interface

The package installs an `nfl-betting` CLI. By default it performs a dry run and prints `everything wired correctly` so you can validate the plumbing quickly:

```powershell
nfl-betting
```

To exercise the wiring end-to-end (requires network and an API key):

```powershell
nfl-betting --fetch --verbose
```

### Fetch Lines Report

Fetch the current best lines for the configured matchups and emit CSV snapshots under `reports/`:

```powershell
copy .env.example .env  # add your key
python -m nfl_betting.cli --lines --games "TB@DET,HOU@SEA"
```

### Run Recommendations

Run the full odds -> metrics -> placeholder model pipeline to generate recommendations and Kelly-sized stakes. Outputs land in `reports/recommend_<date>.csv`:

```powershell
copy .env.example .env  # add your key
python -m nfl_betting.cli --recommend --games "TB@DET,HOU@SEA"
```

### View a Report

Generate Markdown + HTML reports, open the HTML locally, and (optionally) copy it to `docs/index.html` for GitHub Pages:

```powershell
python -m nfl_betting.cli --recommend --games "TB@DET,HOU@SEA" --report --open
```

### Publish via GitHub Pages (optional)

1. Publish the latest report and copy an HTML snapshot to `docs/index.html`:
   ```powershell
   python -m nfl_betting.cli --recommend --games "TB@DET,HOU@SEA" --report --publish
   ```
2. In GitHub -> Settings -> Pages, select branch `main` and folder `/docs`, then save. Your reports will be available at `https://sjb554.github.io/NFL/`.

### Interactive Dashboard

An interactive dashboard (sortable tables, filtering, and dark mode) lives at `site/index.html`. It reads the latest CSVs from `site/data/` whenever you publish a report with `--report --publish`.

```powershell
python -m nfl_betting.cli --recommend --games "TB@DET,HOU@SEA" --report --publish
start site/index.html   # local preview
```

### One-click Vegas run

Set your API key first, then run the entire pipeline (fetch -> models -> reports -> dashboard updates) with a single command. The guardrails cap total stake at 5% of bankroll, require at least three books per side, and skip odds older than 60 minutes.

```powershell
copy .env.example .env  # then edit ODDS_API_KEY
python -m nfl_betting.cli --vegas --open
```

To publish the same run to GitHub Pages after previewing it locally:

```powershell
python -m nfl_betting.cli --vegas --publish
```

The dashboard and reports will be refreshed in both `docs/index.html` and `site/data/` so GitHub Pages hosts the latest snapshot. A run log is saved under `artifacts/` for easy auditing.

### Full slate automation

Use `--slate` to fetch the entire NFL card without typing matchups (requires a valid `ODDS_API_KEY`). Dates are resolved in Pacific Time. The new `week` option grabs the next *n* days (default 7) and deduplicates matchups across the span.

```powershell
python -m nfl_betting.cli --vegas --slate today --open
python -m nfl_betting.cli --vegas --slate sunday --open
python -m nfl_betting.cli --vegas --slate date --date 2025-10-19 --open
python -m nfl_betting.cli --vegas --slate week --week-days 10 --open
python -m nfl_betting.cli --slate today --export-games
```

The export option writes `reports/games_<date>.txt` for single-day pulls or `reports/games_<start>_<end>.txt` for weekly windows, one `AWAY@HOME` token per line (grouped by date for the week view).
Once GitHub Pages is enabled (branch `main`, folder `/docs`), the same dashboard will be viewable online.

## Repository Layout

- `pyproject.toml` - project metadata and dependency declarations
- `requirements.txt` - quick-install dependency list matching the project
- `settings.yaml` - default modeling parameters for later phases
- `src/nfl_betting/config.py` - centralized environment-driven configuration
- `src/nfl_betting/teams.py` - helpers for team aliases and matchup parsing
- `src/nfl_betting/odds_api.py` - odds client with caching and line normalization
- `src/nfl_betting/nflfastr.py` - nflfastR feature loading and caching
- `src/nfl_betting/models/` - placeholder moneyline/spread/total models
- `src/nfl_betting/pricing/` - edge calculations and fractional Kelly sizing
- `src/nfl_betting/report.py` - Markdown/HTML report rendering & dashboard helpers
- `src/nfl_betting/cli.py` - command-line entry point with reporting hooks
- `site/index.html` - interactive dashboard shell (fetches CSV snapshots)

## Generated Artifacts

- `data_cache/odds/<date>/` - cached Odds API JSON responses
- `data_cache/nflfastr/` - cached team feature files
- `reports/lines_<date>.csv` - saved best-line snapshots
- `reports/recommend_<date>.csv` - saved recommendation exports
- `reports/mnf_<date>.html` / `.md` - rendered reports
- `site/data/` - latest CSVs copied for the dashboard
- `docs/index.html` - optional published HTML report

Future phases can build on this skeleton to introduce analytics, modeling, and stake sizing logic.


\n### Troubleshooting\n\nIf nflfastR feature CSVs are not present under data_cache/nflfastR/, the CLI falls back to neutral, league-average team features. Reports and recommendations will still generate, and you can populate the cache later without changing the command syntax.\n

Try normalizing team codes manually if you run into alias issues:
```bash
python -c "from nfl_betting.teams import normalize_team; print([normalize_team(x) for x in 'LAR,LA,JAX,JAC,LV,LV,OAK,LAC,SD'.split(',')])"
```

