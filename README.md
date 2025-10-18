# NFL Betting Helper

Phase 1 scaffolding for a small Python project that ties together The Odds API
and nflfastR public data in support of NFL betting research. The focus is on a
clean, Windows-friendly structure that will expand in later phases to include
edge modeling and staking logic.

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
4. Copy `.env.example` to `.env` when you are ready to use a real Odds API key.
   Leaving it unset keeps the project in dry-run mode.
5. Adjust `settings.yaml` if you want to tweak the default modeling placeholders.

## Configuration Files

- `settings.yaml` holds scaffold defaults such as target markets and staking
  placeholders:
  ```yaml
  markets: ["ml", "spread", "total"]
  min_ev: 0.02
  kelly_fraction: 0.25
  max_stake_pct: 0.02
  lookback_weeks: 6
  train_seasons: [2022, 2023, 2024]
  ```
- `.env.example` documents the `ODDS_API_KEY` variable. Copy it to `.env` and
  replace `YOUR_KEY` when you want live odds. `.env` is already ignored by git.

## Command-Line Interface

The package installs an `nfl-betting` CLI. By default it performs a dry run and
prints `everything wired correctly` so you can validate the plumbing quickly:

```powershell
nfl-betting
```

To exercise the wiring end-to-end (requires network and an API key):

```powershell
nfl-betting --fetch --verbose
```

### Fetch Lines Report

The odds workflow consumes the .env key plus `settings.yaml` markets, fetches
The Odds API lines for the requested games, and prints a best-line table while
caching JSON responses for six hours. It also writes a CSV snapshot under
`reports/` for later reference.

```powershell
copy .env.example .env  # add your key
python -m nfl_betting.cli --lines --games "TB@DET,HOU@SEA"
```

### Run Recommendations

Run the full odds -> metrics -> model placeholder pipeline to generate simple
recommendations. Results print to the console and save to
`reports/recommend_<date>.csv`.

```powershell
copy .env.example .env  # add your key
python -m nfl_betting.cli --recommend --games "TB@DET,HOU@SEA"
```

## Repository Layout

- `pyproject.toml` - project metadata and dependency declarations
- `requirements.txt` - quick-install dependency list matching the project
- `settings.yaml` - default modeling parameters for later phases
- `src/nfl_betting/config.py` - centralized environment-driven configuration
- `src/nfl_betting/teams.py` - helpers for team aliases and matchup parsing
- `src/nfl_betting/odds_api.py` - odds client with caching and line normalization
- `src/nfl_betting/nflfastr.py` - helpers for fetching nflfastR team statistics
- `src/nfl_betting/models/` - placeholder moneyline/spread/total models
- `src/nfl_betting/pricing/` - edge calculations and fractional Kelly sizing
- `src/nfl_betting/cli.py` - command-line entry point with lines, reports, and recommendations

## Generated Artifacts

- `data_cache/odds/<date>/` - cached Odds API JSON responses
- `data_cache/nflfastr/` - cached team feature files
- `reports/lines_<date>.csv` - saved best-line snapshots
- `reports/recommend_<date>.csv` - saved recommendation exports

Future phases can build on this skeleton to introduce analytics, modeling, and
stake sizing logic.
