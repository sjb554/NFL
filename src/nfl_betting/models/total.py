"""Game total placeholder model."""

from __future__ import annotations


def predict_total_points(
    home_pace: float,
    away_pace: float,
    home_off_epa: float,
    away_off_epa: float,
    home_def_epa: float,
    away_def_epa: float,
    *,
    scaling: float = 10.0,
    league_avg_total: float = 44.0,
) -> float:
    """Return a crude expected total points for the matchup."""

    pace_avg = max((home_pace + away_pace) / 2.0, 0.0)
    offensive_edge = (
        (home_off_epa - away_def_epa) + (away_off_epa - home_def_epa)
    ) / 2.0
    return pace_avg * offensive_edge * scaling + league_avg_total


__all__ = ["predict_total_points"]

