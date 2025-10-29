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
    league_avg_total: float = 44.0,
) -> float:
    """Return a crude expected total points for the matchup."""

    pace_avg = max((home_pace + away_pace) / 2.0, 0.0)
    offensive_edge = (home_off_epa + away_off_epa) - (home_def_epa + away_def_epa)
    pace_adjustment = (pace_avg - 60.0) * 0.5
    return league_avg_total + offensive_edge * 15.0 + pace_adjustment


__all__ = ["predict_total_points"]

