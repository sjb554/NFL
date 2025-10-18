"""Moneyline probability placeholders."""

from __future__ import annotations

import math


def predict_home_win_prob(
    home_strength: float,
    away_strength: float,
    *,
    home_field_adv: float = 1.5,
    scale: float = 1.0,
) -> float:
    """Return a logistic win probability for the home team."""

    delta = (home_strength - away_strength + home_field_adv) / max(scale, 1e-6)
    return 1.0 / (1.0 + math.exp(-delta))


__all__ = ["predict_home_win_prob"]

