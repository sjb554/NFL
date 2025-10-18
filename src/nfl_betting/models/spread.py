"""Spread placeholder model."""

from __future__ import annotations


def predict_home_margin(
    home_strength: float,
    away_strength: float,
    *,
    home_field_adv: float = 1.5,
    scaling: float = 7.0,
) -> float:
    """Return an expected home margin in points."""

    return (home_strength - away_strength) * scaling + home_field_adv


__all__ = ["predict_home_margin"]

