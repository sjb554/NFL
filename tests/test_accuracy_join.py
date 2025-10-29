import math
from pathlib import Path

import pandas as pd

from nfl_betting.cli import _build_accuracy_table
from nfl_betting.pricing.edge import ev_from_american


def test_accuracy_join_produces_expected_columns(tmp_path):
    recs = [
        {
            "game_id": "BUF@CAR-20251026",
            "game": "Buffalo Bills @ Carolina Panthers",
            "kickoff": "2025-10-26T17:00:00Z",
            "market": "ml",
            "side": "home",
            "line": "",
            "price": 120,
            "book": "Example",
            "model_prob": 0.55,
            "implied_prob": 100 / (120 + 100),
            "ev_pct": ev_from_american(0.55, 120),
            "stake": 10.0,
            "fair_price": -122,
        }
    ]
    results = pd.DataFrame(
        [
            {
                "game_id": "BUF@CAR-20251026",
                "home_code": "CAR",
                "away_code": "BUF",
                "home_score": 30.0,
                "away_score": 24.0,
                "final_total": 54.0,
            }
        ]
    )

    accuracy = _build_accuracy_table(recs, results)

    assert not accuracy.empty
    assert {"margin", "ml_winner", "home_win"}.issubset(accuracy.columns)
    assert accuracy["margin"].iloc[0] == accuracy["home_score"].iloc[0] - accuracy["away_score"].iloc[0]
    assert accuracy["ml_winner"].iloc[0] == "home"
    assert bool(accuracy["home_win"].iloc[0]) is True
    assert (accuracy["home_score"].notna() & accuracy["away_score"].notna()).all()
    assert accuracy["p_model"].between(0, 1).all()

    recomputed_ev = accuracy.apply(lambda row: ev_from_american(float(row["p_model"]), int(row["odds"])), axis=1)
    assert math.isclose(recomputed_ev.iloc[0], accuracy["ev"].iloc[0], rel_tol=1e-6)
