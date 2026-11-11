"""Pressure engine.  Pressure := LEVERAGE of the delivery = expected |swing in win probability| from that ball,
normalised so the average delivery = 1.0.  It folds required run rate, wickets in hand, balls remaining and contest
closeness into one number because the win-probability model is a function of all of them.
Transparent drivers (RRR gap, wickets in hand, closeness) are also emitted so reports can explain *why* a ball is tense."""
import numpy as np
import pandas as pd
from src.config import STAKES_MULT
from src.models.win_probability import crossfit_win_prob


def compute_pressure(b: pd.DataFrame) -> pd.DataFrame:
    wp = crossfit_win_prob(b)
    out = pd.concat([b, wp], axis=1)
    out["leverage"] = out.exp_swing / out.exp_swing.mean()
    out["stakes_mult"] = out.stage.map(STAKES_MULT).fillna(1.0)
    out["pressure_index_raw"] = out.leverage * out.stakes_mult          # tier input
    out["pressure_pctile"] = out.pressure_index_raw.rank(pct=True) * 100  # 0-100 Pressure Index for display
    # interpretable drivers
    need = out.target - out.cum_runs_before
    out["rrr"] = np.where(out.innings == 2, need * 6 / out.balls_left.clip(lower=1), np.nan)
    out["crr"] = out.cum_runs_before * 6 / (out.max_balls - out.balls_left).clip(lower=1)
    out["wkts_in_hand"] = 10 - out.wkts_before
    out["closeness"] = 1 - 2 * (out.wp_before - 0.5).abs()               # 1 = coin-flip, 0 = decided
    return out
