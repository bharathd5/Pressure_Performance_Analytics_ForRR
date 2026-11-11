"""Expected-runs-remaining model E[runs to end of innings | balls_left, wickets_lost, competition type].
Fitted on first innings only (second-innings totals are truncated by the target).
Run value of a delivery = runs + dERR; league-average value is 0 in every state => measures skill above an average player
in the SAME situation, with the same currency (runs) for batters and bowlers."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from src.config import SEED


class RunExpectancy:
    def fit(self, b: pd.DataFrame):
        d = b[b.innings == 1].copy()
        rem = d.groupby("match_id")["total_runs"].transform(lambda s: s[::-1].cumsum()[::-1])
        self.m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=16, min_samples_leaf=200,
                                               monotonic_cst=[1, -1, 0], random_state=SEED)
        self.m.fit(d[["balls_left", "wkts_before", "is_league"]].values, rem.values)
        return self

    def err(self, bl, wk, league):
        bl, wk, league = map(np.asarray, (bl, wk, league))
        out = np.zeros(len(bl))
        live = (bl > 0) & (wk < 10)
        if live.any():
            out[live] = self.m.predict(np.c_[bl[live], wk[live], league[live]])
        return out


def add_run_values(b: pd.DataFrame, re: RunExpectancy) -> pd.DataFrame:
    b = b.copy()
    bl0, wk0, lg = b.balls_left.values, b.wkts_before.values, b.is_league.values
    bl1 = bl0 - b.is_legal.values.astype(int)
    e0, e_nw, e_w = re.err(bl0, wk0, lg), re.err(bl1, wk0, lg), re.err(bl1, wk0 + 1, lg)
    b["err_before"] = e0
    b["d_state"] = e_nw - e0                                         # ball tax (no wicket)
    b["wkt_cost"] = np.where(b.is_wicket, e_w - e_nw, 0.0)           # <= 0 : run-value lost by the wicket
    b["bat_value"] = b.bat_runs + b.d_state + np.where(b.dismissed_player_id == b.striker_id, b.wkt_cost, 0.0)
    b["bowl_value"] = -(b.bowler_runs + b.d_state) - np.where(b.bowler_wicket, b.wkt_cost, 0.0)
    return b
