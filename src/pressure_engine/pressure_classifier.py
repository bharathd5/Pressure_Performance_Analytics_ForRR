import numpy as np
import pandas as pd
from src.config import HIGH_LI, EXTREME_LI


def classify(b: pd.DataFrame, high=HIGH_LI, extreme=EXTREME_LI) -> pd.DataFrame:
    b = b.copy()
    x = b.pressure_index_raw
    b["pressure_tier"] = np.select([x >= extreme, x >= high, x >= 0.7], ["Extreme", "High", "Medium"], default="Low")
    b["is_high"] = x >= high
    return b


def tier_profile(b: pd.DataFrame) -> pd.DataFrame:
    """Evidence table: how pressure tiers differ from ordinary overs on observable match-state variables."""
    g = b.groupby("pressure_tier")
    t = g.agg(deliveries=("leverage", "size"), mean_leverage=("leverage", "mean"), mean_overs=("over", "mean"),
              wkts_in_hand=("wkts_in_hand", "mean"), closeness=("closeness", "mean"),
              run_rate=("total_runs", lambda s: s.mean() * 6), wicket_pct=("is_wicket", lambda s: 100 * s.mean()),
              abs_wpa=("wpa", lambda s: s.abs().mean() * 100))
    t["share_pct"] = 100 * t.deliveries / t.deliveries.sum()
    return t.loc[[i for i in ["Low", "Medium", "High", "Extreme"] if i in t.index]]
