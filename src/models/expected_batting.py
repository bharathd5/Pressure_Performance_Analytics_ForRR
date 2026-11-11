"""Expected batting value: what an AVERAGE batter would produce in the same (phase x batter hand x pace/spin) context,
and adjustment for the quality of the bowler faced.  adj_bat_value = bat_value - context_baseline + bowler_skill
(a good bowler lowers expectations, so beating him is credited)."""
import pandas as pd

KEYS = ["striker_hand", "pace_or_spin", "phase"]


def batter_view(b: pd.DataFrame) -> pd.DataFrame:
    d = b[~b.is_wide].copy()
    d["striker_hand"] = d["batting_type"]
    d["base_bat"] = d.groupby(KEYS)["bat_value"].transform("mean")
    d["bat_adj1"] = d["bat_value"] - d["base_bat"]
    return d
