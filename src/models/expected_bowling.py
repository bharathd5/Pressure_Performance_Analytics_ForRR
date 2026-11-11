"""Expected bowling value: league-average result for the same (phase x batter hand x pace/spin) context, and adjustment for
batter quality faced.  adj_bowl_value = bowl_value - context_baseline + batter_skill."""
import pandas as pd

KEYS = ["batting_type", "pace_or_spin", "phase"]


def bowler_view(b: pd.DataFrame) -> pd.DataFrame:
    d = b.copy()
    d["base_bowl"] = d.groupby(KEYS)["bowl_value"].transform("mean")
    d["bowl_adj1"] = d["bowl_value"] - d["base_bowl"]
    return d
