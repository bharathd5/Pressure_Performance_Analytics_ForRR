import numpy as np
import pandas as pd


def player_match_table(view: pd.DataFrame, role: str, high_thr: float) -> pd.DataFrame:
    """One row per (player, match): sums over High-pressure vs normal deliveries (adjusted run-value + raw stats)."""
    pid = "striker_id" if role == "bat" else "bowler_id"
    d = view.copy()
    d["hi"] = d.pressure_index_raw >= high_thr
    d["v"], d["v2"] = d.adj_value, d.adj_value ** 2
    if role == "bat":
        d["runs_"], d["balls_"], d["out_"] = d.bat_runs, d.is_legal.astype(int), (d.dismissed_player_id == d.striker_id).astype(int)
        d["wkt_"] = 0
    else:
        d["runs_"], d["balls_"], d["out_"] = d.bowler_runs, d.is_legal.astype(int), 0
        d["wkt_"] = d.bowler_wicket.astype(int)
    d["dot_"] = ((d.total_runs == 0) & d.is_legal).astype(int)
    agg = {}
    for tag, m in (("hi", d.hi), ("lo", ~d.hi)):
        s = d[m].groupby([pid, "match_id"]).agg(**{f"{tag}_n": ("v", "size"), f"{tag}_sum": ("v", "sum"), f"{tag}_ss": ("v2", "sum"),
                                                   f"{tag}_runs": ("runs_", "sum"), f"{tag}_balls": ("balls_", "sum"),
                                                   f"{tag}_out": ("out_", "sum"), f"{tag}_wkt": ("wkt_", "sum"),
                                                   f"{tag}_dots": ("dot_", "sum"), f"{tag}_wpa": ("wpa", "sum")})
        agg[tag] = s
    pm = agg["hi"].join(agg["lo"], how="outer").fillna(0).reset_index().rename(columns={pid: "pid"})
    meta = d.groupby("match_id").agg(date=("match_date", "first"), is_league=("is_league", "first"), stage=("stage", "first"),
                                     comp=("comp_name", "first"))
    return pm.merge(meta, left_on="match_id", right_index=True)
