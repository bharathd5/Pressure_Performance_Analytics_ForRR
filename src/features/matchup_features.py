"""Matchup / form / in-match tendency features (descriptive layer used in player profiles and the dashboard)."""
import numpy as np
import pandas as pd
from src.models.shrinkage import eb_shrink


def hand_vs_type_table(bv: pd.DataFrame, pid: int, high_only=True) -> pd.DataFrame:
    d = bv[bv.striker_id == pid]
    if high_only:
        d = d[d.is_high]
    return d.groupby(["striker_hand", "pace_or_spin"]).agg(balls=("adj_value", "size"), runs=("bat_runs", "sum"),
                                                          adj_value_per_ball=("adj_value", "mean")).round(3)


def head_to_head(bv: pd.DataFrame, min_balls=12) -> pd.DataFrame:
    """Batter-vs-bowler record, shrunk toward each batter's overall level (small H2H samples are noisy)."""
    g = bv.groupby(["striker_id", "bowler_id"]).agg(balls=("adj_value", "size"), runs=("bat_runs", "sum"),
                                                     outs=("is_wicket", "sum"), val=("adj_value", "mean")).reset_index()
    g = g[g.balls >= min_balls]
    sh, B, _ = eb_shrink(g.val.values, bv.adj_value.var() / g.balls.values, mu=0.0)
    g["h2h_value_shrunk"] = sh
    return g.sort_values("h2h_value_shrunk")


def player_form(view: pd.DataFrame, pid_col: str, halflife_matches=8) -> pd.DataFrame:
    """Exponentially-weighted per-match run-value (current form) up to each player's latest match."""
    pm = view.groupby([pid_col, "match_id"]).agg(date=("match_date", "first"), v=("adj_value", "mean")).reset_index().sort_values("date")
    pm["form"] = pm.groupby(pid_col)["v"].transform(lambda s: s.ewm(halflife=halflife_matches).mean())
    last = pm.groupby(pid_col).tail(1).rename(columns={pid_col: "pid"})
    return last[["pid", "form", "date"]]


def in_match_tendencies(b: pd.DataFrame, match_id: int, before_over: int) -> dict:
    """What the bowler/batter did earlier in THIS match (strokes that scored, delivery styles that took wickets / leaked runs).
    The feed has no line/length coordinates; `delivery_style` and `stroke` are the closest available proxies."""
    d = b[(b.match_id == match_id) & (b.over < before_over)]
    bowl = d.groupby(["bowler_name", "delivery_style"]).agg(balls=("total_runs", "size"), runs=("bowler_runs", "sum"),
                                                          wkts=("bowler_wicket", "sum"))
    bat = d.groupby(["striker_name", "stroke"]).agg(balls=("bat_runs", "size"), runs=("bat_runs", "sum"),
                                                  outs=("is_wicket", "sum"))
    return {"bowler_styles": bowl, "batter_strokes": bat}
