"""Clean ball-by-ball data and rebuild the information the feed does not carry.

Findings from data exploration (documented in the report):
  * `runs` is runs off the bat ONLY. Wide / bye / leg-bye / no-ball penalty runs are NOT recorded
    (`extras1_runs` is a category code: 2=wide, 3=legbye, 4=bye, 5=other, 6=no extra, 1=overstep).
  * 'other' extras occupy an extra slot in the over exactly like no-balls, so they are treated as no-balls.
  * Rows 'wide' and 'Overstep'(no-ball) are not legal deliveries; `over_ball_num` counts them.
  * Innings 3/4 are super overs; 16 matches are abandoned (no deliveries).
We impute extras (wide=1, no-ball=1, bye/leg-bye=1 or 4 if boundary flag, penalty=5) and validate the
imputation by re-deriving each match winner from innings totals.
"""
import numpy as np
import pandas as pd
from src.config import FULL_INNINGS_BALLS

BOWLER_WICKETS = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}


def clean_deliveries(raw: dict) -> pd.DataFrame:
    b = raw["ball_by_ball"]
    b = b[b["innings"].isin([1, 2])].copy()                       # drop super overs
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    m = raw["matches"].merge(raw["competitions"], on="comp_id", how="left")
    keep = ["match_id", "comp_id", "comp_name", "match_format", "season", "match_date", "stage", "knock_out",
            "win_team_id", "first_inn_revised_overs", "second_inn_revised_overs", "revised_target", "venue_id"]
    b = b.merge(m[keep], on="match_id", how="left")
    b["match_date"] = pd.to_datetime(b["match_date"])
    b["is_league"] = (b["match_format"] == "T20").astype(int)    # franchise league (IPL etc.) vs international

    et = b["extras_type"].fillna("None")
    b["is_wide"], b["is_noball"] = et.eq("wide"), et.isin(["Overstep", "other"])   # 'other' behaves as a no-ball (7th ball in over)
    b["is_legal"] = ~(b["is_wide"] | b["is_noball"])
    bye_like = et.isin(["byes", "legbyes"])
    extra = np.select([et.eq("wide"), b["is_noball"], bye_like & b["is_four"], bye_like, et.eq("Penalty")],
                      [1, 1, 4, 1, 5], default=0) + b["extras_2_runs"].where(et != "None", 0)
    b["extra_runs"] = extra
    b["bat_runs"] = b["runs"]
    b["total_runs"] = b["runs"] + b["extra_runs"]
    b["bowler_runs"] = b["runs"] + b["is_wide"].astype(int) + b["is_noball"].astype(int)   # byes/leg-byes not charged

    b["dismissed_player_id"] = b["dismissed_player_id"].where(~b["is_wicket"] | b["dismissed_player_id"].notna(),
                                                                b["striker_id"])
    b["bowler_wicket"] = b["is_wicket"] & b["dismissal_type"].isin(BOWLER_WICKETS)

    g = b.groupby(["match_id", "innings"], sort=False)
    b["cum_runs_before"] = g["total_runs"].cumsum() - b["total_runs"]
    b["wkts_before"] = g["is_wicket"].cumsum() - b["is_wicket"].astype(int)
    b["legal_before"] = g["is_legal"].cumsum() - b["is_legal"].astype(int)
    b["innings_total"] = g["total_runs"].transform("sum")

    b["max_balls"] = np.where(b["innings"] == 1, b["first_inn_revised_overs"], b["second_inn_revised_overs"])
    b["max_balls"] = pd.Series(b["max_balls"]).fillna(FULL_INNINGS_BALLS).clip(upper=FULL_INNINGS_BALLS).values
    b["balls_left"] = (b["max_balls"] - b["legal_before"]).clip(lower=0)

    first_tot = b[b.innings == 1].groupby("match_id")["total_runs"].sum()
    b["target"] = np.where(b["innings"] == 2, b["match_id"].map(first_tot) + 1, np.nan)
    b["target"] = np.where((b["innings"] == 2) & b["revised_target"].notna(), b["revised_target"], b["target"])
    b["over"] = b["overid"]
    b["phase"] = pd.cut(b["over"], [0, 6, 15, 99], labels=["Powerplay", "Middle", "Death"]).astype(str)
    b["batting_team_won"] = (b["win_team_id"] == b["batting_team_id"]).astype(float)
    b.loc[b["win_team_id"].isna(), "batting_team_won"] = np.nan
    return b


def validate_cleaning(b: pd.DataFrame) -> dict:
    """Re-derive each match winner from the rebuilt innings totals and compare with matches.csv."""
    inn = b.groupby(["match_id", "innings"]).agg(runs=("total_runs", "sum"), bat=("batting_team_id", "first"),
                                                 win=("win_team_id", "first"), tgt=("target", "first")).reset_index()
    p = inn.pivot(index="match_id", columns="innings")
    ok = p.dropna(subset=[("runs", 1), ("runs", 2), ("win", 1)])
    pred = np.where(ok[("runs", 2)] >= ok[("tgt", 2)], ok[("bat", 2)], ok[("bat", 1)])
    agree = (pred == ok[("win", 1)])
    return {"matches_with_balls": int(b.match_id.nunique()), "deliveries": len(b),
            "winner_agreement_pct": round(100 * agree.mean(), 2), "n_checked": int(len(ok)),
            "mean_first_innings_total": round(inn[inn.innings == 1].runs.mean(), 1),
            "mean_second_innings_total": round(inn[inn.innings == 2].runs.mean(), 1)}
