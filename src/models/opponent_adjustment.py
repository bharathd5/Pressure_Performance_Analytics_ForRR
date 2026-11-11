import pandas as pd
from src.models.expected_batting import batter_view
from src.models.expected_bowling import bowler_view
from src.models.shrinkage import entity_skill


def adjusted_views(b: pd.DataFrame):
    """Two-pass context + opponent adjustment. Returns (batter_rows, bowler_rows) with `adj_value`."""
    bv, wv = batter_view(b), bowler_view(b)
    s_bat = entity_skill(bv, "striker_id", "batting_team_id", "bat_adj1")      # batter skill (runs/ball above avg)
    s_bowl = entity_skill(wv, "bowler_id", "bowling_team_id", "bowl_adj1")     # bowler skill
    k_bowl = pd.MultiIndex.from_arrays([bv.bowler_id, bv.bowling_team_id])
    bv["opp_skill"] = s_bowl.reindex(k_bowl).fillna(0).values
    bv["adj_value"] = bv.bat_adj1 + bv.opp_skill
    k_bat = pd.MultiIndex.from_arrays([wv.striker_id, wv.batting_team_id])
    wv["opp_skill"] = s_bat.reindex(k_bat).fillna(0).values
    wv["adj_value"] = wv.bowl_adj1 + wv.opp_skill
    return bv, wv
