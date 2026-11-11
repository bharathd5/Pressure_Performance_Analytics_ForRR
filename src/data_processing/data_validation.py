import pandas as pd
from src.data_processing.clean_data import validate_cleaning


def run_checks(b: pd.DataFrame) -> dict:
    d = validate_cleaning(b)
    d["duplicate_balls"] = int(b.duplicated(["match_id", "innings", "innings_ball_num"]).sum())
    d["innings_over_120_legal_balls"] = int((b.groupby(["match_id", "innings"]).is_legal.sum() > 120).sum())
    d["null_striker"] = int(b.striker_id.isna().sum())
    return d
