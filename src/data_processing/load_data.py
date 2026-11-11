"""Load raw CSVs and read/write processed tables (parquet if pyarrow present, else pickle fallback)."""
import pandas as pd
from src.config import RAW

FILES = {"ball_by_ball": "ball_by_ball.csv", "matches": "matches.csv", "competitions": "competitions.csv",
         "playing_eleven": "playing_eleven.csv", "teams": "teams.csv",
         "mapping": ["competition_team_player_mapping.csv", "compitition_team_player_mapping.csv"]}


def load_raw(raw_dir=RAW) -> dict:
    out = {}
    for k, f in FILES.items():
        for name in ([f] if isinstance(f, str) else f):
            p = raw_dir / name
            if p.exists():
                out[k] = pd.read_csv(p, low_memory=False); break
        else:
            raise FileNotFoundError(f"{f} not found in {raw_dir}")
    return out


def save_table(df: pd.DataFrame, path):
    try:
        df.to_parquet(path, index=False)
    except Exception:                      # pyarrow not installed -> pickle with same stem
        df.to_pickle(path.with_suffix(".pkl"))


def load_table(path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_pickle(path.with_suffix(".pkl"))
