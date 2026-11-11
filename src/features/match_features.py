import pandas as pd
from src.models.run_expectancy import RunExpectancy, add_run_values
from src.pressure_engine.pressure_calculator import compute_pressure
from src.pressure_engine.pressure_classifier import classify


def build_match_features(clean: pd.DataFrame) -> pd.DataFrame:
    """cleaned deliveries -> + win-prob, leverage/pressure, tiers, run-expectancy values."""
    re = RunExpectancy().fit(clean)
    b = add_run_values(clean, re)
    b = compute_pressure(b)
    return classify(b)
