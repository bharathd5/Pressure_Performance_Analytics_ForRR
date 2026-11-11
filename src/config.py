"""Central configuration. Every tunable lives here so the study can be re-run / re-parameterised."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, RES, REP = (ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "results", ROOT / "reports")

SEED = 42
FULL_INNINGS_BALLS = 120
N_FOLDS = 5                      # match-grouped cross-fitting for the win-probability model

# --- Pressure tiers (on normalised Leverage Index, mean over all balls == 1.0) ---
HIGH_LI = 1.6                    # "High pressure"  : swing in win-prob >= 1.6x the average delivery
EXTREME_LI = 2.6                 # "Extreme"        : subset of High
STAKES_MULT = {"League": 1.0, "Others": 1.0, "Super 4": 1.1, "Qualifier": 1.2,
               "Eliminator": 1.25, "Semi Final": 1.25, "Final": 1.3}   # knock-out stakes weighting

# --- Eligibility (minimum evidence) ---
MIN_HI_BALLS = {"bat": 150, "bowl": 150}      # high-pressure deliveries
MIN_HI_MATCHES = {"bat": 20, "bowl": 20}      # distinct matches with >=1 pressure delivery
MIN_BALLS_FOR_MATCH_HIT = 6                   # deliveries needed for a match to count in consistency

# --- Rating weights: absolute pressure performance / lift vs own normal / match-to-match consistency ---
WEIGHTS = {"P": 0.50, "D": 0.25, "C": 0.25}
