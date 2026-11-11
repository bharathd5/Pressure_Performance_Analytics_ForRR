"""Cross-fitted ball-by-ball win-probability model (gradient boosting).
Predictions for every match come from a model that never saw that match (GroupKFold on match_id), so
leverage and win-probability-added are free of in-sample leakage."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from src.config import N_FOLDS, SEED

FEATS = ["innings", "balls_left", "wickets", "runs", "par_diff", "runs_needed", "rrr", "is_league"]


def make_X(inn, balls_left, wkts, runs, target, league, max_balls):
    inn, bl, wk, rn_ = map(np.asarray, (inn, balls_left, wkts, runs))
    max_balls = np.asarray(max_balls, float); target = np.asarray(target, float)
    need = np.where(inn == 2, target - rn_, np.nan)
    rrr = np.where(inn == 2, need * 6 / np.maximum(bl, 1), np.nan)
    return pd.DataFrame({"innings": inn, "balls_left": bl, "wickets": wk, "runs": rn_, "par_diff": rn_ - 1.35 * (max_balls - bl),
                         "runs_needed": need, "rrr": rrr, "is_league": np.asarray(league)})[FEATS]


def _clf():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=12,
                                          min_samples_leaf=1500, l2_regularization=5.0, random_state=SEED)


def wp_state(clf, inn, bl, wk, runs, target, league, max_balls, result_known=None):
    """P(batting team wins) for a (possibly hypothetical, possibly terminal) state."""
    inn, bl, wk, runs = (np.asarray(a) for a in (inn, bl, wk, runs))
    target, league, max_balls = np.asarray(target, float), np.asarray(league), np.asarray(max_balls, float)
    term = (bl <= 0) | (wk >= 10)
    wp = np.empty(len(inn))
    live = ~term & ~((inn == 2) & (runs >= target))
    if live.any():
        wp[live] = clf.predict_proba(make_X(inn[live], bl[live], wk[live], runs[live], target[live],
                                            league[live], max_balls[live]))[:, 1]
    # terminal: second innings decided by target; first-innings end -> 1 - P(chasing side wins from ball 1)
    t2 = term & (inn == 2); wp[t2] = np.where(runs[t2] >= target[t2], 1.0, np.where(runs[t2] == target[t2] - 1, .5, 0.0))
    wp[(inn == 2) & (runs >= target) & ~term] = 1.0
    t1 = term & (inn == 1)
    if t1.any():
        Xc = make_X(np.full(t1.sum(), 2), max_balls[t1], np.zeros(t1.sum()), np.zeros(t1.sum()),
                    runs[t1] + 1, league[t1], max_balls[t1])
        wp[t1] = 1 - clf.predict_proba(Xc)[:, 1]
    return wp


OUTCOMES = [("dot", 0, 1, 0), ("1", 1, 1, 0), ("2", 2, 1, 0), ("3", 3, 1, 0), ("4", 4, 1, 0),
            ("6", 6, 1, 0), ("W", 0, 1, 1), ("wide", 1, 0, 0)]          # (name, runs, legal_balls, wicket)


def outcome_probs(b: pd.DataFrame) -> pd.DataFrame:
    """League-wide delivery-outcome distribution by phase x wickets-lost bucket (the 'average player')."""
    cat = np.select([b.is_wicket, b.is_wide, b.total_runs.eq(0), b.total_runs.eq(1), b.total_runs.eq(2),
                     b.total_runs.eq(3), b.total_runs.eq(4), b.total_runs >= 5], ["W", "wide", "dot", "1", "2", "3", "4", "6"], default="dot")
    d = pd.DataFrame({"phase": b.phase, "wb": pd.cut(b.wkts_before, [-1, 2, 5, 10], labels=False), "o": cat})
    t = pd.crosstab([d.phase, d.wb], d.o, normalize="index")
    return t[[o[0] for o in OUTCOMES]]


def crossfit_win_prob(b: pd.DataFrame) -> pd.DataFrame:
    """Returns per-delivery wp_before, wp_after, wpa (batting-team view) and expected swing (leverage numerator)."""
    ok = b["batting_team_won"].notna().values
    inn, bl, wk = b.innings.values, b.balls_left.values, b.wkts_before.values
    runs, tgt, lg, mb = b.cum_runs_before.values, b.target.values, b.is_league.values, b.max_balls.values
    legal = b.is_legal.values.astype(int)
    p_out = outcome_probs(b)
    keyP = list(zip(b.phase, pd.cut(b.wkts_before, [-1, 2, 5, 10], labels=False)))
    P = np.vstack([p_out.loc[k].values for k in keyP])               # n x 8 outcome probabilities
    res = pd.DataFrame(index=b.index, columns=["wp_before", "wp_after", "exp_swing"], dtype=float)
    groups = b.match_id.values
    for tr, te in GroupKFold(N_FOLDS).split(b, groups=groups):
        trm = tr[ok[tr]]
        clf = _clf().fit(make_X(inn[trm], bl[trm], wk[trm], runs[trm], tgt[trm], lg[trm], mb[trm]),
                         b.batting_team_won.values[trm])
        wp0 = wp_state(clf, inn[te], bl[te], wk[te], runs[te], tgt[te], lg[te], mb[te])
        # actual after-state (terminal states use the recorded match result: extras imputation can leave chases a few runs short)
        runs_a = runs[te] + b.total_runs.values[te]; wk_a = wk[te] + b.is_wicket.values[te]
        bl_a = bl[te] - legal[te]
        wp1 = wp_state(clf, inn[te], bl_a, wk_a, runs_a, tgt[te], lg[te], mb[te])
        last2 = (inn[te] == 2) & b.groupby(["match_id", "innings"]).cumcount(ascending=False).values[te].astype(bool).__invert__()
        won = b.batting_team_won.values[te]
        wp1 = np.where(last2 & ~np.isnan(won), won, wp1)
        sw = np.zeros(len(te))
        for j, (_, r, l, w) in enumerate(OUTCOMES):
            wpj = wp_state(clf, inn[te], bl[te] - l, wk[te] + w, runs[te] + r, tgt[te], lg[te], mb[te])
            sw += P[te, j] * np.abs(wpj - wp0)
        res.iloc[te, 0], res.iloc[te, 1], res.iloc[te, 2] = wp0, wp1, sw
    res["wpa"] = res.wp_after - res.wp_before
    return res
