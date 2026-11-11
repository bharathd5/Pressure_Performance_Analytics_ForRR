"""Reliability / consistency diagnostics."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.config import SEED


def split_half(pmd, eligible, reps=200, seed=SEED):
    """Random match-level halves. Spearman correlation of each RAW component across the two halves (higher = more reliable)."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["overall", "pressure_mean", "lift_D", "hit_rate", "normal_mean"]}
    dv = lambda a, b: a / np.maximum(b, 1)
    for _ in range(reps):
        wA = (rng.random(len(pmd.codes)) < .5).astype(float)
        st = []
        for w in (wA, 1 - wA):
            S = lambda x: pmd._tot(w * x)
            hn, hs, ln, ls, mt, ht = S(pmd.a["hi_n"]), S(pmd.a["hi_sum"]), S(pmd.a["lo_n"]), S(pmd.a["lo_sum"]), S(pmd.has.astype(float)), S(pmd.hit.astype(float))
            st.append(dict(overall=dv(hs + ls, hn + ln), pressure_mean=dv(hs, hn), lift_D=dv(hs, hn) - dv(ls, ln), hit_rate=dv(ht, mt),
                           normal_mean=dv(ls, ln), ok=(hn >= 15) & (ln >= 30) & (mt >= 2)))
        ok = eligible & st[0]["ok"] & st[1]["ok"]
        for k in out:
            out[k].append(spearmanr(st[0][k][ok], st[1][k][ok])[0])
    return pd.DataFrame({k: [np.nanmean(v), np.nanstd(v)] for k, v in out.items()}, index=["mean_rho", "sd"]).T


def bayesian_bootstrap_ranks(pmd, eligible, reps=300, seed=SEED, weights=None):
    """Cluster (match-level) Bayesian bootstrap: Exp(1) weights on player-match rows; recompute rating; track rank."""
    from src.config import WEIGHTS
    rng = np.random.default_rng(seed)
    base, _ = pmd.components(eligible=eligible, weights=weights or WEIGHTS)
    fixed = dict(pmd.tau2)                      # hold hyper-parameters fixed; resample the DATA (matches)
    el = np.where(eligible)[0]
    ranks = np.empty((reps, len(el)))
    for r in range(reps):
        w = rng.exponential(1.0, len(pmd.codes))
        comp, _ = pmd.components(w=w, eligible=eligible, weights=weights or WEIGHTS, tau2_fixed=fixed)
        ranks[r] = (-comp.rating.values[el]).argsort().argsort() + 1
    return pd.DataFrame({"pid": pmd.uniq[el], "median_rank": np.median(ranks, 0), "rank_p05": np.percentile(ranks, 5, 0),
                         "rank_p95": np.percentile(ranks, 95, 0), "p_top10": (ranks <= 10).mean(0), "p_top5": (ranks <= 5).mean(0)}).set_index("pid")
