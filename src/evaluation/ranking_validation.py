"""Out-of-sample and sensitivity validation of the ranking."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.config import WEIGHTS
from src.features.batter_features import player_match_table
from src.models.player_rating import PMData, eligible_mask


def temporal_validation(view, role, high_thr, cutoff, min_pre_hi=60, min_post_hi=40):
    """Rank on matches BEFORE cutoff; test whether the ranking predicts High-pressure performance AFTER cutoff.
    Compares the hierarchical rating with the naive 'best raw pressure mean' ranking."""
    pm = player_match_table(view, role, high_thr)
    pre, post = pm[pm.date < cutoff], pm[pm.date >= cutoff]
    pd_pre = PMData(pre); el = pd_pre._tot(pd_pre.a["hi_n"]) >= min_pre_hi
    comp, _ = pd_pre.components(eligible=el)
    comp = comp[el]
    po = post.groupby("pid")[["hi_n", "hi_sum", "lo_n", "lo_sum"]].sum()
    po = po[po.hi_n >= min_post_hi]
    po = pd.DataFrame({"post_hi_n": po.hi_n, "post_hi_mean": po.hi_sum / po.hi_n})
    j = comp.join(po, how="inner")
    res = {"n_players": len(j),
           "rho_rating_vs_post": spearmanr(j.rating, j.post_hi_mean)[0],
           "rho_hierarchical_P_vs_post": spearmanr(j.P, j.post_hi_mean)[0],
           "rho_naive_raw_pressure_vs_post": spearmanr(j.P_raw, j.post_hi_mean)[0],
           "rho_overall_raw_vs_post": spearmanr(j.overall_raw, j.post_hi_mean)[0]}
    rng = np.random.default_rng(0); rs = []
    for _ in range(1000):
        i = rng.integers(0, len(j), len(j)); rs.append(spearmanr(j.rating.values[i], j.post_hi_mean.values[i])[0])
    res["rho_rating_ci_lo"], res["rho_rating_ci_hi"] = np.nanpercentile(rs, [5, 95])
    top = j.sort_values("rating", ascending=False).head(10)
    top_naive = j.sort_values("P_raw", ascending=False).head(10)
    res["post_mean_top10_rating"] = np.average(top.post_hi_mean, weights=top.post_hi_n)
    res["post_mean_top10_naive"] = np.average(top_naive.post_hi_mean, weights=top_naive.post_hi_n)
    res["post_mean_all_eligible"] = np.average(j.post_hi_mean, weights=j.post_hi_n)
    return res


def sensitivity(view, role, base_thr, thresholds=(1.3, 1.6, 2.0), weight_sets=None, k=10):
    weight_sets = weight_sets or {"base": WEIGHTS, "P-only": {"P": 1, "D": 0, "C": 0}, "equal": {"P": 1/3, "D": 1/3, "C": 1/3},
                                  "lift-heavy": {"P": .3, "D": .4, "C": .3}}
    def top(thr, w):
        pm = player_match_table(view, role, thr); pmd = PMData(pm); el = eligible_mask(pmd, role)
        comp, _ = pmd.components(eligible=el, weights=w)
        return list(comp[el].sort_values("rating", ascending=False).index[:k])
    base = top(base_thr, WEIGHTS)
    rows = []
    for thr in thresholds:
        rows.append({"variant": f"High threshold LI>={thr}", "overlap_with_base_top10": len(set(base) & set(top(thr, WEIGHTS)))})
    for n, w in weight_sets.items():
        rows.append({"variant": f"weights {n}", "overlap_with_base_top10": len(set(base) & set(top(base_thr, w)))})
    return pd.DataFrame(rows)
