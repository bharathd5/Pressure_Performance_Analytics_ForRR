"""End-to-end run:  raw CSV -> clean -> pressure engine -> adjusted values -> ratings -> validation -> results/."""
import json
import numpy as np
import pandas as pd
from src.config import PROC, RES, HIGH_LI, EXTREME_LI, WEIGHTS, MIN_HI_BALLS, MIN_HI_MATCHES
from src.data_processing.load_data import load_raw, save_table
from src.data_processing.clean_data import clean_deliveries
from src.data_processing.data_validation import run_checks
from src.features.match_features import build_match_features
from src.models.opponent_adjustment import adjusted_views
from src.features.batter_features import player_match_table
from src.features.matchup_features import player_form
from src.models.player_rating import PMData, eligible_mask
from src.pressure_engine.pressure_classifier import tier_profile
from src.evaluation.consistency import split_half, bayesian_bootstrap_ranks
from src.evaluation.ranking_validation import temporal_validation, sensitivity

ROLES = {"bat": ("striker_id", "striker_name"), "bowl": ("bowler_id", "bowler_name")}


def main():
    raw = load_raw()
    clean = clean_deliveries(raw)
    checks = run_checks(clean); print(checks)
    save_table(clean, PROC / "cleaned_deliveries.parquet")
    feats = build_match_features(clean)
    save_table(feats, PROC / "match_features.parquet")
    prof = tier_profile(feats); prof.to_csv(RES / "pressure_tier_profile.csv")
    bv, wv = adjusted_views(feats)
    names = pd.concat([feats[["striker_id", "striker_name"]].set_axis(["pid", "name"], axis=1),
                       feats[["bowler_id", "bowler_name"]].set_axis(["pid", "name"], axis=1)]).drop_duplicates("pid").set_index("pid").name
    out, evals, all_players = {}, [], []
    cutoff = feats.match_date.quantile(0.55)
    meta = {"checks": {k: (float(v) if not isinstance(v, int) else v) for k, v in checks.items()}, "cutoff": str(cutoff.date())}
    for role, view in (("bat", bv), ("bowl", wv)):
        pid_col = ROLES[role][0]
        pm = player_match_table(view, role, HIGH_LI)
        pmd = PMData(pm); el = eligible_mask(pmd, role)
        comp, _ = pmd.components(eligible=el)
        comp["eligible"] = el
        snap = {"tau": dict(pmd.tau), "rel": dict(pmd.reliability), "het": dict(pmd.hetero_p)}   # before bootstrap reuses pmd
        comp["name"] = comp.index.map(names)
        # descriptive raw stats (High pressure vs normal)
        g = pm.groupby("pid")[[c for c in pm.columns if c.startswith(("hi_", "lo_")) and not c.endswith(("_sum", "_ss", "_n"))]].sum()
        comp = comp.join(g)
        comp["n_matches"] = pm.groupby("pid").match_id.nunique()
        comp["share_ipl_pressure_balls"] = pm.assign(w=pm.hi_n * pm.is_league).groupby("pid").w.sum() / pm.groupby("pid").hi_n.sum().replace(0, np.nan)
        comp = comp.join(player_form(view, pid_col).set_index("pid")["form"])
        bb = bayesian_bootstrap_ranks(pmd, el)
        comp = comp.join(bb)
        sh = split_half(pmd, el)
        sh.to_csv(RES / f"reliability_{role}.csv")
        tv = temporal_validation(view, role, HIGH_LI, cutoff)
        sens = sensitivity(view, role, HIGH_LI)
        sens.to_csv(RES / f"sensitivity_{role}.csv", index=False)
        comp = comp.sort_values(["eligible", "rating"], ascending=False)
        comp.to_csv(RES / f"all_players_{role}.csv"); all_players.append(comp.assign(role=role).reset_index())
        top = comp[comp.eligible].head(10).copy(); top.insert(0, "rank", range(1, 11))
        top.to_csv(RES / ("top10_batters.csv" if role == "bat" else "top10_bowlers.csv"))
        meta[role] = {"n_eligible": int(el.sum()), "tau": {k: float(v) for k, v in snap["tau"].items()},
                      "reliability_shrink": {k: float(v) for k, v in snap["rel"].items()}, "temporal": {k: float(v) for k, v in tv.items()},
                      "hetero_p": {k: float(v) for k, v in snap["het"].items()}, "split_half": sh["mean_rho"].round(3).to_dict()}
        evals.append(pd.DataFrame({"role": role, "metric": list(tv.keys()), "value": list(tv.values())}))
        print(role, "\n", top[["rank", "name", "hi_n", "n_matches", "P", "g", "rating", "p_top10", "median_rank"]].round(3).to_string())
        print(sh.round(3).to_string()); print(tv); print(sens.to_string())
    save_table(pd.concat(all_players, ignore_index=True), PROC / "player_features.parquet")
    pd.concat(evals).to_csv(RES / "evaluation_results.csv", index=False)
    json.dump(meta, open(RES / "run_metadata.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
