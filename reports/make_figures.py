"""Figures for the methodology report. Run:  PYTHONPATH=. python reports/make_figures.py"""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import PROC, RES, REP, HIGH_LI, EXTREME_LI
from src.data_processing.load_data import load_table

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
F = REP / "figures"; F.mkdir(exist_ok=True)
f = load_table(PROC / "match_features.parquet")
meta = json.load(open(RES / "run_metadata.json"))

# 1. leverage heat-map: over x innings, and wickets x balls-left in chases
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
pv = f.pivot_table(index="innings", columns="over", values="leverage", aggfunc="mean")
im = ax[0].imshow(pv.values, aspect="auto", cmap="YlOrRd"); ax[0].set_yticks([0, 1], ["1st innings", "2nd innings (chase)"])
ax[0].set_xticks(range(0, 20, 2), range(1, 21, 2)); ax[0].set_xlabel("Over"); ax[0].set_title("Mean leverage (pressure) by over")
plt.colorbar(im, ax=ax[0], fraction=.04)
c = f[f.innings == 2].copy(); c["rrr_b"] = pd.cut(c.rrr, [0, 6, 8, 10, 12, 15, 40]); c["ob"] = pd.cut(c.over, [0, 6, 10, 14, 17, 20])
ph = c.pivot_table(index="rrr_b", columns="ob", values="leverage", aggfunc="mean", observed=True)
im = ax[1].imshow(ph.values, aspect="auto", cmap="YlOrRd"); ax[1].set_yticks(range(len(ph.index)), [str(i) for i in ph.index])
ax[1].set_xticks(range(len(ph.columns)), ["1-6", "7-10", "11-14", "15-17", "18-20"]); ax[1].set_xlabel("Over"); ax[1].set_ylabel("Required run rate")
ax[1].set_title("Chases: leverage by required rate x over"); plt.colorbar(im, ax=ax[1], fraction=.04)
plt.tight_layout(); plt.savefig(F / "fig1_leverage_heatmap.png"); plt.close()

# 2. leverage distribution + tier separation
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0))
ax[0].hist(f.pressure_index_raw.clip(upper=8), bins=80, color="#4a6fa5")
for t, col in ((HIGH_LI, "orange"), (EXTREME_LI, "red")): ax[0].axvline(t, color=col, ls="--")
ax[0].set_xlabel("Pressure index (normalised leverage x stakes)"); ax[0].set_ylabel("Deliveries"); ax[0].set_title("Distribution & tier cut-offs")
tp = pd.read_csv(RES / "pressure_tier_profile.csv").set_index("pressure_tier")
ax[1].bar(tp.index, tp.abs_wpa, color=["#9ecae1", "#6baed6", "#fd8d3c", "#d94801"]); ax[1].set_ylabel("Mean |win-prob swing| per ball (%)")
ax[1].set_title("Tiers separate real match impact")
plt.tight_layout(); plt.savefig(F / "fig2_tiers.png"); plt.close()

# 3. calibration
d = f.dropna(subset=["batting_team_won"]).copy(); d["bin"] = pd.cut(d.wp_before, np.linspace(0, 1, 11))
cal = d.groupby("bin", observed=True).agg(p=("wp_before", "mean"), a=("batting_team_won", "mean"))
fig, ax = plt.subplots(figsize=(3.4, 3.2)); ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.plot(cal.p, cal.a, "o-", color="#4a6fa5")
ax.set_xlabel("Predicted win prob (out-of-fold)"); ax.set_ylabel("Observed win rate"); ax.set_title("Win-probability calibration")
plt.tight_layout(); plt.savefig(F / "fig3_calibration.png"); plt.close()

# 4. reliability by component
fig, ax = plt.subplots(figsize=(5.2, 3.0)); lab = {"overall": "Overall skill\n(all balls)", "normal_mean": "Normal-over\nmean", "pressure_mean": "High-pressure\nmean", "lift_D": "Pressure lift\n(D)", "hit_rate": "Hit-rate\n(C)"}
x = np.arange(len(lab))
for i, (role, col) in enumerate((("bat", "#4a6fa5"), ("bowl", "#e08a3c"))):
    r = pd.read_csv(RES / f"reliability_{role}.csv", index_col=0).loc[list(lab)]
    ax.bar(x + (i - .5) * .38, r.mean_rho, .38, yerr=r.sd, label="Batters" if role == "bat" else "Bowlers", color=col)
ax.axhline(0, color="k", lw=.6); ax.set_xticks(x, list(lab.values()), fontsize=7); ax.set_ylabel("Split-half Spearman rho"); ax.legend(frameon=False)
ax.set_title("How repeatable is each signal?"); plt.tight_layout(); plt.savefig(F / "fig4_reliability.png"); plt.close()

# 5. rank intervals
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
for a, role, fn, ttl in ((ax[0], "bat", "top10_batters", "Batters"), (ax[1], "bowl", "top10_bowlers", "Bowlers")):
    t = pd.read_csv(RES / f"{fn}.csv").iloc[::-1]
    a.hlines(range(10), t.rank_p05, t.rank_p95, color="#4a6fa5"); a.plot(t.median_rank, range(10), "o", color="#d94801")
    a.set_yticks(range(10), t["name"], fontsize=8); a.set_xlabel("Rank (median, 90% bootstrap interval)"); a.set_title(f"{ttl}: rank stability")
plt.tight_layout(); plt.savefig(F / "fig5_rank_intervals.png"); plt.close()

# 6. temporal validation
fig, ax = plt.subplots(figsize=(4.8, 3.0))
for i, (role, ttl) in enumerate((("bat", "Batters"), ("bowl", "Bowlers"))):
    t = meta[role]["temporal"]
    vals = [t["rho_rating_vs_post"], t["rho_naive_raw_pressure_vs_post"]]
    ax.bar(np.arange(2) + i * 3, vals, color=["#4a6fa5", "#bbbbbb"])
    ax.errorbar(i * 3, t["rho_rating_vs_post"], yerr=[[t["rho_rating_vs_post"] - t["rho_rating_ci_lo"]], [t["rho_rating_ci_hi"] - t["rho_rating_vs_post"]]], color="k", capsize=3)
ax.set_xticks([0, 1, 3, 4], ["Our rating", "Naive raw\npressure", "Our rating", "Naive raw\npressure"], fontsize=7)
ax.axhline(0, color="k", lw=.6); ax.set_ylabel("Spearman rho with later pressure value"); ax.set_title("Out-of-sample test (batters | bowlers)", fontsize=9)
plt.tight_layout(); plt.savefig(F / "fig6_temporal.png"); plt.close()
print("figures done")
