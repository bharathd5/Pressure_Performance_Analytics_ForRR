"""Empirical-Bayes shrinkage: pulls small-sample estimates toward a prior mean.
x_i ~ N(theta_i, se2_i), theta_i ~ N(mu_i, tau2)  =>  posterior mean = mu_i + tau2/(tau2+se2_i) * (x_i - mu_i).
tau2 is estimated by method of moments (no tuning constant -> reproducible)."""
import numpy as np
import pandas as pd


def dl_tau2(x, se2):
    """DerSimonian-Laird between-player variance (method of moments, weighted by precision)."""
    w = 1 / se2; mu = np.sum(w * x) / w.sum()
    Q = np.sum(w * (x - mu) ** 2)
    denom = w.sum() - np.sum(w ** 2) / w.sum()
    return max((Q - (len(x) - 1)) / denom, 1e-12)


def eb_shrink(x, se2, mu=None, fit_mask=None, tau2_fixed=None):
    x, se2 = np.asarray(x, float), np.asarray(se2, float)
    fm = np.ones(len(x), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    if mu is None:
        mu = np.average(x[fm], weights=1 / se2[fm])
    tau2 = dl_tau2(x[fm], se2[fm]) if tau2_fixed is None else tau2_fixed
    B = tau2 / (tau2 + se2)
    mu_arr = np.broadcast_to(mu, x.shape)
    return mu_arr + B * (x - mu_arr), B, tau2


def entity_skill(df, pid, team, val, min_n_fit=60):
    """Skill of (player, team) in `val` units per delivery, shrunk toward that team's mean. Returns Series keyed by (pid, team)."""
    g = df.groupby([pid, team])[val].agg(["mean", "size"])
    sigma2 = df[val].var()
    team_mu = df.groupby(team)[val].mean()
    mu = team_mu.reindex(g.index.get_level_values(1)).values
    sh, _, _ = eb_shrink(g["mean"].values, sigma2 / g["size"].values, mu=mu, fit_mask=g["size"].values >= min_n_fit)
    return pd.Series(sh, index=g.index)


def q_test(x, se2, fit_mask=None):
    """Cochran Q heterogeneity test: do players differ by more than sampling noise? Returns p-value (small => real differences)."""
    from scipy.stats import chi2
    x, se2 = np.asarray(x, float), np.asarray(se2, float)
    fm = np.ones(len(x), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    w = 1 / se2[fm]; mu = np.sum(w * x[fm]) / w.sum()
    Q = np.sum(w * (x[fm] - mu) ** 2)
    return float(chi2.sf(Q, fm.sum() - 1))
