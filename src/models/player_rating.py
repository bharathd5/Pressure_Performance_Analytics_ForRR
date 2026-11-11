"""Pressure rating, identical for batters and bowlers (only the underlying run-value sign convention differs).

   P  absolute pressure performance : EB-shrunk mean adjusted run-value per High-pressure delivery
   D  lift vs own normal            : EB-shrunk (High-pressure mean  minus  normal-over mean)
   C  consistency                   : EB-shrunk share of pressure matches with positive net run-value ("hit rate")
   Rating = wP*z(P) + wD*z(D) + wC*z(C);  displayed as 50 + 10*Rating
Everything is computed from player-match rows so a (Bayesian) bootstrap can re-weight matches."""
import numpy as np
import pandas as pd
from src.config import MIN_BALLS_FOR_MATCH_HIT, WEIGHTS
from src.models.shrinkage import eb_shrink, q_test


class PMData:
    def __init__(self, pm: pd.DataFrame, min_balls_hit=MIN_BALLS_FOR_MATCH_HIT):
        self.pm = pm.reset_index(drop=True)
        self.codes, self.uniq = pd.factorize(pm.pid)
        c = pm
        self.a = {k: c[k].values.astype(float) for k in ["hi_n", "hi_sum", "hi_ss", "lo_n", "lo_sum", "lo_ss"]}
        self.has = (c.hi_n.values >= min_balls_hit)
        self.hit = self.has & (c.hi_sum.values > 0)
        self.has_any = c.hi_n.values > 0
        self.k = len(self.uniq)
        self.a["al_n"], self.a["al_sum"] = self.a["hi_n"] + self.a["lo_n"], self.a["hi_sum"] + self.a["lo_sum"]
        tot = lambda x: np.bincount(self.codes, weights=x, minlength=self.k)
        self._tot = tot

    def _cluster_se2(self, key_s, key_n, w):
        """Cluster-robust (by match) sampling variance of a player's ratio estimator sum(s)/sum(n).
        Deliveries inside a match are correlated, so per-ball iid variance understates uncertainty."""
        S = lambda x: self._tot(w * x)
        sn, ss = S(self.a[key_n]), S(self.a[key_s])
        R = np.divide(ss, sn, out=np.zeros(self.k), where=sn > 0)
        e = w * (self.a[key_s] - R[self.codes] * self.a[key_n])
        kk = np.bincount(self.codes, weights=(w > 0) * 1.0, minlength=self.k)
        v = np.bincount(self.codes, weights=e ** 2, minlength=self.k) * kk / np.maximum(kk - 1, 1)
        return np.where(sn > 0, v / np.maximum(sn, 1e-9) ** 2, 1e6), R

    def components(self, w=None, eligible=None, weights=WEIGHTS, tau2_fixed=None):
        """Hierarchical (partial-pooling) estimate.
           g  : overall skill, EB-shrunk over ALL deliveries (stable)            -> prior for pressure performance
           D  : pressure lift = (High-tier mean - normal mean), EB-shrunk toward the league lift
           P  : expected value per High-pressure delivery  =  g + (1 - s) * D     (s = league share of High deliveries)
           C  : pressure-match hit rate, EB-shrunk
           Each component is expressed in units of the estimated SD of TRUE between-player differences (sqrt(tau2)),
           so a component that is mostly noise (small tau2) cannot be inflated by z-scoring."""
        w = np.ones(len(self.codes)) if w is None else w
        S = lambda x: self._tot(w * x)
        hi_n, hi_s, lo_n, lo_s = S(self.a["hi_n"]), S(self.a["hi_sum"]), S(self.a["lo_n"]), S(self.a["lo_sum"])
        al_n, al_s = hi_n + lo_n, hi_s + lo_s
        dv = lambda a, b_: np.divide(a, b_, out=np.zeros(self.k), where=b_ > 0)
        m_h, m_l, m_a = dv(hi_s, hi_n), dv(lo_s, lo_n), dv(al_s, al_n)
        tot_n = self.a["hi_n"].sum() + self.a["lo_n"].sum()
        tot_s = self.a["hi_sum"].sum() + self.a["lo_sum"].sum()
        v_a = max((self.a["hi_ss"].sum() + self.a["lo_ss"].sum()) / tot_n - (tot_s / tot_n) ** 2, 1e-9)
        v_h = max(self.a["hi_ss"].sum() / self.a["hi_n"].sum() - (self.a["hi_sum"].sum() / self.a["hi_n"].sum()) ** 2, 1e-9)
        v_l = max(self.a["lo_ss"].sum() / self.a["lo_n"].sum() - (self.a["lo_sum"].sum() / self.a["lo_n"].sum()) ** 2, 1e-9)
        s_bar = self.a["hi_n"].sum() / tot_n
        elig = np.ones(self.k, bool) if eligible is None else eligible
        ok = (hi_n > 0) & (lo_n > 0)
        tf = tau2_fixed or {}
        se_g_cl, _ = self._cluster_se2("al_sum", "al_n", w)
        se_g = np.maximum(np.where(al_n > 0, v_a / np.maximum(al_n, 1e-9), 1e6), se_g_cl)
        g, Bg, tg = eb_shrink(m_a, se_g, fit_mask=elig & (al_n > 0), tau2_fixed=tf.get("g"))
        se_h_cl, _ = self._cluster_se2("hi_sum", "hi_n", w); se_l_cl, _ = self._cluster_se2("lo_sum", "lo_n", w)
        se_d = np.where(ok, np.maximum(v_h / np.maximum(hi_n, 1e-9), se_h_cl) + np.maximum(v_l / np.maximum(lo_n, 1e-9), se_l_cl), 1e6)
        d_raw = np.where(ok, m_h - m_l, 0.0)
        D, Bd, td = eb_shrink(d_raw, se_d, fit_mask=elig & ok, tau2_fixed=tf.get("D"))
        P = g + (1 - s_bar) * D
        tP = tg + (1 - s_bar) ** 2 * td
        mt, ht = S(self.has.astype(float)), S(self.hit.astype(float))
        p_raw = dv(ht, mt) + 0.5 * (mt == 0)
        pbar = ht[elig].sum() / max(mt[elig].sum(), 1e-9)
        se_c = np.where(mt > 0, pbar * (1 - pbar) / np.maximum(mt, 1e-9), 1e6)
        C, Bc, tc = eb_shrink(p_raw, se_c, fit_mask=elig & (mt > 0), tau2_fixed=tf.get("C"))
        z = lambda x, t: (x - x[elig].mean()) / np.sqrt(t)
        out = pd.DataFrame({"pid": self.uniq, "hi_n": hi_n, "lo_n": lo_n, "pres_matches": mt, "hit_rate_raw": p_raw,
                            "P_raw": m_h, "overall_raw": m_a, "D_raw": d_raw, "g": g, "P": P, "D": D, "C": C,
                            "zP": z(P, tP), "zD": z(D, td), "zC": z(C, tc)})
        out["rating_z"] = weights["P"] * out.zP + weights["D"] * out.zD + weights["C"] * out.zC
        out["rating"] = 50 + 10 * out.rating_z
        self.tau2 = {"g": tg, "D": td, "C": tc}
        self.hetero_p = {"g": q_test(m_a, se_g, elig & (al_n > 0)), "D": q_test(d_raw, se_d, elig & ok), "C": q_test(p_raw, se_c, elig & (mt > 0))}
        self.tau = {"g": np.sqrt(tg), "D": np.sqrt(td), "C": np.sqrt(tc)}
        self.reliability = {"g": Bg[elig].mean(), "D": Bd[elig & ok].mean(), "C": Bc[elig].mean()}
        return out.set_index("pid"), elig


def eligible_mask(pmd: PMData, role: str) -> np.ndarray:
    from src.config import MIN_HI_BALLS, MIN_HI_MATCHES
    hi_n = pmd._tot(pmd.a["hi_n"]); mt = pmd._tot(pmd.has_any.astype(float))
    return (hi_n >= MIN_HI_BALLS[role]) & (mt >= MIN_HI_MATCHES[role])
