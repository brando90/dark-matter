"""Unit tests for py_src/nested_joint_lrt.py (folder 19). Run: python3 -I test_nested_lrt.py  -> prints PASS lines, exits 1 on failure.
They are software checks on a real design grid with synthetic responses; they are not the gate controls."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from py_src.nested_joint_lrt import ALPHA_GRID, GAMMA_GRID, GAP, INFEASIBLE, Design, Fit  # noqa: E402
from py_src.scaling_laws_fit import InsufficientDataError  # noqa: E402

df = pd.read_csv(os.path.join(HERE, "results", "design_owt2.csv"))
N, D, SD = df.N_tied.values, df.D.values, df.sd.values
truth = lambda N, D: 2.0 + 3.0 * (N / 1e6) ** -0.6 + 1.0 * (D / 1e9) ** -0.2  # noqa: E731
des = Design(N, D, SD)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" " + detail if detail else ""))
    if not ok:
        fails.append(name)


rng = np.random.default_rng(0)
y = truth(N, D) + 1e-5 * rng.standard_normal(len(N))
T, f1, f2 = des.statistic(y)
check("M1 recovers exponents at tiny noise", abs(f1.exps[0] - 0.6) < 0.01 and abs(f1.exps[1] - 0.2) < 0.01, str(np.round(f1.exps, 4)))
check("M1 recovers coefficients at tiny noise", np.allclose(f1.coef, [2.0, 3.0, 1.0], atol=0.02), str(np.round(f1.coef, 4)))
check("nested: RSS_2 clipped so T >= 0", T >= 0 and f2.rss >= 0)
planted = truth(N, D) + 0.3 * (N / 5173248.0) ** -1.4 + 1e-5 * rng.standard_normal(len(N))
Tp, g1, g2 = des.statistic(planted)
check("planted fast term: M2 beats M1 by a wide margin", g2.rss < 0.01 * g1.rss and Tp > 100, f"T={Tp:.1f} gamma={g2.exps[2]:.2f}")
check("planted fast term: gamma > alpha + gap", g2.exps[2] >= g2.exps[0] + 0.15 - 1e-9)
a = des.bootstrap_lrt(planted, B=5, rng=np.random.default_rng(3))
b = des.bootstrap_lrt(planted, B=5, rng=np.random.default_rng(3))
check("bootstrap reproducible for a fixed generator", np.array_equal(a["T_boot"], b["T_boot"]) and a["p"] == b["p"])
check("p-value is at least 1/(B+1)", a["p"] >= 1 / 6 - 1e-12)
for bad, msg in ((lambda: Design(N[:6], D[:6], SD[:6]), "too few points"),
                 (lambda: Design(np.full(20, 1e7), D[:20], SD[:20]), "one distinct size")):
    try:
        bad()
        check(f"raises InsufficientDataError: {msg}", False)
    except InsufficientDataError:
        check(f"raises InsufficientDataError: {msg}", True)

# ---- guards added after the stage-3 Mega QA review (finding F1 and F6) ------------------------------------------------
N_inf = N.copy()
N_inf[0] = np.inf
for bad, msg in ((lambda: Design(N_inf, D, SD), "an infinite N"),
                 (lambda: des.fit(y[:1], False), "a length-1 response"),
                 (lambda: des.fit(y[:-1], True), "a response that is one point short")):
    try:
        bad()
        check(f"raises ValueError: {msg}", False)
    except ValueError:
        check(f"raises ValueError: {msg}", True)


class _Fixed(Design):
    """Design whose fits return fixed RSS values, to test statistic() at the edge of its arithmetic."""
    rss_by_model = (0.0, 0.0)

    def fit(self, y, two_terms, n_starts=None):
        return Fit(rss=self.rss_by_model[int(two_terms)], coef=(), exps=(), fitted=np.zeros(self.npts))


fixed = _Fixed(N, D, SD)
fixed.rss_by_model = (0.0, 0.0)
T0 = fixed.statistic(y)[0]
check("exact M1 fit gives T = 0, not nan", T0 == 0.0, f"T={T0}")
fixed.rss_by_model = (1.0, 0.0)
T1 = fixed.statistic(y)[0]
check("exact M2 fit against an imperfect M1 gives a finite large T", np.isfinite(T1) and T1 > 100, f"T={T1:.1f}")

r2 = des._grid_rss(planted, True)
ok2 = np.broadcast_to(des._ok2, r2.shape)
check("grid ranking excludes every cell with gamma < alpha + GAP", np.isinf(r2[~ok2]).all() and np.isfinite(r2[ok2]).all(),
      f"{int((~ok2).sum())} excluded of {r2.size}")
stalled = des._polish([0.6, 0.2, 0.5], planted)   # start violates gamma >= alpha + GAP: Nelder-Mead cannot leave the flat wall
check("polish from an infeasible start returns an infinite RSS, never a fit", not np.isfinite(stalled.rss))

sd_hat = des.bootstrap_lrt(truth(N, D) + SD * rng.standard_normal(len(N)), B=1, rng=np.random.default_rng(1))["sigma_hat"]
bad_picks = bad_fits = 0
gen = np.random.default_rng(11)
for _ in range(40):
    ys = truth(N, D) + sd_hat * SD * gen.standard_normal(len(N))
    for two in (False, True):
        rr = des._grid_rss(ys, two)
        for c in des._distinct_cells(rr, 3 if two else 2):
            bad_picks += int(rr[c] >= INFEASIBLE or (two and GAMMA_GRID[c[2]] < ALPHA_GRID[c[0]] + GAP))
    _, h1, h2 = des.statistic(ys)
    bad_fits += int(not (0.02 <= h2.exps[0] <= 2.0 and 0.02 <= h2.exps[1] <= 1.5 and 0.2 <= h2.exps[2] <= 6.0
                         and h2.exps[2] >= h2.exps[0] + GAP))
check("40 null draws: every start cell feasible and every polished M2 fit feasible", bad_picks == 0 and bad_fits == 0,
      f"infeasible picks {bad_picks}, infeasible fits {bad_fits}")
sys.exit(1 if fails else 0)
