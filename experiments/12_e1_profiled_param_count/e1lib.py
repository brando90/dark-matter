"""Shared E1/E3 helpers: counts, a vectorized 1-term fitter (DM-1b's 1-term branch), profile sets, the DM-1b LRT.

Law convention: alpha is the FAST exponent. In DM-1b's fit_batch output, `beta` is the fast one and `alpha` the
slow one; `two_term_named` translates.
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DM5 = os.path.join(ROOT, "experiments", "dm5_small_to_large")
sys.path.insert(0, DM5)
import dm1  # noqa: E402
import dm1b  # noqa: E402
import dm5  # noqa: E402

N_REF = 1e6
CHI2_95 = 3.841
GRID_FINE = np.round(np.arange(0, 3.00001, 0.01), 3)
GRID_005 = np.round(np.arange(0, 3.00001, 0.05), 3)
GRID_01 = np.round(np.arange(0, 3.00001, 0.1), 3)
C_EXTRA = [4.0, 5.0, 7.0, 10.0]


def n_of(c, ne, nemb):
    """N_c / 1e6 for N_c = N_nonemb + c N_emb."""
    return (np.asarray(ne, float) + c * np.asarray(nemb, float)) / N_REF


def fit1_vec(n, Y, max_rounds=150):
    """1-term fit L = E + a n^-p of every row of Y (R, npts): DM-1b fit_batch's 1-term branch, unchanged (grid on
    DM-1's ALPHA_GRID, compass search to step 1e-7). Returns dict of arrays E, a, p, rss."""
    Y = np.atleast_2d(np.asarray(Y, float))
    ym = Y.mean(1)
    Yc = Y - ym[:, None]
    R = len(Y)
    _, _, rss = dm1b._proj(dm1b._pow(n, dm1.ALPHA_GRID[:, None]), Yc, ym)
    al = dm1.ALPHA_GRID[np.argmin(rss, 0)].copy()
    best = rss.min(0)
    step = np.full(R, dm1.ALPHA_GRID[1] - dm1.ALPHA_GRID[0])
    off = np.array([-1.0, 1.0])
    for _ in range(max_rounds):
        act = np.flatnonzero(step >= 1e-7)
        if not len(act):
            break
        cand = np.clip(al[act, None] + step[act, None] * off[None, :], 0.005, 2.5)
        _, _, rss = dm1b._proj(dm1b._pow(n, cand[..., None]), Yc[act], ym[act])
        j = np.argmin(rss, 1)
        v = rss[np.arange(len(act)), j]
        better = v < best[act]
        al[act] = np.where(better, cand[np.arange(len(act)), j], al[act])
        best[act] = np.where(better, v, best[act])
        step[act] = np.where(better, step[act], step[act] * 0.5)
    E1, c1, rss1 = dm1b._proj(dm1b._pow(n, al[:, None, None]), Yc, ym)
    return dict(E=E1[:, 0], a=c1[:, 0, 0], p=al, rss=rss1[:, 0])


def fit1(n, y):
    f = fit1_vec(n, y[None, :])
    return {k: float(v[0]) for k, v in f.items()}


def pred1(f, n):
    return f["E"] + f["a"] * np.asarray(n, float) ** -f["p"]


def fit2(n, y):
    """DM-1b fit_batch on one series -> (f1, f2) as float dicts (DM-1b names: f2 beta is the fast exponent)."""
    return dm5.one(n, y)


def two_term_named(f2):
    """DM-1b names -> repo convention (alpha fast, beta slow). b = 0 means the 2-term fit collapsed to 1 term."""
    return dict(E=f2["E"], A_fast=f2["b"], alpha_fast=f2["beta"], B_slow=f2["a"], beta_slow=f2["alpha"],
                rss=f2["rss"])


def pred2(f2, n):
    return dm5.law(np.asarray(n, float), f2)


def lrt(n, y, seed):
    """DM-1b bootstrap LRT (B = 200, detect iff (1 + k)/201 <= 0.05), as DM-5's lrt_rule.
    Returns dict(detect, p, T, tested, f1, f2)."""
    f1, f2, det, p, T, tested, nbad = dm5.lrt_rule(n, y, np.random.default_rng(seed))
    return dict(detect=int(det), p=float(p), T=float(T), tested=int(tested), nbad=int(nbad), f1=f1, f2=f2)


def refine(fun, grid, vals):
    """Bounded scalar refinement between the grid minimum's neighbours; returns (c, value)."""
    vals = np.asarray(vals, float)
    i = int(np.nanargmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    if b <= a:
        return float(grid[i]), float(vals[i])
    r = minimize_scalar(fun, bounds=(a, b), method="bounded", options={"xatol": 1e-5})
    return (float(r.x), float(r.fun)) if r.fun <= vals[i] else (float(grid[i]), float(vals[i]))


def profile_set(grid, rss, rss_min, npts):
    """95% profile set {c : n ln(RSS(c)/RSS_min) <= 3.841} (DM-3's profile_ci, any grid). Returns dict with the
    intervals [(lo, hi, open_at_top)], text, total length, excludes 1 / 0, and the boolean mask on the grid."""
    stat = npts * np.log(np.maximum(rss, 1e-300) / rss_min)
    ok = stat <= CHI2_95

    def edge(i, j):
        s0, s1 = stat[i], stat[j]
        return grid[i] + (CHI2_95 - s0) * (grid[j] - grid[i]) / (s1 - s0) if np.isfinite(s1) and s1 != s0 else grid[i]
    ivs, i = [], 0
    while i < len(grid):
        if ok[i]:
            j = i
            while j + 1 < len(grid) and ok[j + 1]:
                j += 1
            lo = edge(i, i - 1) if i > 0 else grid[0]
            hi = edge(j, j + 1) if j < len(grid) - 1 else grid[-1]
            ivs.append((float(lo), float(hi), j == len(grid) - 1))
            i = j + 1
        else:
            i += 1
    txt = " U ".join(f"[{a:.2f}, {b:.2f}{'+' if o else ''}]" for a, b, o in ivs)
    return dict(intervals=ivs, text=txt, length=float(sum(b - a for a, b, _ in ivs)),
                excludes1=not any(a <= 1.0 <= b for a, b, _ in ivs),
                excludes0=not any(a <= 0.0 <= b for a, b, _ in ivs), mask=ok, stat=stat)
