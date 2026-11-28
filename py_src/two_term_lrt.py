"""Calibrated likelihood-ratio test for a second power-law term in N (port of DM-1b).

Model pair, on n = N / N_REF (N_REF = 1e6, as in DM-1b):
    1 term :  L(N) = E + a n^-p                       E, a >= 0
    2 terms:  L(N) = E + a n^-p + b n^-(p + d)        E, a, b >= 0, d > 0

Statistic T = npts * ln(RSS_1 / RSS_2) on the loss L itself (Gaussian additive noise, not log-residuals), calibrated by
a parametric bootstrap: B refits of data simulated from the fitted 1-term law with noise sd sqrt(RSS_1 / (npts - 3)).
Detection rule: (1 + k) / (B + 1) <= level, k = number of bootstrap T* that reach T. DM-1b measured this rule's null
rejection at 0.049 pooled (12,000 null replicates) and 0.070 in the worst cell (experiments/dm1_power_analysis/
RESULTS_DM1b.md). The old chi-square test on the Huber objective, scaling_laws_fit.lr_test_chi2_uncalibrated, is not
calibrated: folder 16 measured its statistic 4e4 to 9e4 times below its cutoff on Chinchilla.

Provenance. The block between the BEGIN and END markers is copied from experiments/dm1_power_analysis/dm1b.py
(commit 3626ba0) with exactly three textual changes: dm1.ALPHA_GRID, dm1.A2_GRID, dm1.D2_GRID became module constants.
experiments/17_*/parity_dm1b.py re-derives this block from dm1b.py, applies the same three changes, and fails if it
differs from this file; it also reproduces committed DM-1b rows (T, k, B used, p, decision) from their seeds.

Naming. DM-1b's fitter calls the SLOW exponent alpha and the FAST exponent beta (beta = alpha + delta). The repo's
fit_two_term_N_only uses the opposite letters (alpha fast, beta slow, beta < alpha). The dataclasses below use the
repo's letters, and every docstring says which term is the fast one (the one that matters at small N).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_REF = 1e6
B_DEFAULT = 200
B_STAGE1 = 50
MIN_POINTS_LRT = 7   # DM-1b: a ladder is testable iff npts > 6
MIN_DISTINCT_N = 5   # five free parameters in the two-term law

CHUNK = 1000  # series fit at once
DELTA_BANDS = [(0.0, 0.8), (0.8, 2.0), (2.0, 4.01)]
ALPHA_GRID = np.linspace(0.01, 2.0, 400)
A2_GRID, D2_GRID = np.linspace(0.02, 1.5, 60), np.linspace(0.02, 3.0, 60)

# --- BEGIN verbatim port of experiments/dm1_power_analysis/dm1b.py (_proj .. fit_chunked) --------------------------
def _proj(X, Yc, ym):
    """Nonnegative-checked least squares with an intercept for many series at once, by stable orthogonalization
    (classical Gram-Schmidt applied twice) of the centred power columns, so RSS = |y_c|^2 - |Q^T y_c|^2 is accurate even
    when n^-alpha and n^-beta are nearly collinear. X: (..., k, n) raw columns (k = 1 or 2), shared ((G, k, n) ->
    results (G, R)) or one set per series ((R, C, k, n) -> results (R, C)). Returns (E, slopes (..., k), rss), rss =
    inf where E or a slope is < 0."""
    xm = X.mean(-1)
    Xc = X - xm[..., None]
    shared = X.ndim == 3
    dot = (lambda Q: np.einsum("gn,rn->gr", Q, Yc)) if shared else (lambda Q: np.einsum("rcn,rn->rc", Q, Yc))
    yy = (Yc ** 2).sum(1)[None, :] if shared else (Yc ** 2).sum(1)[:, None]
    x1 = Xc[..., 0, :]
    r11 = np.linalg.norm(x1, axis=-1)
    q1 = x1 / r11[..., None]
    c1 = dot(q1)
    if X.shape[-2] == 1:
        a = c1 / (r11[:, None] if shared else r11)
        E = (ym[None, :] - a * xm[:, None, 0]) if shared else (ym[:, None] - a * xm[..., 0])
        rss = np.maximum(yy - c1 ** 2, 0.0)
        return E, a[..., None], np.where((E < 0) | (a < 0), np.inf, rss)
    x2 = Xc[..., 1, :]
    r12 = (q1 * x2).sum(-1)
    v = x2 - r12[..., None] * q1
    t = (q1 * v).sum(-1)
    v = v - t[..., None] * q1
    r12 = r12 + t
    r22 = np.linalg.norm(v, axis=-1)
    r22 = np.where(r22 > 0, r22, 1e-300)
    q2 = v / r22[..., None]
    c2 = dot(q2)
    if shared:
        b = c2 / r22[:, None]
        a = (c1 - r12[:, None] * b) / r11[:, None]
        E = ym[None, :] - a * xm[:, None, 0] - b * xm[:, None, 1]
    else:
        b = c2 / r22
        a = (c1 - r12 * b) / r11
        E = ym[:, None] - a * xm[..., 0] - b * xm[..., 1]
    rss = np.maximum(yy - c1 ** 2 - c2 ** 2, 0.0)
    return E, np.stack([a, b], -1), np.where((E < 0) | (a < 0) | (b < 0), np.inf, rss)


def _pow(n, exps):
    """Power columns n^-e for exponents exps (..., k) -> (..., k, n)."""
    return np.exp(-exps[..., None] * np.log(n))


def fit_batch(n, Y, max_rounds=150):
    """1-term and 2-term fits of every row of Y (R, n). Grid search on DM-1's grids, then a compass search per series
    (move to the best neighbour and double the step, up to 4 grid steps, or halve it when the centre is best) until
    the step is < 1e-7 (1-term) or 1e-3 (2-term, then polished by Levenberg-Marquardt).
    Returns dicts of arrays; rss2 = min(rss1, best 2-term rss), so the 1-term fit is the 2-term fit with b = 0."""
    ym = Y.mean(1)
    Yc = Y - ym[:, None]
    R = len(Y)
    ar = np.arange(R)
    # 1-term
    _, _, rss = _proj(_pow(n, ALPHA_GRID[:, None]), Yc, ym)
    al = ALPHA_GRID[np.argmin(rss, 0)].copy()
    best = rss.min(0)
    step = np.full(R, ALPHA_GRID[1] - ALPHA_GRID[0])
    off = np.array([-1.0, 1.0])
    for _ in range(max_rounds):
        act = np.flatnonzero(step >= 1e-7)  # only series still searching
        if not len(act):
            break
        cand = np.clip(al[act, None] + step[act, None] * off[None, :], 0.005, 2.5)
        _, _, rss = _proj(_pow(n, cand[..., None]), Yc[act], ym[act])
        j = np.argmin(rss, 1)
        v = rss[np.arange(len(act)), j]
        better = v < best[act]
        al[act] = np.where(better, cand[np.arange(len(act)), j], al[act])
        best[act] = np.where(better, v, best[act])
        step[act] = np.where(better, step[act], step[act] * 0.5)
    E1, c1, rss1 = _proj(_pow(n, al[:, None, None]), Yc, ym)
    f1 = dict(E=E1[:, 0], a=c1[:, 0, 0], alpha=al, rss=rss1[:, 0])
    # 2-term: DM-1's grid, extended to the delta bound (4.0); the RSS surface is multimodal in delta (a fast term
    # fitting only the smallest models sits at the bound), so a compass search starts from the best grid point in
    # each of three delta bands and from two boundary points (below), and the best end point is kept
    D2 = np.concatenate([D2_GRID, [3.25, 3.5, 3.75, 4.0]])
    AA, DD = np.meshgrid(A2_GRID, D2, indexing="ij")
    aa, dd = AA.ravel(), DD.ravel()
    _, _, rss = _proj(_pow(n, np.stack([aa, aa + dd], 1)), Yc, ym)
    starts = []
    for lo, hi in DELTA_BANDS:
        m = (dd >= lo) & (dd < hi)
        i = np.argmin(np.where(m[:, None], rss, np.inf), 0)
        starts.append((aa[i], dd[i], rss[i, ar]))
    # two boundary starts DM-1's Nelder-Mead reached from the grid: a "spike" (delta at its bound, the fast term fitting
    # only the smallest models) and a near-constant slow term (alpha at its bound, E -> 0)
    for sa0, sd0 in ((f1["alpha"], np.full(R, 3.99)), (np.full(R, 0.01), np.maximum(f1["alpha"] - 0.01, 0.005))):
        _, _, r0 = _proj(_pow(n, np.stack([sa0, sa0 + sd0], -1)[:, None, :]), Yc, ym)
        starts.append((sa0, sd0, r0[:, 0]))
    pa, pd, best = (np.concatenate(x) for x in zip(*starts))
    K = len(starts)
    Yk, ymk, Rk = np.tile(Yc, (K, 1)), np.tile(ym, K), K * R
    sa = np.full(Rk, A2_GRID[1] - A2_GRID[0])
    sd = np.full(Rk, D2_GRID[1] - D2_GRID[0])
    oa = np.array([-1.0, 1.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0])
    od = np.array([0.0, 0.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    SA_MAX, SD_MAX = 4 * sa[0], 4 * sd[0]
    for _ in range(max_rounds):
        act = np.flatnonzero(np.maximum(sa, sd) >= 1e-3)
        if not len(act):
            break
        ra = np.arange(len(act))
        ca = np.clip(pa[act, None] + sa[act, None] * oa[None, :], 0.005, 2.5)
        cd = np.clip(pd[act, None] + sd[act, None] * od[None, :], 0.005, 4.0)
        _, _, rss = _proj(_pow(n, np.stack([ca, ca + cd], -1)), Yk[act], ymk[act])
        j = np.argmin(rss, 1)
        v = rss[ra, j]
        better = v < best[act]
        pa[act], pd[act] = np.where(better, ca[ra, j], pa[act]), np.where(better, cd[ra, j], pd[act])
        best[act] = np.where(better, v, best[act])
        sa[act] = np.where(better, np.minimum(sa[act] * 2.0, SA_MAX), sa[act] * 0.5)
        sd[act] = np.where(better, np.minimum(sd[act] * 2.0, SD_MAX), sd[act] * 0.5)
    # polish every end point with projected Levenberg-Marquardt in all five parameters (narrow curved valleys and the
    # E = 0 bound stall a compass search), then keep the best of the K
    E2, c2, _ = _proj(_pow(n, np.stack([pa, pa + pd], -1)[:, None, :]), Yk, ymk)
    ok = np.isfinite(best)
    P = np.stack([np.where(ok, E2[:, 0], 0.0), np.where(ok, c2[:, 0, 0], 0.0), pa,
                  np.where(ok, c2[:, 0, 1], 0.0), pd], 1)
    P, rssk = _lm(n, np.tile(Y, (K, 1)), P, ok)
    k = np.argmin(rssk.reshape(K, R), 0)
    P, r2 = P.reshape(K, R, 5)[k, ar], rssk.reshape(K, R)[k, ar]
    use2 = np.isfinite(r2) & (r2 < f1["rss"])
    f2 = dict(E=np.where(use2, P[:, 0], f1["E"]), a=np.where(use2, P[:, 1], f1["a"]),
              alpha=np.where(use2, P[:, 2], f1["alpha"]), b=np.where(use2, P[:, 3], 0.0),
              beta=np.where(use2, P[:, 2] + P[:, 4], f1["alpha"] + 1.0), rss=np.where(use2, r2, f1["rss"]))
    return f1, f2


LO = np.array([0.0, 0.0, 0.005, 0.0, 0.005])
HI = np.array([np.inf, np.inf, 2.5, np.inf, 4.0])


def _lm(n, Y, P, ok, iters=40):
    """Projected Levenberg-Marquardt for E + a n^-alpha + b n^-(alpha + delta), P = (E, a, alpha, b, delta) per row,
    bounds LO/HI. Rows with ok = False keep rss = inf. Returns (P, rss) with rss computed from the residuals."""
    ln = np.log(n)

    def resid(P):
        ua = np.exp(-P[:, 2:3] * ln)
        ub = np.exp(-(P[:, 2:3] + P[:, 4:5]) * ln)
        return Y - (P[:, 0:1] + P[:, 1:2] * ua + P[:, 3:4] * ub), ua, ub

    r, ua, ub = resid(P)
    rss = np.where(ok, (r ** 2).sum(1), np.inf)
    lam = np.full(len(P), 1e-3)
    for _ in range(iters):
        act = np.flatnonzero(ok & (lam < 1e8))
        if not len(act):
            break
        Pa, ra, uaa, uba = P[act], r[act], ua[act], ub[act]
        db = -Pa[:, 3:4] * ln * uba
        J = np.stack([np.ones_like(uaa), uaa, -Pa[:, 1:2] * ln * uaa + db, uba, db], 2)  # (A, n, 5)
        D = np.sqrt((J ** 2).sum(1)) + 1e-300
        Js = J / D[:, None, :]
        M = np.einsum("anj,ank->ajk", Js, Js) + lam[act, None, None] * np.eye(5)
        g = np.einsum("anj,an->aj", Js, ra)
        step = np.linalg.solve(M, g[..., None])[..., 0] / D
        Pn = np.clip(Pa + step, LO, HI)
        rn, uan, ubn = resid_rows(Y[act], Pn, ln)
        sn = (rn ** 2).sum(1)
        better = sn < rss[act]
        P[act] = np.where(better[:, None], Pn, Pa)
        r[act] = np.where(better[:, None], rn, ra)
        ua[act] = np.where(better[:, None], uan, uaa)
        ub[act] = np.where(better[:, None], ubn, uba)
        rss[act] = np.where(better, sn, rss[act])
        lam[act] = np.where(better, lam[act] / 3.0, lam[act] * 4.0)
    return P, rss


def resid_rows(Y, P, ln):
    ua = np.exp(-P[:, 2:3] * ln)
    ub = np.exp(-(P[:, 2:3] + P[:, 4:5]) * ln)
    return Y - (P[:, 0:1] + P[:, 1:2] * ua + P[:, 3:4] * ub), ua, ub


def lrt_stat(npts, rss1, rss2):
    return npts * np.log(np.maximum(rss1, 1e-300) / np.maximum(rss2, 1e-300))


def fit_chunked(n, Y):
    parts = [fit_batch(n, Y[i:i + CHUNK]) for i in range(0, len(Y), CHUNK)]
    return [{k: np.concatenate([p[j][k] for p in parts]) for k in parts[0][j]} for j in (0, 1)]


# --- END verbatim port ---------------------------------------------------------------------------------------------


class InsufficientDataError(ValueError):
    """The selection is empty, or too small to fit or test the two-term law. Never answered by using other data."""


class FitError(RuntimeError):
    """A fit returned no usable solution (infeasible, non-finite, or the optimizer failed outright)."""


def validate_ladder(N, L, *, min_points: int, who: str):
    """Return (N, L) as float arrays, or raise: shape mismatch, NaN/inf, non-positive values, too few points.

    A ladder with fewer than `min_points` points or fewer than MIN_DISTINCT_N distinct sizes raises
    InsufficientDataError; callers must not respond by fitting a different (larger) data set.
    """
    N = np.asarray(N, dtype=float)
    L = np.asarray(L, dtype=float)
    if N.ndim != 1 or L.ndim != 1 or N.shape != L.shape:
        raise ValueError(f"{who}: N and L must be 1-D arrays of the same length, got shapes {N.shape} and {L.shape}")
    if N.size == 0:
        raise InsufficientDataError(f"{who}: empty selection (0 points); refusing to fall back to other data")
    bad = ~(np.isfinite(N) & np.isfinite(L))
    if bad.any():
        raise ValueError(f"{who}: {int(bad.sum())} of {N.size} points are NaN or infinite")
    if (N <= 0).any() or (L <= 0).any():
        raise ValueError(f"{who}: N and L must be strictly positive (min N {N.min():.4g}, min L {L.min():.4g})")
    if N.size < min_points:
        raise InsufficientDataError(
            f"{who}: need at least {min_points} points, got {N.size}; refusing to fall back to other data")
    distinct = int(np.unique(N).size)
    if distinct < MIN_DISTINCT_N:
        raise InsufficientDataError(
            f"{who}: need at least {MIN_DISTINCT_N} distinct sizes N for five free parameters, got {distinct}")
    return N, L


@dataclass(frozen=True)
class OneTermFit:
    """L = E + A N^-alpha, N in raw units."""
    E: float
    A: float
    alpha: float
    rss: float


@dataclass(frozen=True)
class TwoTermFit:
    """L = E + A N^-alpha + B N^-beta, N in raw units; alpha is the FAST exponent (alpha > beta), beta the slow one.

    collapsed = True means the best two-term fit is the one-term fit (A = 0); alpha is then a placeholder.
    Attribute names match scaling_laws_fit.TwoTermParams, so predict_two_term accepts this object.
    """
    E: float
    A: float
    alpha: float
    B: float
    beta: float
    rss: float
    collapsed: bool


@dataclass(frozen=True)
class LRTResult:
    T: float                # npts * ln(RSS_1 / RSS_2)
    p_value: float          # (1 + k) / (B_used + 1); exact for B_used == B, see early_stopped
    k_exceed: int           # bootstrap statistics T* that reach T (non-finite T* are counted as reaching it)
    B: int                  # bootstrap size requested
    B_used: int             # refits actually run
    level: float
    detect: bool            # second term detected at `level`; decided on all B refits unless early_stopped
    early_stopped: bool     # True: the first B_STAGE1 refits already forced detect = False, p_value is then rough
    n_points: int
    sigma_hat: float        # sqrt(RSS_1 / (npts - 3)), the bootstrap noise sd
    n_nonfinite_boot: int
    one_term: OneTermFit
    two_term: TwoTermFit

    def summary(self) -> str:
        verdict = "DETECTED" if self.detect else "not detected"
        stop = " (early stop)" if self.early_stopped else ""
        return (f"second N term {verdict} at level {self.level}: T = {self.T:.4g}, p = {self.p_value:.4g} "
                f"({self.k_exceed} of {self.B_used} bootstrap statistics reach T{stop}), n = {self.n_points}")


def _fit_dicts(n, Y):
    f1, f2 = fit_chunked(n, Y)
    return f1, f2


def _to_raw_units(f1, f2, n_ref):
    """DM-1b fit dicts (n = N / n_ref) -> OneTermFit, TwoTermFit in raw N units, repo letters."""
    one = OneTermFit(E=float(f1["E"]), A=float(f1["a"]) * n_ref ** float(f1["alpha"]),
                     alpha=float(f1["alpha"]), rss=float(f1["rss"]))
    collapsed = float(f2["b"]) == 0.0
    fast = float(f2["beta"])   # DM-1b's beta is the fast exponent
    slow = float(f2["alpha"])  # DM-1b's alpha is the slow exponent
    two = TwoTermFit(E=float(f2["E"]), A=float(f2["b"]) * n_ref ** fast, alpha=fast,
                     B=float(f2["a"]) * n_ref ** slow, beta=slow, rss=float(f2["rss"]), collapsed=collapsed)
    return one, two


def fit_one_and_two_term(N, L, *, n_ref: float = N_REF):
    """Least-squares 1-term and 2-term fits of one ladder (DM-1b's fitter): returns (OneTermFit, TwoTermFit)."""
    N, L = validate_ladder(N, L, min_points=MIN_POINTS_LRT, who="fit_one_and_two_term")
    f1, f2 = _fit_dicts(N / n_ref, L[None, :])
    f1, f2 = ({k: float(v[0]) for k, v in f.items()} for f in (f1, f2))
    if not (np.isfinite(f1["rss"]) and np.isfinite(f2["rss"])):
        raise FitError("no feasible one-term fit (E >= 0 and A >= 0 cannot describe this ladder: does the loss "
                       "increase with N?)")
    return _to_raw_units(f1, f2, n_ref)


def bootstrap_lr_test(N, L, *, B: int = B_DEFAULT, level: float = 0.05, rng=0, early_stop: bool = False,
                      chunk: int = 100, n_ref: float = N_REF) -> LRTResult:
    """Parametric-bootstrap likelihood-ratio test of a second power-law term in N (DM-1b rule).

    N, L : 1-D arrays of model sizes and final losses, one ladder, at least MIN_POINTS_LRT points (a ladder with
           fewer is not testable and raises InsufficientDataError; it is never reported as "no second term").
    B    : bootstrap refits (DM-1b used 200, folder 15 used 1000); needs B >= 1 / level - 1 to be able to detect.
    rng  : seed, SeedSequence entropy list, or numpy Generator; the first draw is rng.normal(0, 1, (B, npts)).
    early_stop : DM-1b's exact decision shortcut (stop after B_STAGE1 refits once more than k_max exceed T). The
           decision is unchanged; p_value is then computed from B_used refits and is only a rough indicator.
    The test assumes additive Gaussian noise on L with constant sd; it was calibrated on DM-1's ladders (5 to 33
    points, 1e5 to 1e9), where its pooled null rejection was 0.049 (0.055 on 600 fresh ladders in folder 17). With
    multiplicative (log-normal) noise it over-rejects when the loss range is wide: folder 17, post hoc, found null
    rejection 0.175, 0.12 and 0.05 for loss ranges of about 2 to 10, 2 to 5.5 and 2 to 3.4 at relative sd 0.005.
    """
    if not (0.0 < level < 1.0):
        raise ValueError(f"lr_test: level must be in (0, 1), got {level}")
    if int(B) != B or B < int(np.ceil(1.0 / level)) - 1:
        raise ValueError(f"lr_test: B = {B} cannot reach level {level}; need an integer B >= {int(np.ceil(1.0 / level)) - 1}")
    B = int(B)
    N, L = validate_ladder(N, L, min_points=MIN_POINTS_LRT, who="lr_test")
    n = N / n_ref
    npts = len(n)
    f1, f2 = _fit_dicts(n, L[None, :])
    f1, f2 = ({k: float(v[0]) for k, v in f.items()} for f in (f1, f2))
    if not (np.isfinite(f1["rss"]) and np.isfinite(f2["rss"])):
        raise FitError("lr_test: no feasible one-term fit (E >= 0 and A >= 0 cannot describe this ladder: does the "
                       "loss increase with N?)")
    T = float(lrt_stat(npts, f1["rss"], f2["rss"]))
    k_max = int(np.floor(level * (B + 1) + 1e-9)) - 1  # detect iff k <= k_max, i.e. (1 + k) / (B + 1) <= level
    sig = float(np.sqrt(f1["rss"] / (npts - 3)))
    mu = f1["E"] + f1["a"] * n ** -f1["alpha"]
    k, nbad, b_used = 0, 0, 0
    if T > 0:
        eps = np.random.default_rng(rng).normal(0.0, 1.0, (B, npts))
        stages = [(0, B_STAGE1), (B_STAGE1, B)] if (early_stop and B > B_STAGE1) else [(0, B)]
        tol = 1e-9 * max(T, 1.0)
        for lo, hi in stages:
            if k > k_max:
                break
            for c0 in range(lo, hi, chunk):
                c1 = min(c0 + chunk, hi)
                g1, g2 = fit_chunked(n, mu[None, :] + sig * eps[c0:c1])
                Ts = lrt_stat(npts, g1["rss"], g2["rss"])
                bad = ~np.isfinite(Ts)
                k += int(((Ts >= T - tol) | bad).sum())
                nbad += int(bad.sum())
            b_used = hi
    else:  # T = 0: every T* >= 0 reaches it, so k = B exactly and no refit is needed
        k, b_used = B, B
    one, two = _to_raw_units(f1, f2, n_ref)
    return LRTResult(T=T, p_value=(1 + k) / (b_used + 1), k_exceed=k, B=B, B_used=b_used, level=level,
                     detect=bool(b_used == B and k <= k_max), early_stopped=bool(b_used < B), n_points=npts,
                     sigma_hat=sig, n_nonfinite_boot=nbad, one_term=one, two_term=two)
