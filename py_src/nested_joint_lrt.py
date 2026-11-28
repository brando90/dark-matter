"""Weighted nested likelihood-ratio test: additive joint law against the joint law plus one fast N term.

Models, on n = N / 1e6 and d = D / 1e9 (additive Chinchilla form, never multiplicative):
    M1 :  y = E + A n^-alpha + B d^-beta                                 E, A, B >= 0
    M2 :  y = E + A n^-alpha + B d^-beta + C n^-gamma                    E, A, B, C >= 0, gamma >= alpha + GAP
The extra term in M2 decays faster in N than the main term (gamma > alpha: the FAST term, the one that matters at small N),
so M1 is nested in M2 (C = 0).

Estimator. Weighted least squares with weights w = 1 / sd^2 (relative per-point sd, supplied by the caller; only the
ratios between points matter because the statistic is scale free). For fixed exponents the coefficients are a
non-negative linear least-squares problem (variable projection). The exponents are searched on a grid with batched normal
equations, then the best distinct grid cells are polished by Nelder-Mead on the exact non-negative least-squares RSS.

Statistic T = npts * ln(RSS_1 / RSS_2) with RSS_2 = min(RSS_2 fitted, RSS_1) so that T >= 0 (nested). Calibrated by a
parametric bootstrap under the fitted M1: y* = yhat_1 + s / sqrt(w) * z with s = sqrt(RSS_1 / (npts - 5)) and z ~ N(0, 1),
each y* refitted with both models by the same procedure. p = (1 + #{T* >= T}) / (B + 1); detect iff p <= level.
The test assumes independent noise across points. Points that are the same training run read at different token counts are
correlated in reality; experiments/19_*/ reports a stress control for that.

Fit routines raise InsufficientDataError when the design has too few points or too few distinct sizes or token counts;
they never substitute other data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, nnls

from py_src.scaling_laws_fit import InsufficientDataError

GAP = 0.15
ALPHA_GRID = np.geomspace(0.05, 1.5, 40)
BETA_GRID = np.geomspace(0.05, 1.0, 32)
GAMMA_GRID = np.geomspace(0.3, 4.0, 20)
ALPHA_BOUNDS, BETA_BOUNDS, GAMMA_BOUNDS = (0.02, 2.0), (0.02, 1.5), (0.2, 6.0)
N_PARAMS_M1 = 5   # E, A, alpha, B, beta (M2 adds C and gamma)
MIN_POINTS, MIN_DISTINCT = 12, 5
INFEASIBLE = 1e15
EXCLUDE = 3                       # grid half-width excluded around a chosen cell when picking distinct start cells
B_DEFAULT = 200
LEVEL = 0.05


@dataclass(frozen=True)
class Fit:
    """One fitted model. rss is the weighted residual sum of squares at the polished exponents."""
    rss: float
    coef: tuple           # (E, A, B) for M1, (E, A, B, C) for M2
    exps: tuple           # (alpha, beta) for M1, (alpha, beta, gamma) for M2
    fitted: np.ndarray


class Design:
    """Fixed design (sizes, tokens, per-point sd): precomputes everything that does not depend on the response."""

    def __init__(self, N, D, sd):
        N, D, sd = (np.asarray(x, float) for x in (N, D, sd))
        if not (N.shape == D.shape == sd.shape and N.ndim == 1):
            raise ValueError("N, D and sd must be 1-d arrays of equal length")
        if len(N) < MIN_POINTS:
            raise InsufficientDataError(f"{len(N)} points; need at least {MIN_POINTS}")
        if len(np.unique(N)) < MIN_DISTINCT or len(np.unique(D)) < MIN_DISTINCT:
            raise InsufficientDataError("too few distinct sizes or token counts for five parameters")
        if not (np.all(N > 0) and np.all(D > 0) and np.all(sd > 0)):
            raise ValueError("N, D and sd must be positive")
        self.n, self.d, self.w = N / 1e6, D / 1e9, 1.0 / sd ** 2
        self.npts = len(N)
        self._sw = np.sqrt(self.w)
        self._build_grids()

    def _build_grids(self):
        n, d, w = self.n, self.d, self.w
        a = n[None, :] ** -ALPHA_GRID[:, None]
        b = d[None, :] ** -BETA_GRID[:, None]
        c = n[None, :] ** -GAMMA_GRID[:, None]
        self._a, self._b, self._c = a, b, c
        g11, g1a, g1b, g1c = w.sum(), a @ w, b @ w, c @ w
        gaa, gbb, gcc = (a * a) @ w, (b * b) @ w, (c * c) @ w
        gab, gac, gbc = (a * w) @ b.T, (a * w) @ c.T, (b * w) @ c.T
        nA, nB, nG = len(ALPHA_GRID), len(BETA_GRID), len(GAMMA_GRID)
        G1 = np.empty((nA, nB, 3, 3))
        G1[..., 0, 0] = g11
        G1[..., 0, 1] = G1[..., 1, 0] = g1a[:, None]
        G1[..., 0, 2] = G1[..., 2, 0] = g1b[None, :]
        G1[..., 1, 1] = gaa[:, None]
        G1[..., 1, 2] = G1[..., 2, 1] = gab
        G1[..., 2, 2] = gbb[None, :]
        G2 = np.empty((nA, nB, nG, 4, 4))
        G2[..., 0, 0] = g11
        G2[..., 0, 1] = G2[..., 1, 0] = g1a[:, None, None]
        G2[..., 0, 2] = G2[..., 2, 0] = g1b[None, :, None]
        G2[..., 0, 3] = G2[..., 3, 0] = g1c[None, None, :]
        G2[..., 1, 1] = gaa[:, None, None]
        G2[..., 1, 2] = G2[..., 2, 1] = gab[:, :, None]
        G2[..., 1, 3] = G2[..., 3, 1] = gac[:, None, :]
        G2[..., 2, 2] = gbb[None, :, None]
        G2[..., 2, 3] = G2[..., 3, 2] = gbc[None, :, :]
        G2[..., 3, 3] = gcc[None, None, :]
        self._ginv1, self._ginv2 = _scaled_inverse(G1), _scaled_inverse(G2)
        self._ok2 = GAMMA_GRID[None, None, :] >= ALPHA_GRID[:, None, None] + GAP

    # ---- grid search ----------------------------------------------------------------------------------------------
    def _grid_rss(self, y, two_terms):
        """Weighted RSS of the unconstrained least-squares fit on every grid cell; infeasible cells (a negative
        coefficient, or gamma < alpha + GAP) get INFEASIBLE added so that they rank last but stay orderable."""
        wy = self.w * y
        yy = float(wy @ y)
        r0, ra, rb = float(wy.sum()), self._a @ wy, self._b @ wy
        nA, nB = len(ALPHA_GRID), len(BETA_GRID)
        if not two_terms:
            r = np.empty((nA, nB, 3))
            r[..., 0], r[..., 1], r[..., 2] = r0, ra[:, None], rb[None, :]
            coef = np.einsum("abij,abj->abi", self._ginv1, r)
            rss = yy - np.einsum("abi,abi->ab", coef, r)
            bad = (coef < 0).any(-1)
        else:
            nG = len(GAMMA_GRID)
            rc = self._c @ wy
            r = np.empty((nA, nB, nG, 4))
            r[..., 0], r[..., 1] = r0, ra[:, None, None]
            r[..., 2], r[..., 3] = rb[None, :, None], rc[None, None, :]
            coef = np.einsum("abgij,abgj->abgi", self._ginv2, r)
            rss = yy - np.einsum("abgi,abgi->abg", coef, r)
            bad = (coef < 0).any(-1) | ~self._ok2
        return np.where(bad, rss + INFEASIBLE, rss)

    @staticmethod
    def _distinct_cells(rss, k):
        """Indices of the k best cells, each outside an EXCLUDE-step box around the cells already chosen."""
        work, picks = rss.copy(), []
        for _ in range(k):
            idx = np.unravel_index(int(np.argmin(work)), work.shape)
            if not np.isfinite(work[idx]) or work[idx] >= 1e300:
                break
            picks.append(idx)
            box = tuple(slice(max(i - EXCLUDE, 0), i + EXCLUDE + 1) for i in idx)
            work[box] = np.inf
        return picks

    # ---- exact evaluation and polishing ---------------------------------------------------------------------------
    def _columns(self, exps):
        cols = [np.ones(self.npts), self.n ** -exps[0], self.d ** -exps[1]]
        if len(exps) == 3:
            cols.append(self.n ** -exps[2])
        return np.column_stack(cols)

    def _nnls(self, exps, y):
        X = self._columns(exps)
        coef, rnorm = nnls(X * self._sw[:, None], y * self._sw, maxiter=200)
        return coef, float(rnorm) ** 2, X

    def _objective(self, theta, y):
        if len(theta) == 3 and theta[2] < theta[0] + GAP:
            return 1e30 + (theta[0] + GAP - theta[2])
        lo_hi = (ALPHA_BOUNDS, BETA_BOUNDS, GAMMA_BOUNDS)[:len(theta)]
        for t, (lo, hi) in zip(theta, lo_hi):
            if not (lo <= t <= hi):
                return 1e30 + abs(t - min(max(t, lo), hi))
        return self._nnls(theta, y)[1]

    def _polish(self, start, y):
        res = minimize(self._objective, np.asarray(start, float), args=(y,), method="Nelder-Mead",
                       options=dict(xatol=1e-5, fatol=1e-12, maxiter=600, maxfev=900))
        coef, rss, X = self._nnls(res.x, y)
        return Fit(rss=rss, coef=tuple(coef), exps=tuple(float(t) for t in res.x), fitted=X @ coef)

    def fit(self, y, two_terms, n_starts=None):
        """Best polished fit of M1 (two_terms False) or M2 (True) over the best distinct grid cells."""
        y = np.asarray(y, float)
        k = n_starts or (3 if two_terms else 2)
        cells = self._distinct_cells(self._grid_rss(y, two_terms), k)
        if not cells:
            raise InsufficientDataError("no finite grid cell: the response is not finite or the design is degenerate")
        grids = (ALPHA_GRID, BETA_GRID, GAMMA_GRID)
        fits = [self._polish([g[i] for g, i in zip(grids, cell)], y) for cell in cells]
        return min(fits, key=lambda f: f.rss)

    def statistic(self, y):
        """(T, fit1, fit2) with RSS_2 clipped at RSS_1 so that T >= 0."""
        f1 = self.fit(y, False)
        f2 = self.fit(y, True)
        rss2 = min(f2.rss, f1.rss)
        return self.npts * float(np.log(f1.rss / rss2)), f1, f2

    def bootstrap_lrt(self, y, B=B_DEFAULT, rng=None, level=LEVEL):
        """Parametric-bootstrap p-value of M2 against M1. Returns dict(T, p, detect, B, T_boot, sigma_hat, fit1, fit2)."""
        rng = np.random.default_rng() if rng is None else rng
        T, f1, f2 = self.statistic(y)
        sigma = float(np.sqrt(f1.rss / (self.npts - N_PARAMS_M1)))
        scale = sigma / self._sw
        T_boot = np.empty(B)
        for i in range(B):
            T_boot[i] = self.statistic(f1.fitted + scale * rng.standard_normal(self.npts))[0]
        p = (1 + int((T_boot >= T - 1e-12).sum())) / (B + 1)
        return dict(T=T, p=p, detect=bool(p <= level), B=B, T_boot=T_boot, sigma_hat=sigma, fit1=f1, fit2=f2)


def _scaled_inverse(G):
    """Batched inverse of symmetric positive semi-definite Gram matrices, with unit-diagonal scaling and a tiny ridge."""
    s = 1.0 / np.sqrt(np.einsum("...ii->...i", G))
    Gs = G * s[..., :, None] * s[..., None, :]
    Gs = Gs + 1e-10 * np.eye(G.shape[-1])
    return np.linalg.inv(Gs) * s[..., :, None] * s[..., None, :]
