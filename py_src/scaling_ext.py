"""
Robust ADDITIVE power-law fitters for the "dark matter" experiments.

Every model is additive (AGENTS.md non-negotiable rule: never replace E + A/N^a + B/D^b by a product):

    L_hat = E + sum_k c_k * x_k ** (-e_k),      x_k in {N, D}

A sub-leading ("dark matter") term j has exponent e_j = rho * e_main with rho in (0, 1), so the
two-term N-only conjecture  L(N) = E + A/N^alpha + B/N^beta  (beta < alpha)  is the spec
[N main, N sub-leading of term 0].

Objective: Huber(delta) of the log-residual r = log(L_hat) - log(L) (delta = 1e-3, same convention as
py_src/scaling_laws_fit.py) or the plain sum of squared log-residuals ('ls').

Fitting = random exponent sampling + non-negative least squares for the linear coefficients
(variable projection), then trust-region (scipy least_squares, method trf) refinement of the best starts, optionally warm-started from a
nested simpler fit so that a bigger model can never score worse in-sample than the model it contains.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, nnls

from .scaling_laws_fit import huber

_LOGE_BOUNDS = (-40.0, 6.0)      # log E
_LOGC_BOUNDS = (-40.0, 40.0)     # log c_k
_LOGEXP_BOUNDS = (np.log(0.005), np.log(4.0))
_LOGIT_BOUNDS = (-7.0, 7.0)


@dataclass(frozen=True)
class Term:
    var: str                         # 'N' or 'D'
    sub_of: Optional[int] = None     # index of the main term this term is sub-leading to
    ratio: Optional[float] = None    # fixed e_k / e_main; None = free in (0, 1)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    terms: Tuple[Term, ...]

    @property
    def n_main(self) -> int:
        return sum(t.sub_of is None for t in self.terms)

    @property
    def n_free_ratio(self) -> int:
        return sum(t.sub_of is not None and t.ratio is None for t in self.terms)

    @property
    def n_params(self) -> int:
        return 1 + len(self.terms) + self.n_main + self.n_free_ratio


@dataclass
class FitResult:
    spec: ModelSpec
    E: float
    c: np.ndarray
    e: np.ndarray
    objective: float
    rss: float
    n: int
    ok: bool


# ---- standard specs ----------------------------------------------------------------------------
ONE_N = ModelSpec("1-term N", (Term("N"),))
TWO_N = ModelSpec("2-term N (beta<alpha)", (Term("N"), Term("N", sub_of=0)))
TWO_N_HALF = ModelSpec("2-term N (beta=alpha/2)", (Term("N"), Term("N", sub_of=0, ratio=0.5)))
JOINT = ModelSpec("joint E+A/N^a+B/D^b", (Term("N"), Term("D")))
JOINT_N_DARK = ModelSpec("joint + N dark term", (Term("N"), Term("D"), Term("N", sub_of=0)))
JOINT_D_DARK = ModelSpec("joint + D dark term", (Term("N"), Term("D"), Term("D", sub_of=1)))
JOINT_BOTH_DARK = ModelSpec(
    "joint + N and D dark terms",
    (Term("N"), Term("D"), Term("N", sub_of=0), Term("D", sub_of=1)),
)


def _x(var, N, D):
    return N if var == "N" else D


def _exponents(spec: ModelSpec, phi: np.ndarray) -> np.ndarray:
    """phi = [main exponents in term order..., free ratios in term order...]."""
    e = np.zeros(len(spec.terms))
    i = 0
    for k, t in enumerate(spec.terms):
        if t.sub_of is None:
            e[k] = phi[i]
            i += 1
    for k, t in enumerate(spec.terms):
        if t.sub_of is not None:
            if t.ratio is None:
                rho = phi[i]
                i += 1
            else:
                rho = t.ratio
            e[k] = rho * e[t.sub_of]
    return e


def _design(spec, e, N, D):
    cols = [np.ones_like(N)]
    for t, ek in zip(spec.terms, e):
        cols.append(_x(t.var, N, D) ** (-ek))
    return np.column_stack(cols)


def _obj_from_resid(r, objective, delta):
    return float(np.sum(huber(r, delta=delta))) if objective == "huber" else float(np.sum(r * r))


def _eval_coef(spec, coef, e, N, D, logL, objective, delta):
    Lhat = _design(spec, e, N, D) @ coef
    r = np.log(np.maximum(Lhat, 1e-300)) - logL
    return _obj_from_resid(r, objective, delta)


def predict(res: FitResult, N, D=None):
    N = np.asarray(N, float)
    D = None if D is None else np.asarray(D, float)
    out = np.full(N.shape, res.E, dtype=float)
    for t, ck, ek in zip(res.spec.terms, res.c, res.e):
        out = out + ck * _x(t.var, N, D) ** (-ek)
    return out


def log_resid(res: FitResult, N, L, D=None):
    return np.log(predict(res, N, D)) - np.log(np.asarray(L, float))


def objective_of(res: FitResult, N, L, D=None, objective="huber", delta=1e-3):
    return _obj_from_resid(log_resid(res, N, L, D), objective, delta)


def _encode(spec, E, c, phi):
    th = [np.log(max(E, 1e-6)) if E is not None else -10.0]
    th += [np.log(max(ck, 1e-12)) for ck in c]
    nm = spec.n_main
    th += [np.log(p) for p in phi[:nm]]
    th += [np.log(p / (1 - p)) for p in phi[nm:]]
    return np.clip(np.array(th, float), -39.0, 39.0)


def _decode(spec, th):
    T = len(spec.terms)
    nm = spec.n_main
    E = np.exp(th[0])
    c = np.exp(th[1:1 + T])
    em = np.exp(th[1 + T:1 + T + nm])
    rho = 1.0 / (1.0 + np.exp(-th[1 + T + nm:]))
    return E, c, np.concatenate([em, rho])


def _bounds(spec, exp_max=4.0):
    T = len(spec.terms)
    exp_b = (_LOGEXP_BOUNDS[0], float(np.log(exp_max)))
    b = [_LOGE_BOUNDS] + [_LOGC_BOUNDS] * T + [exp_b] * spec.n_main + [_LOGIT_BOUNDS] * spec.n_free_ratio
    return b


def _checked_inputs(spec, N, L, D):
    """Float arrays, with the positivity and 'spec uses D' checks that fit() raises ValueError for."""
    N = np.asarray(N, float)
    L = np.asarray(L, float)
    D = None if D is None else np.asarray(D, float)
    if (N <= 0).any() or (L <= 0).any() or (D is not None and (D <= 0).any()):
        raise ValueError("N, D, L must be strictly positive")
    if any(t.var == "D" for t in spec.terms) and D is None:
        raise ValueError("spec uses D but D is None")
    return N, L, D


def _candidate(spec, phi, N, D, L, logL, objective, delta):
    """(objective value, phi, coefficients) at exponents phi, with the coefficients from non-negative least squares."""
    e = _exponents(spec, phi)
    coef, _ = nnls(_design(spec, e, N, D) / L[:, None], np.ones_like(L))
    return (_eval_coef(spec, coef, e, N, D, logL, objective, delta), phi, coef)


def _sample_starts(spec, N, D, L, logL, rng, n_starts, e_range, fix_main, objective, delta):
    """n_starts random exponent vectors (frozen exponents fixed), each with its NNLS coefficients, sorted best first."""
    cands = []
    for _ in range(n_starts):
        em = np.exp(rng.uniform(np.log(e_range[0]), np.log(e_range[1]), spec.n_main))
        for i_fix, v_fix in fix_main.items():
            em[i_fix] = v_fix
        rho = rng.uniform(0.05, 0.95, spec.n_free_ratio)
        cands.append(_candidate(spec, np.concatenate([em, rho]), N, D, L, logL, objective, delta))
    cands.sort(key=lambda z: z[0])
    return cands


def _nested_start(spec, warm, L):
    """The exact start that reproduces `warm` inside the bigger `spec`: (exponent vector, [E, coefficients], mask of the new
    exponent slots). New coefficients start at 1e-6 of the median loss."""
    n_warm = len(warm.spec.terms)
    main_idx = [k for k, t in enumerate(spec.terms) if t.sub_of is None]
    em = [warm.e[k] if k < n_warm else 0.5 for k in main_idx]
    em_new = [k >= n_warm for k in main_idx]
    rho, rho_new = [], []
    for k, t in enumerate(spec.terms):
        if t.sub_of is not None and t.ratio is None:
            rho.append(float(warm.e[k] / warm.e[t.sub_of]) if k < n_warm else 0.5)
            rho_new.append(k >= n_warm)
    scale = max(float(np.median(L)), 1e-3)
    c0 = np.concatenate([warm.c, np.full(len(spec.terms) - n_warm, 1e-6 * scale)])
    return np.array(em + rho, float), np.concatenate([[warm.E], c0]), np.array(em_new + rho_new, bool)


def _lift_new_coefficients(spec, warm, coef, phi, N, D, L):
    """A new coefficient that starts at ~0 sits on a flat plateau of the log-parametrisation (zero gradient), so every new
    term is lifted to a small but real contribution (5 % of its natural scale) before refinement."""
    coef = coef.copy()
    e = _exponents(spec, phi)
    for k in range(len(warm.spec.terms), len(spec.terms)):
        t = spec.terms[k]
        x_med = float(np.median(_x(t.var, N, D)))
        if t.sub_of is not None:
            base = max(coef[1 + t.sub_of], 1e-12) * x_med ** (-e[t.sub_of])
        else:
            base = max(float(np.median(L)) - warm.E, 1e-6)
        coef[1 + k] = max(coef[1 + k], 0.05 * base * x_med ** e[k])
    return coef


def _new_slot_seeds(spec, warm, phi_w, is_new, N, D, L, logL, rng, n_starts, e_range, objective, delta):
    """Extra seeds that keep the warm exponents and scan only the NEW exponent slots; the best two, coefficients lifted."""
    seeds = []
    for _ in range(max(40, n_starts // 3)):
        phi = phi_w.copy()
        for j in np.where(is_new)[0]:
            phi[j] = (np.exp(rng.uniform(np.log(e_range[0]), np.log(e_range[1]))) if j < spec.n_main
                      else rng.uniform(0.05, 0.95))
        seeds.append(_candidate(spec, phi, N, D, L, logL, objective, delta))
    seeds.sort(key=lambda z: z[0])
    return [(phi, _lift_new_coefficients(spec, warm, coef, phi, N, D, L)) for _, phi, coef in seeds[:2]]


def _frozen_bounds(spec, exp_max, fix_main):
    """Lower and upper bound vectors in the encoded parametrisation; frozen main exponents get a +-1e-9 window."""
    bnds = _bounds(spec, exp_max)
    for i_fix, v_fix in fix_main.items():
        j = 1 + len(spec.terms) + i_fix
        bnds[j] = (float(np.log(v_fix)) - 1e-9, float(np.log(v_fix)) + 1e-9)
    return np.array([b[0] for b in bnds]), np.array([b[1] for b in bnds])


def _make_resid(spec, logN, logD, logL):
    """Log-sum-exp residual of the additive model as a function of the encoded parameter vector."""
    def resid(th):
        E, c, phi = _decode(spec, th)
        e = _exponents(spec, phi)
        terms = [np.full_like(logN, th[0])]
        for k, t in enumerate(spec.terms):
            terms.append(th[1 + k] - e[k] * (logN if t.var == "N" else logD))
        M = np.vstack(terms)
        m = M.max(axis=0)
        return m + np.log(np.exp(M - m).sum(axis=0)) - logL
    return resid


def _refine(spec, starts, resid, lb, ub, ls_kwargs, objective, delta, max_nfev):
    """Trust-region least squares from every start; returns (best value, best encoded parameters, solver success)."""
    best_val, best_th, ok = np.inf, None, False
    for phi, coef in starts:
        th0 = np.clip(_encode(spec, coef[0], coef[1:], phi), lb + 1e-9, ub - 1e-9)
        try:
            res = least_squares(resid, th0, bounds=(lb, ub), method="trf",
                                xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=max_nfev, **ls_kwargs)
        except Exception:
            continue
        val = _obj_from_resid(resid(res.x), objective, delta)
        if val < best_val:
            best_val, best_th, ok = val, res.x, bool(res.success)
    return best_val, best_th, ok


def fit(
    spec: ModelSpec,
    N,
    L,
    D=None,
    objective: str = "huber",
    delta: float = 1e-3,
    n_starts: int = 200,
    n_refine: int = 4,
    seed: int = 0,
    e_range: Tuple[float, float] = (0.03, 1.5),
    warm: Optional[FitResult] = None,
    fix_main: Optional[dict] = None,
    max_nfev: int = 4000,
    exp_max: float = 4.0,
) -> FitResult:
    """fix_main: {index among the MAIN exponents: value} freezes those exponents (used for profile scans).

    max_nfev caps residual evaluations per refinement. The default is thorough; Monte-Carlo studies on tiny
    ladders pass a smaller cap because degenerate fits crawl along an almost flat valley.

    exp_max caps every MAIN exponent (a dark term's exponent is rho * main, so it is capped too). Unconstrained fits
    (the default, 4.0) can use a very steep term with a huge coefficient as a localised small-N/small-D correction;
    a cap of 1.5 asks for smooth power laws only. See diagnose() for boundary flags."""
    fix_main = dict(fix_main or {})
    if fix_main:
        warm = None
    N, L, D = _checked_inputs(spec, N, L, D)
    logL = np.log(L)
    rng = np.random.default_rng(seed)
    e_range = (e_range[0], min(e_range[1], exp_max))

    # 1) random exponents + NNLS for linear coefficients, best first
    cands = _sample_starts(spec, N, D, L, logL, rng, n_starts, e_range, fix_main, objective, delta)

    # 2) the best random starts, plus (optionally) a nested warm start from the prefix model and seeds around it
    starts = [(phi, coef) for _, phi, coef in cands[:n_refine]]
    n_warm = 0 if warm is None else len(warm.spec.terms)
    if warm is not None and n_warm < len(spec.terms) and spec.terms[:n_warm] == warm.spec.terms:
        phi_w, coef_w, is_new = _nested_start(spec, warm, L)
        starts.insert(0, (phi_w, coef_w))   # exact nested start: can never be worse
        if is_new.any():
            starts.extend(_new_slot_seeds(spec, warm, phi_w, is_new, N, D, L, logL, rng, n_starts, e_range, objective, delta))

    # 3) trust-region refinement of every start; scipy's robust 'huber' loss with f_scale=delta minimises exactly
    # sum(huber(r, delta)), and a trust-region solver copes with the near-L1 kink at delta=1e-3 far better than L-BFGS-B.
    lb, ub = _frozen_bounds(spec, exp_max, fix_main)
    ls_kwargs = dict(loss="huber", f_scale=delta) if objective == "huber" else dict(loss="linear")
    resid = _make_resid(spec, np.log(N), None if D is None else np.log(D), logL)
    best_val, best_th, ok = _refine(spec, starts, resid, lb, ub, ls_kwargs, objective, delta, max_nfev)
    # also compare with the best raw grid point (guards against a bad refine)
    gval, gphi, gcoef = cands[0]
    if gval < best_val:
        best_val, best_th, ok = gval, _encode(spec, gcoef[0], gcoef[1:], gphi), True
    E, c, phi = _decode(spec, best_th)
    out = FitResult(spec, float(E), np.asarray(c, float), _exponents(spec, phi), best_val, 0.0, len(L), ok)
    r = log_resid(out, N, L, D)
    out.rss = float(np.sum(r * r))
    out.objective = _obj_from_resid(r, objective, delta)
    return out


def diagnose(res: FitResult, L=None, exp_max: float = 4.0) -> str:
    """Comma-separated boundary flags for a fit ('' = interior solution).

    c_hi    a coefficient is at its upper bound (e^40): the term is a localised steep correction, not a power law
    e_hi    a main exponent is at the exponent cap
    e_lo    an exponent is at the 0.005 floor
    E~0     the irreducible loss collapsed to ~0 (less than 0.1 % of the smallest observed loss, or 1e-3 if L is None)
    rho_edge a dark exponent is <0.2 % or >99.8 % of its main exponent
    """
    flags = []
    if np.any(res.c >= np.exp(39.5)):
        flags.append("c_hi")
    main_e = [ek for t, ek in zip(res.spec.terms, res.e) if t.sub_of is None]
    if np.any(np.array(main_e) >= 0.999 * exp_max):
        flags.append("e_hi")
    if np.any(res.e <= 0.005 * 1.001):
        flags.append("e_lo")
    floor = 1e-3 * (float(np.min(L)) if L is not None else 1.0)
    if res.E <= floor:
        flags.append("E~0")
    for t, ek in zip(res.spec.terms, res.e):
        if t.sub_of is not None and t.ratio is None:
            rho = ek / res.e[t.sub_of]
            if rho < 0.002 or rho > 0.998:
                flags.append("rho_edge")
                break
    return ",".join(flags)


def describe(res: FitResult) -> str:
    parts = [f"E={res.E:.4g}"]
    for t, ck, ek in zip(res.spec.terms, res.c, res.e):
        parts.append(f"{ck:.4g}/{t.var}^{ek:.4g}")
    return " + ".join(parts)
