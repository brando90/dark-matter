"""Multi-start Huber fits for scaling-law conjectures.

The objective matches Hoffmann approach 3 / the Besiroglu replication:
sum of Huber penalties on log residuals, delta=1e-3, minimized with L-BFGS-B.

Positive-term models are evaluated with log-sum-exp so the prediction used
for the objective is the same function as `predict`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import OptimizeWarning, minimize
from scipy.special import expit, logsumexp
import warnings

from py_src.scaling_laws_fit import huber

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PENALTY = 1e12
N_PARAMS = {
    "single_term": 3,
    "two_term": 5,
    "log_correction": 4,
    "broken_power": 5,
    "d_only": 3,
    "compute_one": 3,
    "compute_two": 5,
    "joint": 5,
    "joint_extra_n": 7,
    "joint_extra_d": 7,
}


@dataclass
class FitResult:
    kind: str
    params: dict
    objective: float
    mean_huber: float
    success: bool
    n_starts: int
    n_finite: int
    n_success_flag: int
    rel_gap_second: float
    message: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return (not np.isfinite(self.objective)) or self.objective >= PENALTY / 10


def _softplus(x: float) -> float:
    x = float(x)
    if x > 20:
        return x
    return float(np.log1p(np.exp(x)))


def _inv_softplus(a: float) -> float:
    a = float(np.clip(a, 1e-6, 20.0))
    return float(np.log(np.expm1(a)))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-4, 1.0 - 1e-4))
    return float(np.log(p / (1.0 - p)))


def _as64(*arrays):
    return [np.asarray(a, dtype=np.float64) for a in arrays]


def _huber_from_log_terms(log_terms: np.ndarray, log_y: np.ndarray, delta: float) -> float:
    residual = logsumexp(log_terms, axis=0) - log_y
    if not np.all(np.isfinite(residual)):
        return PENALTY
    return float(np.sum(huber(residual, delta=delta)))


def _predict_sum(log_terms: np.ndarray) -> np.ndarray:
    return np.exp(logsumexp(np.asarray(log_terms, dtype=np.float64), axis=0))


def predict(kind: str, params: dict, N=None, D=None, C=None) -> np.ndarray:
    """Predict loss. N-models read `N` (or `C` if N is None). Joint models need N and D."""
    if kind in ("compute_one", "compute_two"):
        x = np.asarray(C if C is not None else N, dtype=np.float64)
        mapped = "single_term" if kind == "compute_one" else "two_term"
        return _predict_power(mapped, params, x)
    if kind in ("single_term", "two_term", "log_correction", "broken_power"):
        x = np.asarray(N if N is not None else C, dtype=np.float64)
        return _predict_power(kind, params, x)
    if kind == "d_only":
        return _predict_power("single_term", params, np.asarray(D, dtype=np.float64))
    if kind == "joint":
        return _predict_joint(params, N, D, extra=False)
    if kind == "joint_extra_n":
        return _predict_joint(params, N, D, extra=True)
    if kind == "joint_extra_d":
        return _predict_joint_extra_d(params, N, D)
    raise KeyError(kind)


def _predict_power(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    logx = np.log(x)
    if kind == "log_correction":
        z = logx - np.log(params["n_ref"])
        return params["E"] + params["A"] * np.exp(-params["alpha"] * logx) * (1.0 + params["gamma"] * z)
    terms = [_power_log_terms(kind, params, logx)]
    return _predict_sum(terms[0])


def _power_log_terms(kind: str, params: dict, logx: np.ndarray) -> np.ndarray:
    e = np.full_like(logx, np.log(params["E"]))
    if kind == "single_term":
        t = np.log(params["A"]) - params["alpha"] * logx
        return np.vstack([t, e])
    if kind == "two_term":
        t1 = np.log(params["A"]) - params["alpha"] * logx
        t2 = np.log(params["B"]) - params["beta"] * logx
        return np.vstack([t1, t2, e])
    if kind == "broken_power":
        # log( A * x^{-alpha} / (1 + (x/n_break)^delta) )
        bend = np.logaddexp(0.0, params["delta"] * (logx - np.log(params["n_break"])))
        t = np.log(params["A"]) - params["alpha"] * logx - bend
        return np.vstack([t, e])
    raise KeyError(kind)


def _predict_joint(params: dict, N, D, extra: bool) -> np.ndarray:
    n, d = _as64(N, D)
    logn, logd = np.log(n), np.log(d)
    e = np.full_like(logn, np.log(params["E"]))
    t_a = np.log(params["A"]) - params["alpha"] * logn
    t_d = np.log(params["G"]) - params["gamma"] * logd
    if extra:
        t_b = np.log(params["B"]) - params["beta"] * logn
        return _predict_sum(np.vstack([t_a, t_b, t_d, e]))
    return _predict_sum(np.vstack([t_a, t_d, e]))


def _predict_joint_extra_d(params: dict, N, D) -> np.ndarray:
    """L = E + A N^{-alpha} + G D^{-gamma} + H D^{-delta}, delta < gamma.

    gamma is the fast token exponent. delta is the slow one.
    """
    n, d = _as64(N, D)
    logn, logd = np.log(n), np.log(d)
    e = np.full_like(logn, np.log(params["E"]))
    t_a = np.log(params["A"]) - params["alpha"] * logn
    t_fast = np.log(params["G"]) - params["gamma"] * logd
    t_slow = np.log(params["H"]) - params["delta"] * logd
    return _predict_sum(np.vstack([t_a, t_fast, t_slow, e]))


def _pack_power(kind: str, theta: np.ndarray, logx_train: np.ndarray):
    """Map unconstrained theta to params. log_correction uses the training log x."""
    e = float(np.exp(theta[0]))
    a = float(np.exp(theta[1]))
    alpha = _softplus(theta[2]) + 1e-8
    if kind == "single_term":
        return {"E": e, "A": a, "alpha": alpha}
    if kind == "two_term":
        frac = float(expit(theta[4]))
        beta = max(1e-8, alpha * frac)
        return {"E": e, "A": a, "alpha": alpha, "B": float(np.exp(theta[3])), "beta": beta}
    if kind == "log_correction":
        z = logx_train - np.mean(logx_train)
        max_abs = float(np.max(np.abs(z))) if len(z) else 1.0
        gamma = (0.8 / max(max_abs, 1e-6)) * float(np.tanh(theta[3]))
        return {
            "E": e,
            "A": a,
            "alpha": alpha,
            "gamma": gamma,
            "n_ref": float(np.exp(np.mean(logx_train))),
        }
    if kind == "broken_power":
        return {
            "E": e,
            "A": a,
            "alpha": alpha,
            "n_break": float(np.exp(theta[3])),
            "delta": _softplus(theta[4]) + 1e-8,
        }
    raise KeyError(kind)


def _objective_power(kind: str, theta: np.ndarray, logx: np.ndarray, logy: np.ndarray, delta: float) -> float:
    try:
        params = _pack_power(kind, theta, logx)
        if kind == "log_correction":
            z = logx - np.log(params["n_ref"])
            factor = 1.0 + params["gamma"] * z
            if np.any(factor <= 1e-4) or params["E"] <= 0 or params["A"] <= 0:
                return PENALTY
            log_term = np.log(params["A"]) - params["alpha"] * logx + np.log(factor)
            e = np.full_like(logx, np.log(params["E"]))
            # E + positive term. If the correction stays positive this is exact.
            return _huber_from_log_terms(np.vstack([log_term, e]), logy, delta)
        return _huber_from_log_terms(_power_log_terms(kind, params, logx), logy, delta)
    except Exception:
        return PENALTY


def _starts_power(kind: str, x: np.ndarray, y: np.ndarray, n_starts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    logx = np.log(x)
    x_med = float(np.exp(np.median(logx)))
    y_min = float(np.min(y))
    y_med = float(np.median(y))
    starts: list[np.ndarray] = []

    def one(e_frac, alpha, b_share=0.5, frac=0.45, break_mult=1.0, delta=0.8, gamma_raw=0.0):
        e = max(e_frac * y_min, 1e-8)
        gap = max(y_med - e, 1e-4)
        aa = max(gap * (x_med ** alpha), 1e-8)
        if kind == "single_term":
            return np.array([np.log(e), np.log(aa), _inv_softplus(alpha)], dtype=np.float64)
        if kind == "two_term":
            bb = max(b_share * gap * (x_med ** (alpha * frac)), 1e-8)
            return np.array(
                [np.log(e), np.log(aa), _inv_softplus(alpha), np.log(bb), _logit(frac)],
                dtype=np.float64,
            )
        if kind == "log_correction":
            return np.array([np.log(e), np.log(aa), _inv_softplus(alpha), gamma_raw], dtype=np.float64)
        if kind == "broken_power":
            return np.array(
                [np.log(e), np.log(aa), _inv_softplus(alpha), np.log(x_med * break_mult), _inv_softplus(delta)],
                dtype=np.float64,
            )
        raise KeyError(kind)

    grid_e = (0.55, 0.8)
    grid_a = (0.2, 0.35, 0.55)
    for e_frac in grid_e:
        for alpha in grid_a:
            starts.append(one(e_frac, alpha))
            if kind == "two_term":
                starts.append(one(e_frac, alpha, b_share=0.25, frac=0.35))
                starts.append(one(e_frac, alpha, b_share=1.0, frac=0.7))
            elif kind == "log_correction":
                starts.append(one(e_frac, alpha, gamma_raw=-0.8))
                starts.append(one(e_frac, alpha, gamma_raw=0.8))
            elif kind == "broken_power":
                starts.append(one(e_frac, alpha, break_mult=0.3, delta=0.4))
                starts.append(one(e_frac, alpha, break_mult=3.0, delta=1.5))
    # Deterministic prefix, then seeded random starts if the caller asked for more.
    if len(starts) < n_starts:
        for _ in range(n_starts - len(starts)):
            starts.append(
                one(
                    float(rng.uniform(0.3, 0.95)),
                    float(rng.uniform(0.08, 0.8)),
                    b_share=float(rng.uniform(0.1, 1.5)),
                    frac=float(rng.uniform(0.15, 0.85)),
                    break_mult=float(rng.uniform(0.2, 5.0)),
                    delta=float(rng.uniform(0.2, 2.0)),
                    gamma_raw=float(rng.uniform(-1.2, 1.2)),
                )
            )
    return starts[:n_starts]


def _diagnostics_power(kind: str, params: dict, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    info = {"collapsed_to_single": False}
    if kind == "two_term":
        a_term = params["A"] * x ** (-params["alpha"])
        b_term = params["B"] * x ** (-params["beta"])
        frac = b_term / np.maximum(a_term + b_term, 1e-30)
        info.update(
            {
                "beta_over_alpha": params["beta"] / params["alpha"],
                "median_second_fraction": float(np.median(frac)),
                "second_fraction_min": float(np.min(frac)),
                "second_fraction_max": float(np.max(frac)),
            }
        )
        a_range = (np.max(a_term) - np.min(a_term)) / max(float(np.median(a_term)), 1e-12)
        b_range = (np.max(b_term) - np.min(b_term)) / max(float(np.median(b_term)), 1e-12)
        # Either term dying, or the two exponents collapsing together, means
        # the fit is a single power plus a floor.
        info["collapsed_to_single"] = bool(
            info["median_second_fraction"] < 0.02
            or info["median_second_fraction"] > 0.98
            or info["beta_over_alpha"] > 0.98
            or (b_range < 0.05 and a_range > 0.1)
        )
    elif kind == "log_correction":
        info["gamma"] = params["gamma"]
        info["collapsed_to_single"] = bool(abs(params["gamma"]) < 1e-4)
    elif kind == "broken_power":
        info["n_break"] = params["n_break"]
        info["delta"] = params["delta"]
        info["collapsed_to_single"] = bool(params["n_break"] > 100 * np.max(x) or params["n_break"] < np.min(x) / 100)
    return info


def fit_power(kind: str, x, y, delta: float = 1e-3, n_starts: int = 12, seed: int = 0, record_starts: bool = False) -> FitResult:
    x, y = _as64(x, y)
    if np.any(y <= 0) or np.any(x <= 0):
        raise ValueError("fit_power expects positive x and y")
    logx = np.log(x)
    logy = np.log(y)
    starts = _starts_power(kind, x, y, n_starts=n_starts, seed=seed)

    def objective(theta):
        return _objective_power(kind, theta, logx, logy, delta)

    return _minimize_starts(
        kind,
        objective,
        starts,
        n=len(y),
        finalize=lambda th: _pack_power(kind, th, logx),
        diagnose=lambda p: _diagnostics_power(kind, p, x),
        record_starts=record_starts,
    )


def _pack_joint(theta: np.ndarray, extra: bool) -> dict:
    e = float(np.exp(theta[0]))
    a = float(np.exp(theta[1]))
    alpha = _softplus(theta[2]) + 1e-8
    if not extra:
        beta = _softplus(theta[4]) + 1e-8
        return {"E": e, "A": a, "alpha": alpha, "G": float(np.exp(theta[3])), "gamma": beta}
    frac = float(expit(theta[4]))
    beta = max(1e-8, alpha * frac)
    return {
        "E": e,
        "A": a,
        "alpha": alpha,
        "B": float(np.exp(theta[3])),
        "beta": beta,
        "G": float(np.exp(theta[5])),
        "gamma": _softplus(theta[6]) + 1e-8,
    }


def _objective_joint(theta, logn, logd, logy, delta, extra) -> float:
    try:
        p = _pack_joint(theta, extra)
        e = np.full_like(logn, np.log(p["E"]))
        t_a = np.log(p["A"]) - p["alpha"] * logn
        t_d = np.log(p["G"]) - p["gamma"] * logd
        if extra:
            t_b = np.log(p["B"]) - p["beta"] * logn
            terms = np.vstack([t_a, t_b, t_d, e])
        else:
            terms = np.vstack([t_a, t_d, e])
        return _huber_from_log_terms(terms, logy, delta)
    except Exception:
        return PENALTY


def _starts_joint(n, d, y, extra: bool, n_starts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_med = float(np.exp(np.median(np.log(n))))
    d_med = float(np.exp(np.median(np.log(d))))
    y_min = float(np.min(y))
    y_med = float(np.median(y))
    starts = []

    def one(e_frac, alpha, gamma, frac=0.5, b_share=0.5):
        e = max(e_frac * y_min, 1e-8)
        gap = max(y_med - e, 1e-4)
        aa = max(0.5 * gap * n_med ** alpha, 1e-8)
        gg = max(0.5 * gap * d_med ** gamma, 1e-8)
        if not extra:
            return np.array([np.log(e), np.log(aa), _inv_softplus(alpha), np.log(gg), _inv_softplus(gamma)])
        bb = max(b_share * gap * n_med ** (alpha * frac), 1e-8)
        return np.array(
            [
                np.log(e),
                np.log(aa),
                _inv_softplus(alpha),
                np.log(bb),
                _logit(frac),
                np.log(gg),
                _inv_softplus(gamma),
            ]
        )

    for e_frac in (0.5, 0.75, 0.9):
        for alpha in (0.25, 0.4):
            for gamma in (0.25, 0.4):
                starts.append(one(e_frac, alpha, gamma))
                if extra:
                    starts.append(one(e_frac, alpha, gamma, frac=0.4, b_share=0.4))
    while len(starts) < n_starts:
        starts.append(
            one(
                float(rng.uniform(0.3, 0.95)),
                float(rng.uniform(0.1, 0.7)),
                float(rng.uniform(0.1, 0.7)),
                frac=float(rng.uniform(0.2, 0.8)),
                b_share=float(rng.uniform(0.15, 1.2)),
            )
        )
    return starts[:n_starts]


def fit_joint(n, d, y, extra: bool = False, delta: float = 1e-3, n_starts: int = 12, seed: int = 0,
              extra_starts: list | None = None, record_starts: bool = False) -> FitResult:
    n, d, y = _as64(n, d, y)
    if np.any(y <= 0) or np.any(n <= 0) or np.any(d <= 0):
        raise ValueError("fit_joint expects positive N, D, loss")
    logn, logd, logy = np.log(n), np.log(d), np.log(y)
    kind = "joint_extra_n" if extra else "joint"
    starts = _starts_joint(n, d, y, extra=extra, n_starts=n_starts, seed=seed)
    if extra_starts:
        starts = list(extra_starts) + starts

    def objective(theta):
        return _objective_joint(theta, logn, logd, logy, delta, extra)

    def diagnose(p):
        info = {"collapsed_to_single": False}
        if extra:
            info.update(_diagnostics_power("two_term", {"E": p["E"], "A": p["A"], "alpha": p["alpha"], "B": p["B"], "beta": p["beta"]}, n))
        return info

    return _minimize_starts(
        kind,
        objective,
        starts,
        n=len(y),
        finalize=lambda th: _pack_joint(th, extra),
        diagnose=diagnose,
        record_starts=record_starts,
    )


def _pack_joint_extra_d(theta: np.ndarray) -> dict:
    e = float(np.exp(theta[0]))
    a = float(np.exp(theta[1]))
    alpha = _softplus(theta[2]) + 1e-8
    gamma = _softplus(theta[4]) + 1e-8
    frac = float(expit(theta[6]))
    delta = max(1e-8, gamma * frac)
    return {
        "E": e,
        "A": a,
        "alpha": alpha,
        "G": float(np.exp(theta[3])),
        "gamma": gamma,
        "H": float(np.exp(theta[5])),
        "delta": delta,
    }


def _objective_joint_extra_d(theta, logn, logd, logy, delta_huber) -> float:
    try:
        p = _pack_joint_extra_d(theta)
        e = np.full_like(logn, np.log(p["E"]))
        t_a = np.log(p["A"]) - p["alpha"] * logn
        t_fast = np.log(p["G"]) - p["gamma"] * logd
        t_slow = np.log(p["H"]) - p["delta"] * logd
        return _huber_from_log_terms(np.vstack([t_a, t_fast, t_slow, e]), logy, delta_huber)
    except Exception:
        return PENALTY


def fit_joint_extra_d(n, d, y, delta: float = 1e-3, n_starts: int = 12, seed: int = 0) -> FitResult:
    """Second token power: gamma is fast (larger), delta = gamma * sigmoid is slow."""
    n, d, y = _as64(n, d, y)
    if np.any(y <= 0) or np.any(n <= 0) or np.any(d <= 0):
        raise ValueError("fit_joint_extra_d expects positive N, D, loss")
    logn, logd, logy = np.log(n), np.log(d), np.log(y)
    # Reuse the 7-parameter starts built for the extra-N model. The layout matches
    # _pack_joint_extra_d: the fraction multiplies the fast token exponent.
    starts = _starts_joint(n, d, y, extra=True, n_starts=n_starts, seed=seed)

    def objective(theta):
        return _objective_joint_extra_d(theta, logn, logd, logy, delta)

    def diagnose(p):
        fast = p["G"] * d ** (-p["gamma"])
        slow = p["H"] * d ** (-p["delta"])
        frac = slow / np.maximum(fast + slow, 1e-30)
        info = {
            "delta_over_gamma": p["delta"] / p["gamma"],
            "median_slow_fraction": float(np.median(frac)),
        }
        info["collapsed_to_single"] = bool(
            info["median_slow_fraction"] < 0.02
            or info["median_slow_fraction"] > 0.98
            or info["delta_over_gamma"] > 0.98
        )
        return info

    return _minimize_starts(
        "joint_extra_d",
        objective,
        starts,
        n=len(y),
        finalize=_pack_joint_extra_d,
        diagnose=diagnose,
    )


def literature_joint_start() -> np.ndarray:
    """Besiroglu/Hoffmann-replication coefficients as an unconstrained start.

    Published point: E=1.82, A=482, alpha=0.34, B=2085, beta=0.36.
    In this code the D-term coefficient is `G` and its exponent is `gamma`.
    """
    return np.array(
        [np.log(1.82), np.log(482.0), _inv_softplus(0.34), np.log(2085.0), _inv_softplus(0.36)],
        dtype=np.float64,
    )


def _minimize_starts(kind, objective, starts, n: int, finalize, diagnose, record_starts: bool = False) -> FitResult:
    ranked = []
    recorded = []
    n_success = 0
    n_finite = 0
    for theta0 in starts:
        res = minimize(objective, np.asarray(theta0, dtype=np.float64), method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-14, "maxls": 40})
        fun = float(res.fun) if np.isfinite(res.fun) else PENALTY
        if fun < PENALTY / 10:
            n_finite += 1
            # Polish the best basin a bit harder.
            res2 = minimize(objective, res.x, method="L-BFGS-B", options={"maxiter": 800, "ftol": 1e-15, "maxls": 40})
            fun2 = float(res2.fun) if np.isfinite(res2.fun) else PENALTY
            if fun2 <= fun:
                res = res2
                fun = fun2
        if bool(res.success) and fun < PENALTY / 10:
            n_success += 1
        ranked.append((fun, np.array(res.x, dtype=np.float64), bool(res.success), str(res.message)))
        if record_starts and fun < PENALTY / 10:
            recorded.append({"objective": fun, "success": bool(res.success), **{k: float(v) for k, v in finalize(res.x).items()}})
    ranked.sort(key=lambda t: t[0])
    best_fun, best_x, best_ok, best_msg = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else np.inf
    rel_gap = float((second - best_fun) / max(abs(best_fun), 1e-12)) if np.isfinite(second) else np.inf
    params = finalize(best_x) if best_fun < PENALTY / 10 else {}
    diagnostics = diagnose(params) if params else {"collapsed_to_single": True}
    if record_starts:
        diagnostics["starts"] = recorded
    return FitResult(
        kind=kind,
        params={k: float(v) for k, v in params.items()},
        objective=float(best_fun),
        mean_huber=float(best_fun / max(n, 1)),
        success=bool(best_ok and best_fun < PENALTY / 10),
        n_starts=len(starts),
        n_finite=int(n_finite),
        n_success_flag=int(n_success),
        rel_gap_second=rel_gap,
        message=best_msg,
        diagnostics=diagnostics,
    )


def regression_metrics(y, yhat) -> dict:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    err = yhat - y
    finite = np.isfinite(err)
    if not np.any(finite):
        return {"mae": np.nan, "rmse": np.nan, "log_mae": np.nan, "log_rmse": np.nan, "n": int(len(y)), "n_nonpositive_pred": int(len(y)), "n_nonfinite_pred": int(len(y))}
    mae = float(np.mean(np.abs(err[finite])))
    rmse = float(np.sqrt(np.mean(err[finite] ** 2)))
    pos = finite & (y > 0) & (yhat > 0)
    if np.all(pos) and len(y):
        le = np.log(yhat) - np.log(y)
        log_mae = float(np.mean(np.abs(le)))
        log_rmse = float(np.sqrt(np.mean(le ** 2)))
    else:
        log_mae = np.nan
        log_rmse = np.nan
    return {
        "mae": mae,
        "rmse": rmse,
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "n": int(len(y)),
        "n_nonpositive_pred": int(np.sum(~(yhat > 0))),
        "n_nonfinite_pred": int(np.sum(~np.isfinite(yhat))),
    }


def holdout_count(n_models: int) -> int:
    """Pre-specified rule: 2 largest models when the ladder has < 15 sizes, else 20%."""
    if n_models < 6:
        raise ValueError(f"need at least 6 model sizes, got {n_models}")
    if n_models < 15:
        return 2
    return max(2, int(round(0.20 * n_models)))


def assign_holdout_by_size(df, n_col: str = "N"):
    """Mark the largest sizes as test. Returns a copy plus the test size values."""
    out = df.copy()
    sizes = np.sort(out[n_col].unique())
    k = holdout_count(len(sizes))
    test_sizes = set(sizes[-k:])
    out["split"] = np.where(out[n_col].isin(test_sizes), "test", "train")
    return out, np.array(sorted(test_sizes))


def assert_self_check(seed: int = 0) -> None:
    """Noiseless recovery. Raises if the fitter is not solving the model it claims to solve."""
    rng = np.random.default_rng(seed)
    n = np.geomspace(1e8, 1e11, 32)
    e, a, alpha = 1.5, 12649.0, 0.5
    y = e + a * n ** (-alpha)
    fit = fit_power("single_term", n, y, n_starts=8, seed=seed)
    if abs(fit.params["alpha"] - alpha) > 0.02 or abs(fit.params["E"] - e) / e > 0.05:
        raise AssertionError(f"single-term self-check failed: {fit.params}")
    b, beta = 25.24, 0.2
    y2 = e + a * n ** (-alpha) + b * n ** (-beta)
    fit2 = fit_power("two_term", n, y2, n_starts=10, seed=seed)
    if abs(fit2.params["alpha"] - alpha) > 0.04 or abs(fit2.params["beta"] - beta) > 0.04:
        raise AssertionError(f"two-term self-check failed: {fit2.params}")
    if not (fit2.params["beta"] < fit2.params["alpha"]):
        raise AssertionError("beta < alpha constraint violated")
    # Joint, mild noise, both terms visible.
    d = np.geomspace(2e9, 2e11, 32)
    nn, dd = np.meshgrid(n[::4], d[::4])
    nn, dd = nn.ravel(), dd.ravel()
    yj = 1.7 + 400.0 * nn ** (-0.35) + 1500.0 * dd ** (-0.30)
    yj = yj * np.exp(rng.normal(0.0, 0.005, size=yj.shape))
    fitj = fit_joint(nn, dd, yj, n_starts=8, seed=seed)
    if abs(fitj.params["alpha"] - 0.35) > 0.05 or abs(fitj.params["gamma"] - 0.30) > 0.05:
        raise AssertionError(f"joint self-check failed: {fitj.params}")


if __name__ == "__main__":
    assert_self_check()
    print("self-check ok")
