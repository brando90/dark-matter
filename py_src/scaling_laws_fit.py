# file: scaling_laws_fit.py
import warnings

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Callable
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from scipy.stats import chi2

try:  # imported as part of the py_src package
    from .two_term_lrt import FitError, InsufficientDataError, LRTResult, bootstrap_lr_test, validate_ladder
except ImportError:  # run as a script: python py_src/scaling_laws_fit.py (py_src/ is then sys.path[0])
    from two_term_lrt import FitError, InsufficientDataError, LRTResult, bootstrap_lr_test, validate_ladder

# Fewest points fit_two_term_N_only accepts: five free parameters need at least six points to leave one residual
# degree of freedom. lr_test needs 7 (two_term_lrt.MIN_POINTS_LRT, DM-1b's testability rule).
MIN_POINTS_TWO_TERM_FIT = 6

# ---------------------------
# Utilities
# ---------------------------
def huber(z, delta=1e-3):
    """Elementwise Huber loss (same delta used in the Chinchilla replication)."""
    az = np.abs(z)
    quad = az <= delta
    out = np.empty_like(z)
    out[quad] = 0.5 * (z[quad] ** 2)
    out[~quad] = delta * (az[~quad] - 0.5 * delta)
    return out

def huber_logpdf(z, delta=1e-3):
    """
    Density whose negative log-likelihood equals Huber loss (up to a constant).
    We only need log-likelihood differences for LR tests, so constants cancel.
    See Epoch AI replication Appendix A.2.
    """
    # -log p ∝ huber(z); return log p up to additive constant
    return -huber(z, delta=delta)

def softplus(x):  # stable softplus
    return np.where(x > 20, x, np.log1p(np.exp(x)))

# ---------------------------
# Data access
# ---------------------------
def load_ladder_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Minimal cleaning
    assert {'N_params', 'tokens_D', 'loss'}.issubset(df.columns)
    df = df.dropna(subset=['N_params', 'tokens_D', 'loss'])
    return df

# ---------------------------
# (A) Joint fit: L(N, D) = E + A/N^alpha + B/D^beta
#     Robust Chinchilla-style Huber objective with log-sum-exp
# ---------------------------
@dataclass
class JointParams:
    E: float; A: float; alpha: float; B: float; beta: float

def _pack_joint(theta):
    # Unconstrained params -> positive scales/exponents
    # theta = [e_raw, a_raw, alpha_raw, b_raw, beta_raw]
    E = np.exp(theta[0])     # keep positive
    A = np.exp(theta[1])
    alpha = softplus(theta[2]) + 1e-6
    B = np.exp(theta[3])
    beta = softplus(theta[4]) + 1e-6
    return E, A, alpha, B, beta

def fit_joint_powerlaw(N, D, L, delta=1e-3) -> JointParams:
    """
    Fit using the Huber objective used in Hoffmann et al. (Approach 3)
    as reconstructed in the Epoch AI replication: minimize
      sum Huber( LSE(a - alpha log N, b - beta log D, e) - log L ),
    where a=log A, b=log B, e=log E.
    """
    logN = np.log(N); logD = np.log(D); logL = np.log(L)

    def obj(theta):
        E, A, alpha, B, beta = _pack_joint(theta)
        a, b, e = np.log(A), np.log(B), np.log(E)
        # r_i = LSE(a - alpha log N_i, b - beta log D_i, e) - log L_i
        X = np.vstack([a - alpha * logN,
                       b - beta * logD,
                       np.full_like(logN, e)])
        r = logsumexp(X, axis=0) - logL
        return np.sum(huber(r, delta=delta))

    # Sensible init from rough scalings
    theta0 = np.array([np.log(np.median(L)),
                       np.log(np.median(L)*N.mean()**0.3),
                       np.log(0.3),  # alpha_raw ~ log(0.3) -> softplus ~ 0.3
                       np.log(np.median(L)*D.mean()**0.3),
                       np.log(0.3)])
    res = minimize(obj, theta0, method='L-BFGS-B')
    E, A, alpha, B, beta = _pack_joint(res.x)
    return JointParams(E, A, alpha, B, beta)

def predict_joint(params: JointParams, N, D):
    return params.E + params.A / (N ** params.alpha) + params.B / (D ** params.beta)

# ---------------------------
# (B) Two-term N-only fit: L(N) = E + A/N^alpha + B/N^beta, with beta < alpha
# ---------------------------
@dataclass
class TwoTermParams:
    """L(N) = E + A N^-alpha + B N^-beta with beta < alpha: alpha is the FAST exponent, beta the slow one.

    converged / objective / message come from the optimizer (fit_two_term_N_only); they default to values that
    mean "not recorded" when a TwoTermParams is built by hand.
    """
    E: float; A: float; alpha: float; B: float; beta: float
    converged: bool = True
    objective: float = float("nan")
    message: str = ""

def _pack_two_term(theta):
    """
    Enforce beta < alpha by construction:
      alpha = softplus(u) + eps
      beta  = alpha * sigmoid(v)
    Keep E, A, B positive via exp.
    """
    u_alpha, v_frac, e_raw, a_raw, b_raw = theta
    alpha = softplus(u_alpha) + 1e-6
    frac = expit(v_frac)  # in (0,1)
    beta = np.maximum(1e-6, alpha * frac)
    E = np.exp(e_raw); A = np.exp(a_raw); B = np.exp(b_raw)
    return E, A, alpha, B, beta

def fit_two_term_N_only(N, L, delta=1e-3) -> TwoTermParams:
    """Huber fit of L(N) = E + A/N^alpha + B/N^beta, beta < alpha (alpha fast), from a single start.

    Fails loudly: raises InsufficientDataError for an empty selection or fewer than MIN_POINTS_TWO_TERM_FIT points
    (or fewer than five distinct sizes), ValueError for NaN/inf/non-positive values or mismatched shapes, and
    FitError if the optimizer returns a non-finite solution. It never substitutes other data. If L-BFGS-B stops
    without converging, a RuntimeWarning is issued and the result carries converged=False and the optimizer message.
    """
    N, L = validate_ladder(N, L, min_points=MIN_POINTS_TWO_TERM_FIT, who="fit_two_term_N_only")
    logN = np.log(N); logL = np.log(L)
    def obj(theta):
        E, A, alpha, B, beta = _pack_two_term(theta)
        # Construct log(E + A N^{-alpha} + B N^{-beta}) via log-sum-exp in log-space
        t1 = np.log(A) - alpha * logN
        t2 = np.log(B) - beta * logN
        t3 = np.log(E) + 0.0 * logN
        # r_i = log(pred_i) - log(L_i)
        r = logsumexp(np.vstack([t1, t2, t3]), axis=0) - logL
        return np.sum(huber(r, delta=delta))
    theta0 = np.array([np.log(0.3),  # u_alpha
                       0.0,           # v_frac -> beta ~ 0.5*alpha
                       np.log(np.median(L)), 
                       np.log(np.median(L) * N.mean()**0.3),
                       np.log(np.median(L) * N.mean()**0.1)])
    res = minimize(obj, theta0, method='L-BFGS-B')
    if not (np.all(np.isfinite(res.x)) and np.isfinite(res.fun)):
        raise FitError(f"fit_two_term_N_only: optimizer returned a non-finite solution ({res.message})")
    E, A, alpha, B, beta = _pack_two_term(res.x)
    if not res.success:
        warnings.warn(f"fit_two_term_N_only: L-BFGS-B did not converge ({res.message}); returning the last iterate",
                      RuntimeWarning, stacklevel=2)
    return TwoTermParams(E, A, alpha, B, beta, converged=bool(res.success), objective=float(res.fun),
                         message=str(res.message))

def predict_two_term(p: TwoTermParams, N):
    return p.E + p.A / (N ** p.alpha) + p.B / (N ** p.beta)

# ---------------------------
# Likelihood-ratio test (Huber-as-likelihood)
# ---------------------------
def huber_loglike(residuals, delta=1e-3):
    return np.sum(huber_logpdf(residuals, delta=delta))

def lr_test(N, L, *args, **kwargs) -> LRTResult:
    """Calibrated bootstrap likelihood-ratio test for a second power-law term in N (DM-1b), one ladder at a time.

        res = lr_test(N, L)             # res.detect, res.p_value, res.T, res.one_term, res.two_term
        res = lr_test(N, L, B=1000, rng=0)

    A parametric bootstrap of T = npts * ln(RSS_1 / RSS_2) (Gaussian noise on L), implemented in
    py_src/two_term_lrt.py as a verbatim port of experiments/dm1_power_analysis/dm1b.py; pooled null rejection 0.049
    at level 0.05 (DM-1b, 12,000 null ladders). It needs at least 7 points and raises InsufficientDataError below
    that. In the result, two_term.alpha is the FAST exponent and two_term.beta the slow one. It assumes additive,
    constant-sd Gaussian noise on L; with multiplicative noise and a wide loss range it over-rejects (folder 17).

    This replaced the chi-square test on the Huber objective, which was not calibrated (folder 16: its statistic sat
    4e4 to 9e4 times below the 5.99 cutoff on simulated ten-size ladders, so it could never reject). Calling the old form,
    lr_test(loglike_small, k_small, loglike_big, k_big), raises TypeError.
    """
    if args or np.ndim(N) == 0:
        raise TypeError(
            "lr_test(loglike_small, k_small, loglike_big, k_big) was removed because its chi-square statistic on the "
            "Huber objective is not calibrated (experiments/16_*). Call lr_test(N, L) for the bootstrap test, or "
            "lr_test_chi2_uncalibrated(...) to reproduce the old number.")
    return bootstrap_lr_test(N, L, **kwargs)


def lr_test_chi2_uncalibrated(loglike_small, k_small, loglike_big, k_big):
    """The OLD lr_test: 2 * (loglike_big - loglike_small) referred to chi-square(k_big - k_small).

    Kept only to reproduce the folder 16 finding. The Huber objective is not a likelihood (delta = 1e-3 makes it ~1e-5
    in these units), so the statistic never reaches the cutoff and the test cannot reject; do not use it to decide.
    """
    warnings.warn("lr_test_chi2_uncalibrated is not calibrated and cannot reject; use lr_test(N, L)",
                  UserWarning, stacklevel=2)
    stat = 2.0 * (loglike_big - loglike_small)
    dof = k_big - k_small
    p = 1.0 - chi2.cdf(stat, dof)
    return stat, dof, p

def residuals_joint(params: JointParams, N, D, L):
    Lhat = predict_joint(params, N, D)
    return np.log(Lhat) - np.log(L)

def residuals_two_term(params: TwoTermParams, N, L):
    Lhat = predict_two_term(params, N)
    return np.log(Lhat) - np.log(L)

# ---------------------------
# Compute-optimal policy under C ≈ k N D
# For L = E + A/N^alpha + B/D^beta, the optimal exponents are:
#   N* ∝ C^{β/(α+β)},  D* ∝ C^{α/(α+β)},  and a = β/(α+β)
# The tokens-per-parameter ratio at optimum:
#   D/N = ((β B)/(α A))^{1/β} * N^{α/β - 1}
# ---------------------------
def compute_optimal_from_joint(p: JointParams, k=6.0):
    a = p.beta / (p.alpha + p.beta)
    b = 1.0 - a
    return dict(a=a, b=b)

def tokens_per_param_opt(p: JointParams, N_value):
    const = ((p.beta * p.B) / (p.alpha * p.A)) ** (1.0 / p.beta)
    return const * (N_value ** (p.alpha / p.beta - 1.0))

# ---------------------------
# Synthetic data generator (for dry runs)
# ---------------------------
def synth_joint(Ns, Ds, E=1.8, A=500.0, alpha=0.35, B=2000.0, beta=0.36, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    L = E + A / (Ns ** alpha) + B / (Ds ** beta)
    L *= np.exp(rng.normal(0.0, noise, size=L.shape))  # multiplicative noise in log-loss
    return L

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("SCALING LAWS FIT - SYNTHETIC DATA TEST")
    print("=" * 80)
    print()
    
    # Ground truth parameters
    print("Ground Truth Parameters:")
    print("  E = 1.8, A = 500.0, α = 0.35, B = 2000.0, β = 0.36")
    print()
    
    # 1) Try synthetic to verify the pipeline
    Ns = np.geomspace(1e7, 1e11, 20)
    Ds = np.geomspace(1e9, 1e12, 20)
    # make a small grid
    NN, DD = np.meshgrid(Ns, Ds); NN = NN.ravel(); DD = DD.ravel()
    L = synth_joint(NN, DD)
    
    print(f"Generated {len(L)} synthetic data points")
    print(f"N range: [{NN.min():.2e}, {NN.max():.2e}]")
    print(f"D range: [{DD.min():.2e}, {DD.max():.2e}]")
    print(f"Loss range: [{L.min():.4f}, {L.max():.4f}]")
    print()
    print("-" * 80)

    # Fit joint law
    print("(1) FITTING JOINT LAW: L(N,D) = E + A/N^α + B/D^β")
    print()
    jp = fit_joint_powerlaw(NN, DD, L)
    print(f"  E     = {jp.E:.4f}  (true: 1.8)")
    print(f"  A     = {jp.A:.2f}  (true: 500.0)")
    print(f"  α     = {jp.alpha:.4f}  (true: 0.35)")
    print(f"  B     = {jp.B:.2f}  (true: 2000.0)")
    print(f"  β     = {jp.beta:.4f}  (true: 0.36)")
    print()
    
    # Compute-optimal exponents
    opt = compute_optimal_from_joint(jp)
    print(f"  Compute-optimal split: a = β/(α+β) = {opt['a']:.4f}")
    print(f"  This means ~{opt['a']*100:.1f}% compute on params, ~{opt['b']*100:.1f}% on tokens")
    print()
    print("-" * 80)
    
    # Fit two-term N-only on a genuine N-ladder: N varies, D is held fixed (a slice of the grid cannot be used: the
    # old narrow D-band selected 0 points and the script silently fitted all 400 points, with D varying, instead).
    print("(2) FITTING TWO-TERM N-ONLY LAW: L(N) = E + A/N^α + B/N^β with β < α  (α is the fast exponent)")
    print()
    D_FIX = float(np.median(DD))
    N_lad = np.geomspace(1e7, 1e11, 20)
    L_lad = synth_joint(N_lad, np.full_like(N_lad, D_FIX), seed=1)
    print(f"  N-ladder: {len(N_lad)} sizes in [{N_lad.min():.1e}, {N_lad.max():.1e}] at fixed D = {D_FIX:.2e}")
    print("  (true law along N is one term plus a constant: E + B/D^β is absorbed into the floor)")

    # fit_two_term_N_only raises on an empty or too-small ladder; there is deliberately no fallback to other data.
    tp = fit_two_term_N_only(N_lad, L_lad)

    print(f"  E     = {tp.E:.4f}")
    print(f"  A     = {tp.A:.2f}")
    print(f"  α     = {tp.alpha:.4f}  (fast)")
    print(f"  B     = {tp.B:.2f}")
    print(f"  β     = {tp.beta:.4f}  (slow)")
    print(f"  Constraint satisfied: β < α? {tp.beta < tp.alpha}")
    print(f"  Optimizer converged: {tp.converged}")
    print()
    print("-" * 80)

    # Calibrated bootstrap LRT (DM-1b) for a second term on the same ladder.
    print("(3) BOOTSTRAP LIKELIHOOD-RATIO TEST: 1-term vs 2-term N-only model (lr_test)")
    print()
    res = lr_test(N_lad, L_lad, B=100, rng=0, early_stop=True)
    print(f"  Test statistic T = n ln(RSS_1/RSS_2): {res.T:.4g}")
    print(f"  Bootstrap statistics reaching T: {res.k_exceed} of {res.B_used}")
    print(f"  p-value: {res.p_value:.3g}   (level {res.level}; B = {res.B}, so detection needs k <= "
          f"{int(np.floor(res.level * (res.B + 1) + 1e-9)) - 1})")
    if res.detect:
        print("  ✓ Second term detected at the 5% level")
    else:
        print("  × Second term NOT detected at the 5% level (the ladder has one term by construction)")
    print()
    print("=" * 80)
    print("SCALING LAWS FIT TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)

