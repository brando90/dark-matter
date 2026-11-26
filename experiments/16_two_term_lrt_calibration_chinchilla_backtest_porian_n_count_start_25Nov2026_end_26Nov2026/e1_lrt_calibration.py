"""
E1 -- Is the two-term N-only "dark matter" test calibrated, and does the extra term help extrapolation?

Simulates N-ladders (K=10 sizes, geometric 1e7..1.2e10) from
    H0 (one term):  L = 1.7 + 120 / N^0.30
    H1 (two terms): L = 1.7 + 0.7 (N/1e7)^-0.5 + 0.5 (N/1e7)^-0.15
with multiplicative log-normal noise sigma in {0.003, 0.01}, and compares three things:

  * repo-style statistic   2 * (sum Huber_1 - sum Huber_2), delta = 1e-3, referred to chi^2_2
  * Gaussian LRT           K * log(RSS_1 / RSS_2) on log-residuals, referred to chi^2_2
  * the empirical H0 distribution of both (size-corrected power, AUC)

plus a held-out backtest: fit on the 7 smallest models, extrapolate to the 3 largest, 1-term vs 2-term vs a
"gated" rule that uses the 2-term model only when the training-set LRT clears the empirical H0 threshold.

Usage:
    python e1_lrt_calibration.py --reps0 150 --reps1 100 [--workers 2]
    python e1_lrt_calibration.py --analyze-only
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, chi2, rankdata

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from py_src import scaling_ext as se  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
REPS_FILE = OUT / "e1_reps.jsonl"
SUMMARY_JSON = OUT / "e1_summary.json"

K, K_TRAIN = 10, 7
N_MIN, N_MAX = 1e7, 1.2e10
NS = np.geomspace(N_MIN, N_MAX, K)
SIGMAS = (0.003, 0.01)
FIT = dict(n_starts=150, n_refine=3, max_nfev=1000)
TIE_TOL = 1e-6        # held-out RMSEs closer than this are ties (the extra term was switched off, or optimizer noise); same as E3
N_BOOT = 4000         # bootstrap resamples of ladders for the interval on the median error ratio

TRUTH = {
    "H0": lambda N: 1.7 + 120.0 / N ** 0.30,
    "H1": lambda N: 1.7 + 0.7 * (N / 1e7) ** -0.5 + 0.5 * (N / 1e7) ** -0.15,
}
TRUE_H1 = dict(alpha=0.5, beta=0.15, ratio=0.3)


def _pair(N, L, objective, seed):
    one = se.fit(se.ONE_N, N, L, objective=objective, seed=seed, **FIT)
    two = se.fit(se.TWO_N, N, L, objective=objective, seed=seed + 1, warm=one, **FIT)
    return one, two


def one_rep(args):
    truth, sigma, rep = args
    seed = 100_000 * (0 if truth == "H0" else 1) + 10_000 * SIGMAS.index(sigma) + rep
    rng = np.random.default_rng(seed)
    clean = TRUTH[truth](NS)
    L = clean * np.exp(rng.normal(0.0, sigma, K))

    h1, h2 = _pair(NS, L, "huber", seed)           # repo-style (Huber, delta=1e-3) on all K points
    g1, g2 = _pair(NS, L, "ls", seed + 2)          # Gaussian on all K points

    tr = slice(0, K_TRAIN)
    te = slice(K_TRAIN, K)
    th1, th2 = _pair(NS[tr], L[tr], "huber", seed + 3)   # backtest fits (Huber), 7 smallest models
    tg1, tg2 = _pair(NS[tr], L[tr], "ls", seed + 4)      # training-set Gaussian LRT for the gate

    def err(res, ref):
        return np.log(se.predict(res, NS[te])) - np.log(ref)

    row = dict(
        truth=truth, sigma=sigma, rep=rep, seed=seed,
        h_obj1=h1.objective, h_obj2=h2.objective, h_rss1=h1.rss, h_rss2=h2.rss,
        g_rss1=g1.rss, g_rss2=g2.rss,
        h1_E=h1.E, h1_A=float(h1.c[0]), h1_a=float(h1.e[0]),
        h2_E=h2.E, h2_c1=float(h2.c[0]), h2_a=float(h2.e[0]), h2_c2=float(h2.c[1]), h2_b=float(h2.e[1]),
        tr_g_rss1=tg1.rss, tr_g_rss2=tg2.rss,
        bt_true_rmse1=float(np.sqrt(np.mean(err(th1, clean[te]) ** 2))),
        bt_true_rmse2=float(np.sqrt(np.mean(err(th2, clean[te]) ** 2))),
        bt_obs_rmse1=float(np.sqrt(np.mean(err(th1, L[te]) ** 2))),
        bt_obs_rmse2=float(np.sqrt(np.mean(err(th2, L[te]) ** 2))),
        bt_last1=float(err(th1, clean[te])[-1]), bt_last2=float(err(th2, clean[te])[-1]),
    )
    return row


# ------------------------------------------------------------------------------------------------
def run(reps0: int, reps1: int, workers: int):
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if REPS_FILE.exists():
        for line in REPS_FILE.read_text().splitlines():
            r = json.loads(line)
            done.add((r["truth"], r["sigma"], r["rep"]))
    jobs = []
    for sigma in SIGMAS:
        for rep in range(reps0):
            jobs.append(("H0", sigma, rep))
        for rep in range(reps1):
            jobs.append(("H1", sigma, rep))
    jobs = [j for j in jobs if j not in done]
    # interleave so partial results are balanced across settings
    jobs.sort(key=lambda j: (j[2], j[0], j[1]))
    print(f"[E1] {len(jobs)} reps to run ({len(done)} already done), workers={workers}", flush=True)
    t0 = time.time()
    with Pool(workers) as pool, open(REPS_FILE, "a") as f:
        for i, row in enumerate(pool.imap_unordered(one_rep, jobs, chunksize=1), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f"[E1] {i}/{len(jobs)} reps, {time.time() - t0:.0f}s elapsed", flush=True)


def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    r = rankdata(allv)
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% interval for a binomial proportion k / n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def noiseless_one_term_misfit():
    """How much of the two-term truth can a one-term law absorb? Least-squares fit of E + A/N^a to the noise-free
    two-term ladder (same ten sizes); the misfit is the largest signal any test or fit could use at zero noise."""
    clean = TRUTH["H1"](NS)
    res = se.fit(se.ONE_N, NS, clean, objective="ls", n_starts=300, n_refine=5, seed=0, max_nfev=4000)
    lr = np.log(se.predict(res, NS)) - np.log(clean)
    return dict(E=float(res.E), alpha=float(res.e[0]), max_abs_log_resid=float(np.abs(lr).max()),
                rms_log_resid=float(np.sqrt(np.mean(lr ** 2))))


def analyze():
    rows = [json.loads(l) for l in REPS_FILE.read_text().splitlines()]
    crit = float(chi2.ppf(0.95, 2))
    summary = dict(chi2_2_crit_95=crit, K=K, K_train=K_TRAIN, tie_tol=TIE_TOL,
                   noiseless_two_term_truth_one_term_fit=noiseless_one_term_misfit(), settings={})
    for sigma in SIGMAS:
        s = {}
        sel = lambda t: [r for r in rows if r["truth"] == t and r["sigma"] == sigma]
        r0, r1 = sel("H0"), sel("H1")
        if not r0 or not r1:
            continue

        def stats(rs):
            h = np.array([2.0 * (r["h_obj1"] - r["h_obj2"]) for r in rs])
            g = np.array([K * np.log(r["g_rss1"] / r["g_rss2"]) for r in rs])
            tg = np.array([K_TRAIN * np.log(r["tr_g_rss1"] / r["tr_g_rss2"]) for r in rs])
            return np.maximum(h, 0.0), np.maximum(g, 0.0), np.maximum(tg, 0.0)

        h0, g0, tg0 = stats(r0)
        h1, g1, tg1 = stats(r1)
        qh, qg, qtg = (float(np.quantile(x, 0.95)) for x in (h0, g0, tg0))
        s["n_H0"], s["n_H1"] = len(r0), len(r1)
        # repo-style statistic
        s["repo_stat_H0_median"] = float(np.median(h0))
        s["repo_stat_H0_q95"] = qh
        s["repo_stat_H1_median"] = float(np.median(h1))
        s["repo_pvalue_H0_median"] = float(np.median(chi2.sf(h0, 2)))
        s["repo_pvalue_H1_median"] = float(np.median(chi2.sf(h1, 2)))
        s["repo_nominal_false_positive_rate"] = float(np.mean(chi2.sf(h0, 2) < 0.05))
        s["repo_nominal_power"] = float(np.mean(chi2.sf(h1, 2) < 0.05))
        s["repo_corrected_power"] = float(np.mean(h1 > qh))
        s["repo_auc"] = auc(h1, h0)
        # Gaussian LRT
        s["gauss_stat_H0_median"] = float(np.median(g0))
        s["gauss_stat_H0_q95"] = qg
        s["gauss_stat_H1_median"] = float(np.median(g1))
        s["gauss_nominal_false_positive_rate"] = float(np.mean(chi2.sf(g0, 2) < 0.05))
        s["gauss_nominal_power"] = float(np.mean(chi2.sf(g1, 2) < 0.05))
        s["gauss_corrected_power"] = float(np.mean(g1 > qg))
        s["gauss_auc"] = auc(g1, g0)
        # calibration detail: counts with Wilson intervals, the point mass at zero under the null, and the seven-size ladders
        k10, k7 = int(np.sum(chi2.sf(g0, 2) < 0.05)), int(np.sum(chi2.sf(tg0, 2) < 0.05))
        s["gauss_nominal_false_positive_count"] = [k10, len(r0)]
        s["gauss_nominal_false_positive_ci95"] = wilson(k10, len(r0))
        s["gauss_stat_H0_zero_share"] = float(np.mean(g0 <= 1e-12))
        s["train7_gauss_nominal_false_positive_count"] = [k7, len(r0)]
        s["train7_gauss_nominal_false_positive_rate"] = k7 / len(r0)
        s["train7_gauss_nominal_false_positive_ci95"] = wilson(k7, len(r0))
        s["train7_gauss_stat_H0_zero_share"] = float(np.mean(tg0 <= 1e-12))
        # identifiability: where does the 2-term Huber fit land?
        def corner(rs):
            return float(np.mean([(r["h2_b"] < 0.02) or (r["h2_E"] < 0.2 * r["h1_E"]) for r in rs]))
        s["two_term_corner_rate_H0"] = corner(r0)
        s["two_term_corner_rate_H1"] = corner(r1)
        # boundary solutions of the two-term Huber fit (see scaling_ext.diagnose): a coefficient at its e^40 cap,
        # E below 0.1% of the smallest loss (about 1.8), or the shallow exponent below 0.02
        def bounds(rs):
            c_hi = np.array([(r["h2_c1"] >= np.exp(39.5)) or (r["h2_c2"] >= np.exp(39.5)) for r in rs])
            e0 = np.array([r["h2_E"] <= 1.8e-3 for r in rs])
            b_lo = np.array([r["h2_b"] <= 0.02 for r in rs])
            return dict(coef_at_cap=float(c_hi.mean()), E_near_zero=float(e0.mean()), shallow_exp_below_0p02=float(b_lo.mean()),
                        any_boundary=float((c_hi | e0 | b_lo).mean()))
        s["two_term_boundary_H0"], s["two_term_boundary_H1"] = bounds(r0), bounds(r1)
        a = np.array([r["h2_a"] for r in r1]); b = np.array([r["h2_b"] for r in r1])
        s["H1_alpha_hat_median_IQR"] = [float(np.median(a)), float(np.quantile(a, 0.25)), float(np.quantile(a, 0.75))]
        s["H1_beta_hat_median_IQR"] = [float(np.median(b)), float(np.quantile(b, 0.25)), float(np.quantile(b, 0.75))]
        rho = b / a
        s["H1_ratio_hat_median_IQR"] = [float(np.median(rho)), float(np.quantile(rho, 0.25)), float(np.quantile(rho, 0.75))]
        s["H1_frac_beta_within_0.05_of_truth"] = float(np.mean(np.abs(b - TRUE_H1["beta"]) < 0.05))
        # backtest vs the noise-free truth
        def bt(rs, label):
            out = {}
            r1_ = np.array([r["bt_true_rmse1"] for r in rs]); r2_ = np.array([r["bt_true_rmse2"] for r in rs])
            gate = np.array([max(K_TRAIN * np.log(r["tr_g_rss1"] / r["tr_g_rss2"]), 0.0) > qtg for r in rs])
            rg = np.where(gate, r2_, r1_)
            f = lambda x: dict(median=float(np.median(x)), mean=float(np.mean(x)), q90=float(np.quantile(x, 0.9)))
            out["always_1term"], out["always_2term"], out["gated"] = f(r1_), f(r2_), f(rg)
            # tie-aware paired comparison per ladder (strict < counts optimizer-noise differences of 1e-10 as wins)
            d = r2_ - r1_
            w, t, l = int(np.sum(d < -TIE_TOL)), int(np.sum(np.abs(d) <= TIE_TOL)), int(np.sum(d > TIE_TOL))
            out["two_term_wins_ties_losses"] = [w, t, l]
            out["sign_test_p_wins_vs_losses"] = float(binomtest(w, w + l, 0.5).pvalue) if w + l else 1.0
            # median error ratio with a bootstrap interval over ladders
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(r1_), (N_BOOT, len(r1_)))
            ratio = np.median(r2_) / np.median(r1_) - 1.0
            boot = np.array([np.median(r2_[i]) / np.median(r1_[i]) - 1.0 for i in idx])
            out["median_error_ratio_2term_vs_1term"] = dict(
                value=float(ratio), ci95=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))])
            out["gate_fires"] = float(np.mean(gate))
            return out
        s["backtest_H0"] = bt(r0, "H0")
        s["backtest_H1"] = bt(r1, "H1")
        s["train_gauss_q95_H0"] = qtg
        summary["settings"][f"sigma={sigma}"] = s
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps0", type=int, default=150)
    ap.add_argument("--reps1", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if not a.analyze_only:
        run(a.reps0, a.reps1, a.workers)
    analyze()
