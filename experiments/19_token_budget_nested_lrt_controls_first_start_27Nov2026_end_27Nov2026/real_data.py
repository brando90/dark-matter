"""Folder 19 real-data stage, gated: runs the nested joint-law LRT on a dataset's real tuned-constant ladder ONLY if all three
control gates of that dataset passed (results/controls_summary.json). B = 1000, seed fixed. Verdict: FAST TERM DETECTED iff
p <= 0.05, else NOT DETECTED. Also reports (descriptive) the fitted fast-term size at 5.17M, and a sensitivity p-value from
a bootstrap with within-width correlated noise (rho = 0.5). Writes results/real_results.json (or real_not_run.json).
Usage: python3 -I real_data.py"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from py_src.nested_joint_lrt import Design  # noqa: E402

SEED, B_REAL, N5, RHO = 20261127, 1000, 5173248.0, 0.5


def corr_bootstrap_p(design, f1, T, width, B, rng):
    """Sensitivity: bootstrap under M1 with noise shared within a width (rho = 0.5); the test statistic is unchanged."""
    sigma = float(np.sqrt(f1.rss / (design.npts - 5)))
    scale = sigma / design._sw
    ids = {w: i for i, w in enumerate(np.unique(width))}
    idx = np.array([ids[w] for w in width])
    T_boot = np.empty(B)
    for i in range(B):
        u = rng.standard_normal(len(ids))[idx]
        noise = np.sqrt(RHO) * u + np.sqrt(1 - RHO) * rng.standard_normal(design.npts)
        T_boot[i] = design.statistic(f1.fitted + scale * noise)[0]
    return (1 + int((T_boot >= T - 1e-12).sum())) / (B + 1)


def fit_dataset(df, i):
    """The real-data computation for one dataset's design table (i = its index in ("rw", "owt2")); main() saves the result."""
    design = Design(df.N_tied.values, df.D.values, df.sd.values)
    r = design.bootstrap_lrt(df.y.values, B=B_REAL, rng=np.random.default_rng([SEED, 100 + i]))
    f1, f2 = r["fit1"], r["fit2"]
    use2 = f2.rss < f1.rss
    return dict(T=r["T"], p=r["p"], B=B_REAL, verdict="FAST TERM DETECTED" if r["detect"] else "NOT DETECTED",
                sigma_hat=r["sigma_hat"], m1_exps=f1.exps, m1_coef=f1.coef, m2_exps=f2.exps if use2 else None,
                m2_coef=f2.coef if use2 else None,
                fast_at_5M=float(f2.coef[3] * (N5 / 1e6) ** -f2.exps[2]) if use2 else 0.0,
                p_correlated_noise=corr_bootstrap_p(design, f1, r["T"], df.width.values, 500,
                                                    np.random.default_rng([SEED, 200 + i])))


def main():
    summ = json.load(open(os.path.join(HERE, "results", "controls_summary.json")))
    out, skipped = {}, {}
    for i, ds in enumerate(("rw", "owt2")):
        if not summ[ds]["all_gates_passed"]:
            skipped[ds] = summ[ds]
            continue
        out[ds] = fit_dataset(pd.read_csv(os.path.join(HERE, "results", f"design_{ds}.csv")), i)
    if out:
        json.dump(out, open(os.path.join(HERE, "results", "real_results.json"), "w"), indent=1)
    if skipped:
        json.dump(skipped, open(os.path.join(HERE, "results", "real_not_run.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)); print("not run:", json.dumps(skipped))


if __name__ == "__main__":
    main()
