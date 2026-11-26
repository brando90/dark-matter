"""
Sanity checks for py_src/scaling_ext.py against the repo's own single-start fitter, on synthetic and real data.
The log is saved to results/validate_scaling_ext.txt.

    python validate_scaling_ext.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from py_src import scaling_ext as se  # noqa: E402
from py_src.scaling_laws_fit import fit_joint_powerlaw  # noqa: E402

RES = Path(__file__).resolve().parent / "results"


def as_fit(jp):
    return se.FitResult(se.JOINT, jp.E, np.array([jp.A, jp.B]), np.array([jp.alpha, jp.beta]), 0.0, 0.0, 0, True)


def line(name, res, N, L, D, extra=""):
    r = se.log_resid(res, N, L, D)
    return (f"  {name:<28s} {se.describe(res):<62s} huber={se.objective_of(res, N, L, D):.6f} "
            f"rss={float(np.sum(r ** 2)):.5f} mean_resid={float(np.mean(r)):+.5f} {extra}")


def main():
    out = []
    P = lambda *a: (out.append(" ".join(str(x) for x in a)), print(*a, flush=True))
    rng = np.random.default_rng(1)

    P("TEST 1  synthetic joint data, truth E=1.82 A=482 a=0.34 B=2085 b=0.36, 400 points, 2% log-normal noise")
    N = np.exp(rng.uniform(np.log(7e7), np.log(1.6e10), 400))
    D = np.exp(rng.uniform(np.log(5e9), np.log(5e11), 400))
    L = (1.82 + 482 / N ** 0.34 + 2085 / D ** 0.36) * np.exp(rng.normal(0, 0.02, 400))
    t = time.time(); mine = se.fit(se.JOINT, N, L, D, n_starts=200); tm = time.time() - t
    t = time.time(); rf = as_fit(fit_joint_powerlaw(N, D, L)); tr = time.time() - t
    P(line("multi-start (scaling_ext)", mine, N, L, D, f"{tm:.2f}s"))
    P(line("repo single start", rf, N, L, D, f"{tr:.2f}s"))

    P("\nTEST 2  nested property on a 10-point two-term ladder (1-term fit, then 2-term warm-started from it)")
    Ns = np.geomspace(1e7, 1.2e10, 10)
    Ln = (1.7 + 0.7 * (Ns / 1e7) ** -0.5 + 0.5 * (Ns / 1e7) ** -0.15) * np.exp(rng.normal(0, 0.005, 10))
    one = se.fit(se.ONE_N, Ns, Ln, n_starts=60)
    two = se.fit(se.TWO_N, Ns, Ln, n_starts=200, warm=one)
    P(f"  1-term: {se.describe(one)} | huber {one.objective:.4e}")
    P(f"  2-term: {se.describe(two)} | huber {two.objective:.4e} | nested (2-term <= 1-term): {two.objective <= one.objective + 1e-12}")

    P("\nTEST 3  real Chinchilla points (MassiveText), n = 245")
    df = pd.read_csv(RES / "chinchilla_245.csv")
    N, D, L = df.N.values, df.D.values, df.L.values
    mine = se.fit(se.JOINT, N, L, D, n_starts=300, n_refine=6)
    rf = as_fit(fit_joint_powerlaw(N, D, L))
    ref = se.FitResult(se.JOINT, 1.82, np.array([482.0, 2085.0]), np.array([0.34, 0.36]), 0.0, 0.0, len(L), True)
    hof = se.FitResult(se.JOINT, 1.69, np.array([406.4, 410.7]), np.array([0.34, 0.28]), 0.0, 0.0, len(L), True)
    P(line("multi-start (scaling_ext)", mine, N, L, D))
    P(line("repo single start", rf, N, L, D))
    P(line("repo 'target' constants", ref, N, L, D, "(E=1.82, A=482, a=0.34, B=2085, b=0.36)"))
    P(line("Hoffmann et al. original", hof, N, L, D, "(E=1.69, A=406.4, a=0.34, B=410.7, b=0.28)"))

    P("\nTEST 4  seed stability of the multi-start joint fit on the 245 points")
    for s in range(5):
        m = se.fit(se.JOINT, N, L, D, n_starts=200, n_refine=6, seed=100 + s)
        P(f"  seed {100 + s}: huber={m.objective:.6f}  {se.describe(m)}")

    P("\nTEST 5  profile freezing: free 2-term fit vs the same fit with alpha frozen (the frozen fit can never be better)")
    free = se.fit(se.TWO_N, Ns, Ln, n_starts=200)
    for a in (0.3, 0.5, 0.8):
        fz = se.fit(se.TWO_N, Ns, Ln, n_starts=200, fix_main={0: a})
        P(f"  alpha frozen at {a}: huber={fz.objective:.4e} vs free {free.objective:.4e} -> frozen >= free: {fz.objective >= free.objective - 1e-12}")
    (RES / "validate_scaling_ext.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
