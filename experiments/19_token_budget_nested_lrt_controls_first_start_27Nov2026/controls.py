"""Folder 19 controls: calibration and power of the nested joint-law LRT on the real Porian design grids.

For each dataset (rw, owt2) the synthetic responses use exactly the real (N, D) points and the real per-point noise sd
(results/design_<ds>.csv). Truth is folder 14's: y = 2.0 + 3.0 (N/1e6)^-0.6 + 1.0 (D/1e9)^-0.2 (additive Chinchilla form).
Controls (20 replicates each, B = 200 bootstrap draws, detect iff p <= 0.05):
  null          no extra term                                       GATE G1: detections <= 4 of 20
  fast03        + 0.3 * (N / 5173248)^-1.4 (folder 14's planted term) GATE G2: detections >= 16 of 20
  fast01        + 0.1 * (N / 5173248)^-1.4                           GATE G3: detections >= 16 of 20
  corr_noise    null truth; noise shared within a width, rho = 0.5   report only (stress: independence assumption)
  dterm_misfit  null truth + B2 d^-1.0, B2 = 0.1 * min(d)            report only (stress: misspecified data term)
Resumable: rows already in results/controls_<ds>.csv are skipped. Usage: python3 -I controls.py [--workers 2] [--only gates|stress|all]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from py_src.nested_joint_lrt import Design  # noqa: E402

SEED, REPS, B = 20261127, 20, 200
DATASETS = ("rw", "owt2")
GATES = ("null", "fast03", "fast01")
STRESS = ("corr_noise", "dterm_misfit")
ALL = GATES + STRESS
N5 = 5173248.0
FAST = {"fast03": 0.3, "fast01": 0.1}
RHO = 0.5


def truth(N, D):
    return 2.0 + 3.0 * (N / 1e6) ** -0.6 + 1.0 * (D / 1e9) ** -0.2


def synthetic(des_df, control, rng):
    N, D, sd, w = des_df.N_tied.values, des_df.D.values, des_df.sd.values, des_df.width.values
    y = truth(N, D)
    z = rng.standard_normal(len(N))
    noise = sd * z
    if control in FAST:
        y = y + FAST[control] * (N / N5) ** -1.4
    if control == "corr_noise":
        u = {wi: rng.standard_normal() for wi in np.unique(w)}
        noise = sd * (np.sqrt(RHO) * np.array([u[wi] for wi in w]) + np.sqrt(1 - RHO) * z)
    if control == "dterm_misfit":
        y = y + 0.1 * (D / 1e9).min() * (D / 1e9) ** -1.0
    return y + noise


def one(task):
    ds, control, rep = task
    des_df = pd.read_csv(os.path.join(HERE, "results", f"design_{ds}.csv"))
    ds_i, c_i = DATASETS.index(ds), ALL.index(control)
    y = synthetic(des_df, control, np.random.default_rng([SEED, ds_i, c_i, rep]))
    design = Design(des_df.N_tied.values, des_df.D.values, des_df.sd.values)
    t0 = time.time()
    r = design.bootstrap_lrt(y, B=B, rng=np.random.default_rng([SEED, ds_i, c_i, rep, 1]))
    f1, f2 = r["fit1"], r["fit2"]
    gamma_fast = f2.exps[2]
    fast5 = f2.coef[3] * (N5 / 1e6) ** -gamma_fast if f2.rss < f1.rss else 0.0
    return dict(dataset=ds, control=control, rep=rep, T=r["T"], p=r["p"], detect=int(r["detect"]), B=B,
                sigma_hat=r["sigma_hat"], m1_alpha=f1.exps[0], m1_beta=f1.exps[1], m2_alpha=f2.exps[0],
                m2_beta=f2.exps[1], m2_gamma=gamma_fast, m2_C=f2.coef[3], fast_at_5M=fast5, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", choices=["gates", "stress", "all"], default="all")
    a = ap.parse_args()
    controls = {"gates": GATES, "stress": STRESS, "all": ALL}[a.only]
    done = set()
    for ds in DATASETS:
        path = os.path.join(HERE, "results", f"controls_{ds}.csv")
        if os.path.exists(path):
            done |= {(ds, c, int(r)) for c, r in pd.read_csv(path)[["control", "rep"]].itertuples(index=False)}
    tasks = [(ds, c, r) for c in controls for ds in DATASETS for r in range(REPS) if (ds, c, r) not in done]
    print(f"{len(tasks)} replicates to run, {len(done)} already done", flush=True)
    with Pool(a.workers) as pool:
        for row in pool.imap_unordered(one, tasks):
            path = os.path.join(HERE, "results", f"controls_{row['dataset']}.csv")
            pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(row["dataset"], row["control"], row["rep"], "p=%.4f" % row["p"], "T=%.2f" % row["T"], flush=True)


if __name__ == "__main__":
    main()
