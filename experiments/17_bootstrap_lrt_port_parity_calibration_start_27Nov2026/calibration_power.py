#!/usr/bin/env python3
"""Criteria 3 and 4 of the folder 17 README: null calibration and power of the public lr_test with fresh seeds.

Data come from DM-1's generator (dm1.law, E0 = 1.69, alpha0 = 0.34, N_REF = 1e6) with new seeds [cell, rep, 20261127];
the bootstrap uses [cell, rep, 20261127, 1]. Every ladder is run through py_src.scaling_laws_fit.lr_test (B = 200,
DM-1b's exact early stop).

    python calibration_power.py --workers 2          # run (resumable), then summarize
    python calibration_power.py --summarize-only
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "dm1_power_analysis"))
import dm1  # noqa: E402
from py_src.scaling_laws_fit import lr_test  # noqa: E402

SEED = 20261127
CELLS = [  # name, window, points per decade, noise sd, relative size s of the second term, DM-1 beta, reps, role
    dict(id=0, name="null_1e5_1e9_4pd_sd0.01", nmin=1e5, nmax=1e9, ppd=4, sigma=0.01, s=0.0, beta=1.0, reps=200, role="null"),
    dict(id=1, name="null_1e6_1e9_4pd_sd0.01", nmin=1e6, nmax=1e9, ppd=4, sigma=0.01, s=0.0, beta=1.0, reps=200, role="null"),
    dict(id=2, name="null_1e7_1e9_4pd_sd0.02", nmin=1e7, nmax=1e9, ppd=4, sigma=0.02, s=0.0, beta=1.0, reps=200, role="null"),
    dict(id=3, name="power_control_1e5_1e9_8pd_sd0.005_s1", nmin=1e5, nmax=1e9, ppd=8, sigma=0.005, s=1.0, beta=1.0, reps=40, role="power"),
    dict(id=4, name="power_weak_1e7_1e9_4pd_sd0.01_s0.1", nmin=1e7, nmax=1e9, ppd=4, sigma=0.01, s=0.1, beta=1.0, reps=40, role="weak"),
]
GATE_POOLED, GATE_CELL, GATE_POWER = 0.07, 0.10, 36  # README criteria 3 and 4, written before the first run


def one(job):
    c, rep = job
    N = dm1.ladder(c["nmin"], c["nmax"], c["ppd"])
    n = N / dm1.N_REF
    y = dm1.law(n, dm1.E0, dm1.A_SCALED, dm1.ALPHA0, c["s"] * dm1.A_SCALED, c["beta"]) + \
        np.random.default_rng([c["id"], rep, SEED]).normal(0, c["sigma"], len(n))
    t0 = time.time()
    r = lr_test(N, y, B=200, level=0.05, early_stop=True, rng=np.random.default_rng([c["id"], rep, SEED, 1]))
    return dict(cell=c["id"], name=c["name"], role=c["role"], rep=rep, npts=len(n), T=r.T, k_exceed=r.k_exceed,
                B_used=r.B_used, p_value=r.p_value, detect=int(r.detect), sigma_hat=r.sigma_hat,
                collapsed=int(r.two_term.collapsed), alpha_fast=r.two_term.alpha, beta_slow=r.two_term.beta,
                nonfinite_boot=r.n_nonfinite_boot, seconds=round(time.time() - t0, 3))


def ci(k, n):
    r = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [round(r.low, 4), round(r.high, 4)]


def summarize(df):
    cells = {}
    for c in CELLS:
        d = df[df.cell == c["id"]]
        cells[c["name"]] = dict(role=c["role"], n=len(d), detections=int(d.detect.sum()),
                                rate=round(float(d.detect.mean()), 4), exact_95ci=ci(d.detect.sum(), len(d)),
                                npts=int(d.npts.iloc[0]) if len(d) else None, median_T=float(d["T"].median()),
                                nonfinite_boot_ladders=int((d.nonfinite_boot > 0).sum()),
                                mean_seconds=round(float(d.seconds.mean()), 3))
    nul = df[df.role == "null"]
    pw = df[df.role == "power"]
    pooled = float(nul.detect.mean())
    worst = max(v["rate"] for v in cells.values() if v["role"] == "null")
    out = dict(cells=cells,
               null_pooled=dict(n=len(nul), detections=int(nul.detect.sum()), rate=round(pooled, 4),
                                exact_95ci=ci(nul.detect.sum(), len(nul))),
               null_worst_cell_rate=worst,
               power_control=dict(n=len(pw), detections=int(pw.detect.sum()), exact_95ci=ci(pw.detect.sum(), len(pw))),
               gates=dict(pooled_max=GATE_POOLED, cell_max=GATE_CELL, power_min_detections=GATE_POWER),
               criterion_3_pass=bool(pooled <= GATE_POOLED and worst <= GATE_CELL),
               criterion_4_pass=bool(pw.detect.sum() >= GATE_POWER),
               numpy=np.__version__, python=sys.version.split()[0], seed=SEED)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / "calibration.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    have = set(zip(done.cell, done.rep)) if len(done) else set()
    jobs = [(c, r) for c in CELLS for r in range(c["reps"]) if (c["id"], r) not in have]
    if jobs and not args.summarize_only:
        print(f"{len(have)} ladders done, {len(jobs)} to run on {args.workers} workers", flush=True)
        t0 = time.time()
        with Pool(args.workers) as pool:
            new = []
            for k, rec in enumerate(pool.imap_unordered(one, jobs, chunksize=2)):
                new.append(rec)
                if (k + 1) % 50 == 0:
                    print(f"{k + 1}/{len(jobs)} ladders, {time.time() - t0:.0f}s", flush=True)
                    pd.concat([done, pd.DataFrame(new)]).to_csv(path, index=False)
        done = pd.concat([done, pd.DataFrame(new)]).sort_values(["cell", "rep"]).reset_index(drop=True)
        done.to_csv(path, index=False)
    s = summarize(done)
    (out / "calibration_summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))
    sys.exit(0 if (s["criterion_3_pass"] and s["criterion_4_pass"]) else 1)


if __name__ == "__main__":
    main()
