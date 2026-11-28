#!/usr/bin/env python3
"""POST HOC robustness check (added after criteria 1 to 5 were evaluated; not preregistered).

DM-1b's test assumes additive Gaussian noise of constant size on the loss. Real ladders are closer to multiplicative: a
loss near 10 at the small end and near 2 at the large end does not wobble by the same number of nats. This repeats the
null part of calibration_power.py with log-normal multiplicative noise, y = law * exp(Normal(0, rel_sd)), on the same
three windows, 200 one-term ladders each, and reports the rejection rate of the public lr_test (B = 200, early stop).

    python posthoc_multiplicative_noise.py --workers 2
Writes results/posthoc_multiplicative.csv and results/posthoc_multiplicative_summary.json.
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
REL_SD = 0.005  # about 0.01 nats where the loss is 2 and 0.05 nats where it is 10
CELLS = [
    dict(id=10, name="mult_null_1e5_1e9_4pd", nmin=1e5, nmax=1e9, ppd=4),
    dict(id=11, name="mult_null_1e6_1e9_4pd", nmin=1e6, nmax=1e9, ppd=4),
    dict(id=12, name="mult_null_1e7_1e9_4pd", nmin=1e7, nmax=1e9, ppd=4),
]
REPS = 200


def one(job):
    c, rep = job
    N = dm1.ladder(c["nmin"], c["nmax"], c["ppd"])
    n = N / dm1.N_REF
    clean = dm1.law(n, dm1.E0, dm1.A_SCALED, dm1.ALPHA0)
    y = clean * np.exp(np.random.default_rng([c["id"], rep, SEED]).normal(0, REL_SD, len(n)))
    r = lr_test(N, y, B=200, level=0.05, early_stop=True, rng=np.random.default_rng([c["id"], rep, SEED, 1]))
    return dict(cell=c["id"], name=c["name"], rep=rep, npts=len(n), loss_min=float(y.min()), loss_max=float(y.max()),
                T=r.T, k_exceed=r.k_exceed, B_used=r.B_used, p_value=r.p_value, detect=int(r.detect))


def ci(k, n):
    r = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [round(r.low, 4), round(r.high, 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out = HERE / "results"
    jobs = [(c, r) for c in CELLS for r in range(REPS)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        df = pd.DataFrame(list(pool.imap_unordered(one, jobs, chunksize=4))).sort_values(["cell", "rep"])
    df.to_csv(out / "posthoc_multiplicative.csv", index=False)
    cells = {}
    for c in CELLS:
        d = df[df.cell == c["id"]]
        cells[c["name"]] = dict(n=len(d), detections=int(d.detect.sum()), rate=round(float(d.detect.mean()), 4),
                                exact_95ci=ci(d.detect.sum(), len(d)), npts=int(d.npts.iloc[0]),
                                loss_range=[round(float(d.loss_min.min()), 2), round(float(d.loss_max.max()), 2)])
    summary = dict(rel_sd=REL_SD, cells=cells,
                   pooled=dict(n=len(df), detections=int(df.detect.sum()), rate=round(float(df.detect.mean()), 4),
                               exact_95ci=ci(df.detect.sum(), len(df))),
                   seconds=round(time.time() - t0, 1), numpy=np.__version__, python=sys.version.split()[0], seed=SEED,
                   label="post hoc, not preregistered")
    (out / "posthoc_multiplicative_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
