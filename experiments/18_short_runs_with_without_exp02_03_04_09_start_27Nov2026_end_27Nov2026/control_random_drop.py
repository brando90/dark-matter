#!/usr/bin/env python3
"""POST HOC control (not preregistered): is it these 11 runs, or would removing any 11 runs move the joint fit?

Uses the same fit as experiment 04's best fit (py_src.conjecture_fits.fit_joint, Huber delta 1e-3, 24 starts, seed 0) on
  (a) all 245 runs, (b) the 234 runs with D/N >= 1, (c) 200 random removals of 11 runs, (d) three structured removals of
  11 runs: the highest D/N, the smallest N, the largest N.
Writes results/control_random_drop.csv and results/control_random_drop_summary.json.
"""
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from py_src.conjecture_fits import fit_joint  # noqa: E402
from py_src.ladders import load_chinchilla_runs  # noqa: E402

N_DRAWS = 200
SEED = 20261127


def fit(df, label, kind, **extra):
    f = fit_joint(df["N"], df["D"], df["loss"], n_starts=24, seed=0)
    p = f.params
    return {"label": label, "kind": kind, "n_runs": len(df), "objective": f.objective, "success": f.success,
            "E": p["E"], "A": p["A"], "alpha": p["alpha"], "G": p["G"], "gamma": p["gamma"], **extra}


def main():
    runs = load_chinchilla_runs()
    n = len(runs)
    short = runs["ratio"] < 1.0
    assert n == 245 and int(short.sum()) == 11, (n, int(short.sum()))
    rows = [fit(runs, "all245", "all"), fit(runs[~short].reset_index(drop=True), "drop_DN_lt_1", "short")]
    for name, idx in [
        ("drop_11_highest_DN", runs["ratio"].nlargest(11).index),
        ("drop_11_smallest_N", runs["N"].nsmallest(11).index),
        ("drop_11_largest_N", runs["N"].nlargest(11).index),
    ]:
        rows.append(fit(runs.drop(index=idx).reset_index(drop=True), name, "structured"))
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for i in range(N_DRAWS):
        drop = rng.choice(n, size=11, replace=False)
        keep = np.setdiff1d(np.arange(n), drop)
        rows.append(fit(runs.iloc[keep].reset_index(drop=True), f"random_{i:03d}", "random",
                        n_short_removed=int(short.to_numpy()[drop].sum())))
        if (i + 1) % 25 == 0:
            print(f"random draw {i+1}/{N_DRAWS} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "control_random_drop.csv", index=False)
    rnd = df[df.kind == "random"]
    g_short = float(df.loc[df.label == "drop_DN_lt_1", "gamma"].iloc[0])
    g_all = float(df.loc[df.label == "all245", "gamma"].iloc[0])
    summ = {
        "n_draws": int(len(rnd)), "seed": SEED, "gamma_all245": g_all, "gamma_drop_DN_lt_1": g_short,
        "random_gamma_min": float(rnd.gamma.min()), "random_gamma_p10": float(rnd.gamma.quantile(0.10)),
        "random_gamma_median": float(rnd.gamma.median()), "random_gamma_p90": float(rnd.gamma.quantile(0.90)),
        "random_gamma_max": float(rnd.gamma.max()),
        "random_draws_gamma_below_0.40": int((rnd.gamma < 0.40).sum()),
        "random_draws_gamma_at_or_below_drop_DN_lt_1": int((rnd.gamma <= g_short).sum()),
        "random_objective_min": float(rnd.objective.min()), "random_objective_median": float(rnd.objective.median()),
        "random_objective_max": float(rnd.objective.max()),
        "objective_all245": float(df.loc[df.label == "all245", "objective"].iloc[0]),
        "objective_drop_DN_lt_1": float(df.loc[df.label == "drop_DN_lt_1", "objective"].iloc[0]),
        "random_draws_failed": int((~rnd.success).sum()),
        "random_draws_removing_k_short_runs": {str(k): int(v) for k, v in rnd.n_short_removed.value_counts().sort_index().items()},
        "structured": {r.label: {"gamma": float(r.gamma), "objective": float(r.objective), "alpha": float(r.alpha), "E": float(r.E)}
                       for r in df[df.kind == "structured"].itertuples()},
    }
    (out / "control_random_drop_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
