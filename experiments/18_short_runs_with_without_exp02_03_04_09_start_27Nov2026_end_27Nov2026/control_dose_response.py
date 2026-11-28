#!/usr/bin/env python3
"""POST HOC control (not preregistered): remove the k runs with the lowest tokens-per-parameter ratio, k = 0 to 15, and refit
the joint law with experiment 04's best-fit settings (Huber delta 1e-3, 24 starts, seed 0). k = 5 is the set that the
Besiroglu et al. replication drops as unexplained high-loss outliers (240 of 245 points). Writes results/control_dose_response.csv.
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from py_src.conjecture_fits import fit_joint  # noqa: E402
from py_src.ladders import load_chinchilla_runs  # noqa: E402

runs = load_chinchilla_runs().sort_values("ratio").reset_index(drop=True)
rows = []
for k in range(0, 16):
    keep = runs.iloc[k:].reset_index(drop=True)
    f = fit_joint(keep["N"], keep["D"], keep["loss"], n_starts=24, seed=0)
    p = f.params
    rows.append({"k_removed": k, "n_runs": len(keep), "largest_removed_ratio": float(runs.ratio.iloc[k - 1]) if k else float("nan"),
                 "objective": f.objective, "E": p["E"], "A": p["A"], "alpha": p["alpha"], "G": p["G"], "gamma": p["gamma"], "success": f.success})
    print(rows[-1], flush=True)
(HERE / "results").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(HERE / "results" / "control_dose_response.csv", index=False)
